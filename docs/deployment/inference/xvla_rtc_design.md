# X-VLA RTC 移植设计 (Real-Time Chunking → `generate_actions`)

> **状态**: 设计 (未实施)。**Gating**: 必须先经 E0 确认 X-VLA 真读视觉 (`eval_xvla_vision_ablation_offline.py` 视觉/本体比 0.000→≳0.5) 再实施 —— 否则是在给一个开环不看图的策略做平滑 (见 [`docs/training/future_plans/plans/xvla_proprio_shortcut_openloop_fix.md`](../../training/future_plans/plans/xvla_proprio_shortcut_openloop_fix.md))。
>
> **关系**: π₀.5 的 RTC 实现见 [`rtc_implementation.md`](rtc_implementation.md)。本文是把同一思想移植到 X-VLA flow-matching 的设计,**移植机制而非照搬代码** (两者去噪范式/动作空间/引导方式都不同,见 §2)。

---

## 1. 目标与现状

X-VLA 每次推理由 `generate_actions` 吐出整块 `(1, chunk, 20)` EE6D 动作,机械臂高频消费。推理 (~10 step flow-matching) 比 publish 慢 → chunk N 未消费完 chunk N+1 已到,朴素替换在 chunk 边界产生不连续动作 → 真机卡顿。

**当前 X-VLA 连续性手段 (非 RTC)**,见 `kai0/scripts/serve_policy_xvla.py`:
- **固定 flow-matching 噪声种子** (`--seed`): 同 obs 跨次输出确定,消除随机重采的 ~55mm chunk 间跳变。
- **预测式 proprio** (`proprio_feedback`) + `resync` 兜底: 用上一 chunk 末步当下次 proprio。

二者都是开环整块执行下的弱启发式,**不保证 chunk 边界与"上一块尚未执行完的尾部"对齐** —— 正是 RTC 要解决的。

**RTC** = 让模型在生成新 chunk 时,显式被引导朝向上一 chunk 的延迟尾区 `[d, exec_h)`,把连续性作为**约束注入采样过程**,而非事后插值。

---

## 2. 与 π₀ RTC 的关键差异 (决定移植方式)

| 维度 | π₀ (`pi0_rtc.py`) | X-VLA (`generate_actions`) |
|---|---|---|
| 去噪范式 | velocity-prediction (积分 ODE) | **x0-prediction** (每步直接预测 clean action,再重构 `x_t`) |
| 动作空间 | 32D **归一化** latent (z-score by norm_stats) | **20D 物理 EE6D**: xyz(米) / 6D 回转生值 / gripper logit。**无归一化** (`XYZ_SCALE=500`/`ROT_SCALE=10` 只是 loss 权重) |
| 引导实现 | `jax.vjp` 把 prefix error 反传到 `x_t` (pseudo-inverse correction) | **前向过程 inpaint** (见 §4),x0 范式下更简单且更稳 |
| 输出 decode | norm⁻¹ → 14D joint | EE6D → world16 `[xyz, quat_wxyz, grip]` → firmware **EndPoseCtrl (IK 在固件)** |

**推论 1 — RTC 必须在 EE6D 原生空间做**: X-VLA 没有独立"归一化 action latent"空间,flow 向量场就活在物理 EE6D。RTC 必须作用在向量场所在空间 ⇒ EE6D 与 EE pose 空间**重合**,"先转 action 空间再做 RTC"对 X-VLA 是伪命题。转 joint 空间更错: 需主机 IK (已删,是真机卡顿根因,见 [`xvla_inference_bringup.md`](xvla_inference_bringup.md))、且模型从不在 joint 空间去噪。

**推论 2 — x0 范式让前向 inpaint 极自然**: 每步 `x_t = x1·t + action·(1-t)` 由上一步 clean 预测重构,只要把 prefix 的 clean 目标钳成已知值,下一次重构自动把它带下去。

---

## 3. X-VLA `generate_actions` 现状 (插点定位)

`xvla/X-VLA/models/modeling_xvla.py`:

```python
x1 = randn(B, num_actions, 20)          # 噪声端
action = zeros_like(x1)
for i in range(steps, 0, -1):
    t   = i / steps
    x_t = x1*t + action*(1-t)           # ← RTC 插点: 在此对 prefix 做 inpaint
    proprio_m, x_t_m = action_space.preprocess(proprio, x_t)   # gripper 通道置零
    action = transformer(x_t_m, proprio_m, t, **enc)           # 预测 clean 20D
return action_space.postprocess(action)  # 仅对 gripper(9,19) 做 sigmoid
```

`EE6DActionSpace` (`action_hub.py`): `dim=20`, 位置 `(0,1,2)/(10,11,12)`, 回转 6D `(3..8)/(13..18)`, gripper `(9,19)`。`preprocess` 把 gripper 通道**置零喂入**;`postprocess` 末尾 sigmoid。

---

## 4. RTC 引导: 前向过程 inpaint (推荐) vs 算术平均 (否决)

设 `prev` = 上一 chunk 已承诺动作 (raw 20D EE6D,arm-base 系),`d` = 推理延迟期间已执行步数,`exec_h` = 引导尾窗终点,`w = get_prefix_weights(d, exec_h, chunk, "exp")` (与 π₀ 同:`[0,d)` 硬冻、`[d,exec_h)` exp 衰减、`≥exec_h` 自由)。

### 4.1 前向过程 inpaint (推荐)

在每步去噪重构 `x_t` 时,把 prefix 钳成"已知动作在时刻 t 的加噪值":

```python
known_noised = x1*t + prev*(1-t)             # 已知 prefix 走与训练同一前向过程
free_noised  = x1*t + action*(1-t)           # 模型自有预测的重构 (原逻辑)
x_t = w[:,None]*known_noised + (1-w[:,None])*free_noised   # 仅 prefix 受影响
# 之后照常 preprocess → transformer 预测 clean action
```

要点:
- **被钳的是噪声态 `x_t`,不是 clean 输出**。每步模型仍**重新预测整块 clean action**,返回值 `action` 始终是模型生成、落在训练可行流形上的动作。
- 连续性来自**条件化** (prefix 约束让模型把 suffix 续接上去),不是来自对输出做平均。
- gripper 通道反正被 `preprocess` 置零,**不参与 inpaint** (沿用现有 binarize/阈值)。

### 4.2 算术平均 (否决) —— 即"输出后混合"

```python
out = w*prev_clean + (1-w)*model_clean        # 直接平均两个 clean 20D 输出
```

这是把 RTC 退化成 ACT 式 temporal-ensemble:对**两个独立采样的 clean chunk 求平均**。

### 4.3 失败率差异 (本设计核心结论)

| | 前向过程 inpaint (§4.1) | 算术平均 / 输出后混合 (§4.2) |
|---|---|---|
| 返回给 decode/IK 的动作 | 模型 clean 输出 → **在可行流形上** | 两个 clean 的线性插值 → **可能离开流形** |
| **A. 6D→旋转矩阵** decode (`interleaved_6d_to_rotation_matrix`, Gram-Schmidt) | 输入是合法 6D → 始终合法 SO(3),不失败 | 平均两个合法 6D → reproject 仍合法但**非测地**,朝向摆过非预期姿态、角速度大 |
| **B. EE→关节 IK** (firmware EndPoseCtrl) | 模型只在可行轨迹上训练 → 可行性基本保住,**IK 失败率不显著升** | 可行集在 EE 空间非凸,中间 pose 不保证可行 → 撞工作空间/奇异/限位 → **IK 失败率升** |
| gimbal | 无 (deployment 发 quat 不发 Euler) | 无 |

**结论**: 用户担心的"EE6D 非线性 → 求逆失败率升高"**只对 §4.2 成立**。§4.1 因为生成全程过模型、返回值始终在流形上,A 类几乎不失败、B 类不显著升。**这反而是支持 in-loop EE6D + 前向 inpaint 的又一条论据。**

> 补充: 若实现上仍需对回转做软混合 (而非纯前向 inpaint),务必在 SO(3) 上 SLERP 或混后 re-orthonormalize,**不要在生 6D 上 lerp**;xyz 纯 Euclidean 可直接处理。

---

## 5. IK / decode 兜底 (多层)

| 层 | 机制 | 状态 |
|---|---|---|
| L0 引导上限 | `rtc_max_guidance_weight` 封顶,保留模型权威留在流形上 (默认从 0.5 起调) | 传输字段已存在 (`shm_transport.py`) |
| L1 冷启动/切换 | `prev=None` (首推、observe→execute 切换) → 跳过 inpaint,走标准去噪 | 复用现有 `_pred_proprio=None` 时机 |
| L2 proprio resync | pred-proprio 偏离实测 EE > 阈 → 回实测 | **已实现** (`serve_policy_xvla.py`) |
| L3 decode 跳变护栏 | EE6D→quat decode 后,quat 角跳 vs 上一已发 > 阈 → 该步降引导 / hold | **新增** |
| L4 IK 失败信号 | firmware EndPoseCtrl 返回 IK 失败/饱和 → 该步回退 (hold last feasible) + 下周期降引导 | **新增** (把 IK 失败当显式信号,而非静默饱和) |

L3/L4 是把"非线性求逆"的风险显式监控并降级,而非寄望它不发生。

---

## 6. 异步执行前置条件 (否则 RTC 为空操作)

RTC 的收益来自 `d>0` —— 推理延迟期间已执行的步数。**当前 X-VLA serve 是同步开环跑完整 chunk (`d=0`),RTC 退化成空操作。**

需把运行时改成**执行与下一次推理重叠** (双时钟,类比 π₀ 的 `policy_inference_node` 3Hz 推理 / 30Hz publish):
- 边执行当前 chunk 边后台推下一 chunk;新 chunk 回来时已执行 `d ≈ infer_ms × publish_rate / 1000` 步。
- `prev` = 上一 chunk 的 raw 20D EE6D (即 serve 已缓存的 `acts`,**转 world16 之前**,单位天然对齐),`acts[-1]` 已被 `_pred_proprio` 复用。
- X-VLA 客户端复用现有 `policy_inference_node --execution-mode ee_pose` (见 [memory: XVLA inference stack])。

---

## 7. 参数 (沿用 π₀ RTC 口径)

| 参数 | 含义 | 初值 |
|---|---|---|
| `enable_rtc` | 总开关 (false → 标准去噪,A/B 对照) | true |
| `inference_delay` `d` | `round(last_infer_ms/1000 × publish_rate)` 实测换算 | 动态 |
| `rtc_execute_horizon` `exec_h` | 引导尾窗终点 | ~2×d |
| `rtc_max_guidance_weight` | 引导幅度上限 | 0.5 起调 |
| `prefix_attention_schedule` | `[d,exec_h)` 衰减曲线 | "exp" |

---

## 8. 与现有连续性手段的取舍

- **固定噪声种子**: RTC 上线后**取代**它 —— RTC 靠构造给跨 chunk 一致性,prefix 之外反而想要*新鲜*噪声 (固定种子会降低 suffix 多样性)。建议 `enable_rtc=true` 时关固定种子。
- **预测式 proprio**: 正交 (conditioning 侧),保留;其 `acts[-1]` 缓存恰好给 RTC 提供 `prev` 来源。
- **运行时线性平滑** (若引入双时钟运行层): 作为 RTC 之外的执行端尾过渡兜底,与模型层 RTC 正交互补 (同 π₀ 两层架构)。

---

## 9. 实施顺序 (实施时)

1. **[gate]** E0 确认读视觉。未过则停。
2. `generate_actions` 增 `prev_action_chunk` / `d` / `exec_h` / `max_guidance_weight` 入参,实现 §4.1 前向 inpaint (默认关,等价原行为)。
3. 离线验证: 同 obs 下 RTC-on vs off 的 chunk-to-chunk MAE↓ 且任务轨迹不畸变;统计 decode/IK 失败率 (L3/L4 计数) 确认不升。
4. 运行层改异步双时钟 (§6),接 L3/L4 兜底。
5. 真机 A/B: RTC-on vs (固定噪声+预测proprio) baseline,看 FFT 主频 / 卡顿 (判别法见 [memory: 真机晃动=0.99Hz闭环])。

---

## 10. 代码索引

| 文件 | 角色 |
|---|---|
| `xvla/X-VLA/models/modeling_xvla.py::generate_actions` | RTC 插点 (§3 去噪环) |
| `xvla/X-VLA/models/action_hub.py::EE6DActionSpace` | 维度布局 / preprocess(gripper 置零) / postprocess(sigmoid) |
| `kai0/scripts/serve_policy_xvla.py` | `predict_action_chunk` 包装;`_ee6d_to_world8` (decode→quat wxyz);proprio_feedback/resync;固定种子 |
| `kai0/scripts/shm_transport.py` | `rtc_max_guidance_weight` 传输字段 (已存在) |
| `docs/deployment/inference/rtc_implementation.md` | π₀.5 RTC 参考实现 (get_prefix_weights / 双时钟 / 两层架构) |
