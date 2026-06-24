# E0_v1_official 训练结果 — vision-blind 修复实验 (❌ 未修好)

> 时间：2026-06-21 (训练) ~ 2026-06-22 (离线视觉消融判定)
> 硬件：volc 8×A100-80GB (训练) · gf0 1×A100-80GB (离线消融)
> 启动命令：
> ```bash
> # volc 8×A100, 完全按官方配方
> torchrun --standalone --nproc_per_node=8 train_scripts/xvla/launch/xvla_train.py \
>   --config E0_v1_official \
>   --output_dir xvla/ckpts/xvla_e0_v1_official --workers 4
> ```
> Plan：[`future_plans/plans/xvla_proprio_shortcut_openloop_fix.md`](../../future_plans/plans/xvla_proprio_shortcut_openloop_fix.md) (E0)

---

## 0. 一句话结论

**E0 失败。换成真实 `action≠state` 数据 (v1) + 完全按官方配方训练, 但保留 `use_proprio=True` → 模型仍 100% vision-blind: 三路相机全置黑, 完整 20D 动作输出只变 `1.2e-5` (浮点噪声), 视觉/本体影响比 = `0.000`。** 与之前 p0/d5anchor (`action≡state` 数据) 的失明签名逐位一致。

→ 这推翻了 plan §4.4 的判断"E0 (真实 action≠state) 是必需主路径、足以唤起视觉"。**真实动作数据是必要条件, 但不充分**: 折叠任务准静态 → 未来 2s 的 EE 位姿仍可由当前 EE6D proprio 高度预测, 模型在 `use_proprio=True` 下照样从 proprio 低 loss 回归出 action anchors, 绕开视觉。**数据链 + 架构链需同时切断。**

---

## 1. 实验设定

| 参数 | 值 |
|---|---|
| config_name | `E0_v1_official` |
| init | `xvla/xvla_ckpts` (lerobot XVLA base, Florence2) |
| dataset | `A_v1_noRelabel_ee6d` 6 日期 (04-23..04-30), 639 ep / static-skip 后 581,713 sample, domain_id=20, **action≠state** (真实 teleop, 非 relabel) |
| use_proprio | **True** (保留 proprio — 这是与 E1 的关键差异, 也是失败主因) |
| param_groups | 4group_official (vlm & soft_prompt ×0.1; transformer_core & action_head ×1.0); VLM 10×LR bug 已修 |
| freeze | 前 1000 步冻 vlm+transformer_core |
| steps / bs / 卡 | 50,000 / per-device 16 × 8 = eff 128 |
| lr / warmup / schedule | 1e-4 / 2000 / **constant** (官方 cosine 默认 OFF) |
| weight_decay | 0.0 |
| action 表示 | `action_qdur=2.0` (intention abstraction) |
| 图像 | ImageNet norm + ColorJitter(0.2) |
| 精度 | bf16 mixed |
| static_skip | True (丢未来首步双臂几乎不动的退化帧, 剔 21,529 帧) |

---

## 2. 训练曲线

**无 inline-eval MAE** — 该 trainer 只记 flow-matching loss + gnorm, 未启用 MAE 离线校验 (且 MAE/val-loss 本就测不出 vision-blind, 见 plan §3)。

| 指标 | 观察 |
|---|---|
| flow loss | 全程在 ~2–5 噪声带震荡 (无单调下降趋势, flow-matching 正常) |
| gnorm | 100–430, 偏高但未发散 |
| 结束 | step 49,950 正常落 `step_final` |

> 训练 loss 完全无法判别视觉是否被读 — 这正是必须跑离线视觉消融的原因。

---

## 3. ⭐ 离线视觉消融 (判定指标)

**问题**: 动作是否依赖相机图像, 还是 proprio 的纯函数 (开环)。
**方法**: 新写**数据集版消融** [`eval_xvla_vision_ablation_dataset.py`](../../../../train_scripts/kai/eval/eval_xvla_vision_ablation_dataset.py) — E0 无 trace 且 v1 数据集 `observation.state` 是 20D EE6D (非 14D 关节, trace 版用不了)。直接复用训练 `multi_domain_dataset` 预处理 (ImageNet norm 图 + 20D state + BART tokens + domain_id), 固定 flow-matching seed → 每次推理是 `(images, state)` 的确定函数。gf0 A100, date 2026-04-23, n=12。

| 扰动 | xyz (mm) | gripper |
|---|---|---|
| 换图 hold state (`d_img`) | **0.00** | 0.0000 |
| 整图置黑 hold state (`d_blank`) | **0.00** | 0.0000 |
| 换 state hold image (`d_state`) | 315.47 | 0.3171 |
| **视觉/本体影响比 `d_img/d_state`** | **0.000** | **0.000** |

### 3.1 已排除测试假象 (决定性核验)

为排除"图没喂进模型"的 harness bug, 单帧逐项核验:
- `config.image_features` = 我喂的三键 `observation.images.image/image2/image3` (全 VISUAL, `empty_cameras=0`) → 图像确被消费;
- 输入图 vs 黑图张量差 `max=2.12` → 图像确实被改变;
- **完整 20D 输出** (非仅 xyz/grip) 在三路全黑下 `max|Δ|=1.2e-5` (逐通道几乎全 0, 浮点噪声);
- 对照: state 扰动 0.1 → 输出 `max|Δ|=0.368` (大 3 万倍)。

→ 图像有路径、确被喂入、确被改变, 但对输出零贡献。**位级零 = 真·vision-blind, 非弱依赖, 非测不出。**

---

## 4. Checkpoint 路径

```
gf0:/vePFS/tim/workspace/deepdive_kai0/xvla/ckpts/xvla_e0_v1_official/
  ├── config.json
  ├── step_002000 .. step_048000/ (每 2000 一存)
  └── step_final/state_dict.pt        (3.52 GB, step≈50000, lerobot XVLAPolicy 格式)
本地 (已拉取 + 补 sidecar, 但失明不上机):
  /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_e0_v1_official_step_final/
  ├── state_dict.pt   (与远端字节一致 3519354117)
  └── sidecar.json    (image_norm=imagenet, deploy_domain_id=20, deploy_prompt)
```

---

## 5. 与对照实验的差异 / 结论

| 组 | 数据 action | use_proprio | `d_img` | 结论 |
|---|---|---|---|---|
| p0 / d5anchor | ≡state | True | 0.03–0.08mm | 失明 (数据捷径 + proprio) |
| **E1** (旧实验, 2026-06-10) | ≡state | **False** | 0.00mm | 失明 (转常量开环, 断架构链不够) |
| **E0** (本实验) | **≠state (v1)** | True | **0.00mm** | **失明 (断数据链不够, 保留 proprio 仍开环)** |

**关键推论**: E1 (断架构链) 和 E0 (断数据链) **各自单独都不够**。下一步必须**两者叠加 = v1 真实数据 + `use_proprio=False`** (新 config `E1_v1_official`), 这是当前唯一未被证伪的路径。详见 plan §4.5 + §6。

- ❌ 不上真机 (失明)。
- ⭐ 下一步: 提交 `E1_v1_official` (E0 配方 + `use_proprio=False`), 训完过 §3 同款数据集消融门禁 (`d_img` 绝对值 ≫ 0 才放行)。
