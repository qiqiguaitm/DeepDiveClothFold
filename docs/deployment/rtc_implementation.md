# RTC (Real-Time Chunking) 实现方案

> 现行 autonomy 栈中 π₀.5 + RTC 的端到端实现说明: 模型层引导 + 运行时双时钟流水
> + 调参入口. 适用于 sim01 部署 (`./start_scripts/start_autonomy.sh`).

## 1. 设计目标与命名

**问题**: chunk-based action 预测 (π₀ / π₀.5) 每次推理输出长度 `action_horizon` 的
完整轨迹 chunk; 但实际机械臂以 publish_rate 高频消费. 推理 (~333 ms) 比 publish
间隔 (33 ms) 慢 10×, 因此 chunk N 还没消费完 chunk N+1 就到了 — 朴素替换会在
chunk 边界产生不连续动作.

**RTC (Real-Time Chunking)** = 让模型在生成新 chunk 时, 显式被引导朝向上一次
chunk 的延迟区间 `[d, exec_h)` — 把"连续性"作为约束注入采样过程, 而不是事后插值.

论文参考: *Real-Time Action Chunking with Large Models* (kinetix). 本仓中实现位于
`kai0/src/openpi/models/pi0_rtc.py`, 由 `policy_inference_node.py` 在运行时调用.

## 2. 两层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Model layer — pi0_rtc.py::Pi0RTC.sample_actions                │
│  (RTC guidance 在 sample_actions 的去噪迭代内部生效)            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ 50-step chunk (Pi0Config.action_horizon=50)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Runtime layer — policy_inference_node.py                       │
│  • _inference_loop: 3 Hz 后台线程 → policy.infer()              │
│  • integrate_new_chunk: 把新 chunk 与残余做线性叠加平滑          │
│  • _publish_action: 30 Hz 定时器 → /master/joint_left|right     │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ /master/joint_* (sensor_msgs/JointState)
                          ▼
            arm_reader_node (mode=1) → CAN → 机械臂
```

两层正交且互补:
- **模型层 RTC** 解决"模型预测的 chunk N+1 与 chunk N 在数学层面尽量贴合".
- **运行时线性平滑** 解决"执行时刻 buffer 切换的尾过渡", 是兜底.

## 3. 双时钟

| 时钟 | 频率 | 默认值 | 控制源 |
|---|---|---|---|
| **publish_rate** (执行) | 30 Hz | 33.33 ms/step | `policy_inference_node.py:283` |
| **inference_rate** (推理) | 3.0 Hz | 333.33 ms/cycle | `policy_inference_node.py:284` |
| **chunk_size** | 50 actions | π₀ `action_horizon` 默认 (`pi0_config.py:27`) | `policy_inference_node.py:285` |

比值 = `publish_rate / inference_rate = 10` — 每 10 个 publish step (≈1 chunk 的 1/5)
推理就出一个新 chunk. 这是 RTC 假设的"replan cadence".

## 4. StreamActionBuffer + 线性平滑

每个新 chunk 从模型回来后, 通过 `StreamActionBuffer.integrate_new_chunk(actions, max_k, min_m)`
合并到当前执行队列 `cur_chunk`:

1. **裁前 (latency 补偿)**: `drop_n = min(self.k, latency_k=8)` — 把新 chunk
   头 8 步丢掉. 理由: 推理用的图片是 ~333 ms 前的, 那几步动作已被旧 chunk 同时消费完了.
2. **overlap 平滑**: 与 `cur_chunk` 残余 (取 `min(old, new)` 步) 线性渐变叠加:
   - 首帧 100 % 旧 / 0 % 新 → 末帧 0 % 旧 / 100 % 新
   - 实际窗口至少 `min_smooth_steps=8` 步
3. **k = 0 重置**: 下个 `_publish_action` tick 从新合并 chunk 头开始 pop.

参数 (ros2 param, 都可热改):

| 参数 | 默认 | 含义 |
|---|---|---|
| `latency_k` | 8 | 头部裁剪步数 (≈ infer_period × publish_rate) |
| `min_smooth_steps` | 8 | overlap 平滑窗最小长度 |
| `decay_alpha` | 0.25 | legacy, 线性叠加里实际未用 |

## 5. 模型层 RTC guidance (pi0_rtc.py)

`Pi0RTC.sample_actions(...)` 接收 4 个额外参数:

| 参数 | 类型 | 来源 | 默认 |
|---|---|---|---|
| `prev_action_chunk` | `(b, ah, ad)` float32 | 运行时把上一次返回的 chunk normalize 后传入 | None → fallback 到 base_step |
| `inference_delay` | int | `round(last_infer_ms / 1000 × publish_rate)` 实测换算 | 0 |
| `execute_horizon` | int | ros2 param `rtc_execute_horizon` | 16 |
| `max_guidance_weight` | float | ros2 param `rtc_max_guidance_weight` | 0.5 |
| `enable_rtc` | bool | ros2 param `enable_rtc` | true |

### 5.1 guidance 数学 (节选)

去噪 step 内, 用 `get_prefix_weights(d, exec_h, action_horizon, schedule='exp')`
生成长度 `action_horizon` 的权重向量, 仅在 **延迟尾窗** `[d, exec_h)` 内非零, 其余为 0
(超过 exec_h 的步完全自由).

```python
exec_h = clamp(execute_horizon, 1, action_horizon)
d      = clamp(inference_delay, 0, action_horizon)
weights = get_prefix_weights(d, exec_h, action_horizon, "exp")  # → shape (ah,)

# 每步去噪
x_1, vjp_fun, v_local = jax.vjp(denoiser, x_t, has_aux=True)
error = (prev_chunk - x_1) * weights[None, :, None] * dim_mask     # 仅 [d, exec_h) 内有 error
pinv_correction = vjp_fun(error)[0]                                 # 反传到 x_t 空间
# 用 max_guidance_weight 上限把 correction 按时间反相加回 v_local
```

直觉:
- `d` = 已经被消费掉、来不及改的"过去前缀" — 跳过.
- `exec_h` = 想引导的尾窗终点 — 默认 16 ≈ `2 × latency_k`, 覆盖 chunk 切换边界 + 几步余量.
- `max_guidance_weight = 0.5` = guidance correction 的绝对幅度上限, 防止把模型预测彻底拉死成 prev_chunk 的复刻.
- `prefix_attention_schedule = "exp"` = 离 `d` 越近权重越大, 平滑过渡.

### 5.2 维度对齐 (R1 fix)

`prev_action_chunk` 在客户端是 **14-dim 原始 joint 空间**, 但模型内部是 **32-dim
归一化空间**. 运行时:
1. 用 `norm_stats.json` 的 `mean / std` 把 14 维归一化.
2. Pi0RTC 内部把 14 → 32 用 0 padding, 同时 `dim_mask = (jnp.arange(ad) < 14)`
   把 padded 维从 guidance error 中屏蔽 (`pi0_rtc.py:320-321`).
3. 所以 RTC 只引导真实 14 维, padded 0 不会拉错 chunk.

代码 `policy_inference_node.py:1898-1901`.

## 6. 运行时数据流

```
       inference_thread (3 Hz)              publish_timer (30 Hz)
       ─────────────────────                ──────────────────────
       _get_observation()                   pop_next_action()
            │                                     │
       attach prev_action_chunk             jump-protection check
       (normalized 14-dim)                       │
       inference_delay                      gripper_offset 校正
       execute_horizon=16                        │
       max_guidance_weight=0.5              publish to /master/joint_left|right
            │                                     │
       policy.infer(obs)                     [arm_reader → CAN → arm]
            │
       50-step chunk
            │
       _rtc_prev_chunk = chunk         (snapshot 给下一次 inference)
            │
       integrate_new_chunk(             (与 cur_chunk 做线性平滑)
         chunk,
         max_k=latency_k=8,
         min_m=min_smooth_steps=8
       )
            │
       sleep(period_live)               (= 1/inference_rate, 动态读 param)
```

竞态保护 (`policy_inference_node.py:1930-1940`): 推理回调路径上检查
`self._replay_mode == 'replay'` — 若 replay 启用且 buffer 已被填入整段录制 episode,
丢弃本次 policy chunk, 避免 `integrate_new_chunk` 把整段 episode 当成"旧 chunk"
覆盖掉.

## 7. 调参入口

### 7.1 通过 ros2 param (热改, 即时生效)

```bash
ros2 param set /policy_inference enable_rtc              {true,false}
ros2 param set /policy_inference rtc_execute_horizon     <int>     # 默认 16
ros2 param set /policy_inference rtc_max_guidance_weight <float>   # 默认 0.5
ros2 param set /policy_inference inference_rate          <float Hz># 默认 3.0
ros2 param set /policy_inference latency_k               <int>     # 默认 8
ros2 param set /policy_inference min_smooth_steps        <int>     # 默认 8
ros2 param set /policy_inference decay_alpha             <float>   # 默认 0.25
```

### 7.2 通过 `rtc_apply.sh` 预设 (快捷方式)

`start_scripts/rtc_apply.sh` 把 7 个参数打包成几个常用组合:

| preset | enable_rtc | exec_h | max_guid | infer_rate | lat_k | smooth | 用途 |
|---|---|---|---|---|---|---|---|
| `off` | false | 16 | 0.5 | 3.0 | 8 | 8 | 纯线性平滑, RTC 关. A/B 对照 |
| `on` (默认) | true | 16 | 0.5 | 3.0 | 8 | 8 | 默认: RTC + 中度平滑 |
| `rtc_tight` | true | 12 | 0.8 | 10.0 | 3 | 3 | 快变场景: 高频 replan + 短引导窗 |
| `rtc_long` | true | 50 | 0.5 | 3.0 | 8 | 8 | A/B 对照: 全 horizon 引导, 最强连续性 / 最弱响应 |
| `rtc_paper` | true | 25 | 0.5 | 3.0 | 8 | 8 | Paper Table 4 conservative |
| `rtc_paper_strong` | true | 25 | **5.0** | 3.0 | 8 | 8 | Paper Table 4 numeric (10× 强引导, ⚠ 单位存疑) |
| `show` | — | — | — | — | — | — | 仅打印当前值 |

用法:
```bash
./start_scripts/start_autonomy.sh                  # 在终端 A 起栈
./start_scripts/rtc_apply.sh                       # 终端 B: 查看当前
./start_scripts/rtc_apply.sh rtc_tight              # 切到 tight 预设
```

### 7.3 通过 launch 参数 (启动时一次性)

```bash
ros2 launch piper autonomy_launch.py \
    enable_rtc:=true \
    rtc_execute_horizon:=20 \
    rtc_max_guidance_weight:=0.8 \
    [其他参数...]
```

`autonomy_launch.py:148-156` 声明这三个 launch arg, 转发到 `/policy_inference`.

## 8. 失效与降级行为

| 情况 | 行为 |
|---|---|
| `enable_rtc=false` | `prev_action_chunk` 不传入, `sample_actions` 走 `base_step` (纯标准去噪). 完全等价于关 RTC. 运行时线性平滑仍生效. |
| 第一次推理 (cold start) | `_rtc_prev_chunk = None` → 不传 prev_action_chunk → 自动 fallback 到 base_step. |
| `observe → execute` 切换 | `_flush_stale_buffer` 同时 `_rtc_prev_chunk = None` (`policy_inference_node.py:501-502`), 第一拍重新冷起. |
| `replay_mode=replay` 切换瞬间 | 本次 policy chunk 整段丢弃 (避免覆盖已填入的录制 episode buffer), 但 `_rtc_prev_chunk` 仍快照保存供下次. |
| jump protection 触发 | `cur_chunk` flush + `_last_published_action = None`, RTC 不影响 (`_rtc_prev_chunk` 不变). |

## 9. 验证 / 监控

- **inference 延迟**: log 行 `infer XXXms | chunk=(50,14) | L[0]=... R[0]=...` 在
  `_inference_loop` 末尾打印, 同时通过 Float32MultiArray 发到
  `/timeseries/inference_ms` 给 Rerun.
- **chunk diversity (RTC 强度 sanity)**: 比较连续两次 chunk 的 MAE 与 baseline
  (`enable_rtc=false`) — 若 RTC 显著降低 chunk-to-chunk MAE 且仍保持
  end-effector 任务性能, 说明 guidance 在起作用而非死灯泡.
- **rerun 视图**: `pub_action_chunk` 把完整 50 步 chunk 发为
  `Float32MultiArray`, rerun_viz_node 渲染整段预测轨迹 — 可视上看 chunk
  切换边界是否平滑.

## 10. 文件索引

| 文件 | 角色 |
|---|---|
| `kai0/src/openpi/models/pi0_rtc.py` | Pi0RTC 模型类 + sample_actions + get_prefix_weights |
| `kai0/src/openpi/models/pi0_config.py` | action_horizon=50, action_dim=32 默认 |
| `kai0/src/openpi/training/config.py:1316` | `pi05_flatten_fold_normal` 配置 (本仓默认) |
| `ros2_ws/src/piper/scripts/policy_inference_node.py` | 运行时调度 + StreamActionBuffer + RTC 注入 |
| `ros2_ws/src/piper/launch/autonomy_launch.py` | 启动 wrapper + launch 参数声明 |
| `start_scripts/start_autonomy.sh` | 一键启动脚本 (设置 XLA_FLAGS + GPU 选择 + ros2 launch) |
| `start_scripts/rtc_apply.sh` | 运行时 preset 切换器 |

## 11. 已知问题 / 后续

- **`mask_prefix_delay`** 在 `sample_actions` 签名里, 但运行时不转发 — 因为它
  传到 JIT boundary 内会变成 tracer, `if mask_prefix_delay:` 在 `pi0_rtc.py:323`
  报 `TracerBoolConversionError`. 现用函数默认 `False`. 见
  `policy_inference_node.py:296-300` 注释.
- **`max_guidance_weight` 单位**: rtc_apply.sh 的 "paper default" 是 `0.5`, 但
  论文 Table 4 标 `5.0`. 可能归一化口径不同 (`unnormalized vs normalized`).
  `rtc_paper_strong` preset 提供 `5.0`, 真机上要从 `rtc_paper` (`0.5`)
  逐步上调验证, 避免一上来引导过强把 chunk 拉成 prev_chunk 复刻.
- **Blackwell SIGSEGV workaround**: `XLA_FLAGS=--xla_gpu_autotune_level=0`
  在 `start_autonomy.sh` 的 replay 分支和 autonomy 分支都已设置 (jax 0.5.3 +
  RTX 5090 sm_120 在 autotune 时会 SIGSEGV).
