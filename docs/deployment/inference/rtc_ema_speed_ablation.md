# RTC / EMA 对模型执行速度影响 — 控制变量调研

> 目的: 量化部署管线各环节 (RTC guidance、EMA 发布平滑) 对机械臂**执行速度**的衰减, 找回"模型部署 ≈ 遥操 0.6× 手臂 / 0.3× 夹爪"里管线贡献了多少。
> 节点: `policy_inference_node.py` (launch 后节点名 `/policy_inference`, 两条链路 v0/v1 同一节点)
> 最近更新: 2026-07-19  |  状态: ⬜ 待真机采数

---

## 0. 背景 (已知结论)

- 实测模型自主执行速度 ≈ 遥操 **0.63× (手臂) / 0.30× (夹爪)** (dagger 受控 + base/inference 交叉双证)。
- V1 (websocket) 比 V0 (进程内) **更慢**, 机制: `inference_delay = round(往返延迟_s × publish_rate)` → 延迟越大, RTC 冻结/引导的 chunk 前缀越长 → 低通越强 → 速度越慢。
- 两个正交的降速环节, 本调研分别量化:
  1. **RTC guidance** (`enable_rtc`): chunk 接缝一致性引导, 重叠混合 = 低通, 压手臂速度。
  2. **EMA 发布平滑** (`publish_smooth_alpha` α<1): `cmd[t]=α·cmd+(1-α)·last_pub`, 加相位滞后 (夹爪 j6,j13 在节点内被排除, 故 EMA 主要影响手臂)。

---

## 1. 开关方法速查

### 1.1 RTC (可运行时热改 ✅)

| 方式 | 命令 | 说明 |
|---|---|---|
| **运行时热改 (推荐 A/B)** | `ros2 param set /policy_inference enable_rtc false` (关) / `true` (开) | 同一 session 内翻, 无需重启; 翻转会清 `_rtc_prev_chunk` 避免用陈旧引导 |
| 预设助手 | `./start_scripts/kai/rtc_apply.sh off` (纯平滑无引导) / `on` (JAX 默认) / `v1_default` | 一次设好 enable_rtc + exec_horizon + guidance_weight + rate + latency_k |
| v0 启动即关 | 启动命令追加 `enable_rtc:=false` (透传到 launch) | — |
| v1 启动即关 | `start_autonomy_from_ckpt_v1.sh <ckpt> --execute --no-rtc` | v1 内置 `--no-rtc` 开关 |
| 查当前值 | `ros2 param get /policy_inference enable_rtc` 或 `./start_scripts/kai/rtc_apply.sh show` | — |

相关子参数 (也可热改): `rtc_execute_horizon` (引导窗宽), `rtc_max_guidance_weight` (引导权重上限, 默认 0.5), `inference_rate`, `latency_k`。

### 1.2 EMA 发布平滑 (⚠ 不可热改, 改需重启节点)

`publish_smooth_alpha` 只在节点 init 读一次, **不在 `_on_set_parameters` 里** → `ros2 param set` 改它无效, **必须重启**。
- `α = 1.0` → EMA **关**  |  `α ∈ (0,1)` → EMA **开** (越小越平滑/越滞后)

| 链路 | 关 EMA (α=1.0) | 开 EMA (默认) |
|---|---|---|
| **v0** (`start_autonomy_from_ckpt.sh`) | `KAI0_PUBLISH_SMOOTH_ALPHA=1.0 ./start_scripts/kai/start_autonomy_from_ckpt.sh <ckpt> --execute` | 默认 `α=0.5` (直接启动) |
| **v1** (`start_autonomy_from_ckpt_v1.sh`) | 改 `start_scripts/kai/start_autonomy_v1.sh` 里 `publish_smooth_alpha:=0.7` → `1.0` 后启动 | 默认 `α=0.7` |

> 每次启动后用 `ros2 param get /policy_inference publish_smooth_alpha` **核实实际生效值** (v1 的 launch 硬编码可能盖过命令行透传)。

### 1.3 两链路默认状态

| | RTC | EMA α | inference_rate | 备注 |
|---|---|---|---|---|
| **v0** (进程内 JAX) | on | 0.5 | 3 Hz (jax_legacy) | 低延迟 → inference_delay 小 |
| **v1** (websocket Triton) | on | 0.7 | 20 Hz | 往返延迟大 → inference_delay 大 → 更慢 |

---

## 2. 实验设计 (控制变量)

### 2.1 因子 (2×2 主表 + 单因子扩展)

主实验固定链路 (先 v0 或 v1 选一条), 固定 ckpt / 场景 / prompt / speed_factor=1.0, 只翻 RTC 与 EMA:

| 组 | RTC | EMA | 启动方式 |
|---|---|---|---|
| **A0 基线全关** | off | off (α=1.0) | 关 RTC + α=1.0 |
| **A1 仅 RTC** | on | off (α=1.0) | 开 RTC (热改) + α=1.0 |
| **A2 仅 EMA** | off | on (α=0.5/0.7) | 关 RTC (热改) + 默认 α |
| **A3 全开 (默认)** | on | on | 出厂默认 |

- RTC 用 **运行时热改** → A0↔A1、A2↔A3 可在**同一次启动**内翻转, 消除启动间场景差异 (最干净)。
- EMA 需重启 → A0/A1 (α=1.0) 一次启动, A2/A3 (默认 α) 另一次启动。
- 每组重复 **≥3 段** 相同起始场景 (同一块布、同一摆放), 每段跑到同一里程碑 (如"抓起→展平"), 避免任务进度不同污染速度均值。

### 2.2 扩展单因子 (确认主项后再做)

- **RTC 强度**: `rtc_max_guidance_weight ∈ {0.0, 0.5, 5.0}` × `rtc_execute_horizon ∈ {6, 16, 50}` (rtc_apply 预设 `rtc_tight/rtc_long/rtc_paper`)。
- **EMA 强度**: `α ∈ {1.0, 0.7, 0.5, 0.3}` (每档重启)。
- **v0 vs v1 同 ckpt**: 验证延迟差 → 速度差 (配合 `--profile-latency` 记录往返延迟)。

---

## 3. 测量方法

### 3.1 速度指标 (主)

执行到的**关节速度** (rad/s), 由记录的 `observation.state` 逐帧差分算 (与数据集分析同口径):
- 手臂: 12 关节 dims `[0-5,7-12]` 的 `mean(|Δq|)·FPS`
- 夹爪: 2 dims `[6,13]` 的 `max(|Δq|)·FPS`
- 只统计**活动帧** (arm speed > 1e-3), 报 `mean / median / p90 / p99`
- 关键派生: **相对遥操倍率** (拿 base/v4 遥操分布做分母), 以及**组间比值** (A1/A0 等)

采数方式 (二选一):
- **A. `--trace`**: 启动加 `--trace` → server/client 逐帧落盘 obs/20D/16D/14D 到 `${KAI0_XVLA_LOG_DIR:-/tmp}/trace_<ts>/`, 用 state 列算速度。
- **B. 录 rosbag**: 录 `/joint_states` 或从臂反馈 topic, 离线算速度 (同 §3.1 公式)。

### 3.2 辅助指标

- **cycle 完成时间**: 同一里程碑 (抓起→展平) 的墙钟秒数 → 直接反映"整体快慢"。
- **往返延迟 / image_age**: `--profile-latency` → `/tmp/kai0_latency_<pid>.csv` (server_infer_ms + 总往返), 解释 RTC 降速幅度 (延迟越大 inference_delay 越大)。
- **FFT 主频** (可选): 判有无 0.99Hz 病态振荡 (EMA 关后是否抖)。

### 3.3 采数脚本

> ⬜ 待建: `train_scripts/kai/eval/measure_deploy_speed.py <trace_or_bag_dir>` — 输入一段执行记录, 输出 §3.1 表。可复用数据集速度分析的差分逻辑。

---

## 4. 结果表 (待填)

**链路**: ⬜v0 / ⬜v1   **ckpt**: `____`   **场景**: `____`   **每组 N 段**: `__`

### 4.1 手臂关节速度 (rad/s, 活动帧)

| 组 | RTC | EMA α | mean | median | p90 | vs A3(默认) | vs 遥操 |
|---|---|---|---|---|---|---|---|
| A0 全关 | off | 1.0 | | | | | |
| A1 仅RTC | on | 1.0 | | | | | |
| A2 仅EMA | off | 0.5/0.7 | | | | | |
| A3 全开 | on | 0.5/0.7 | | | | 1.00× | |

### 4.2 夹爪速度 (归一化/s)

| 组 | RTC | EMA α | mean | median | p90 | vs A3 |
|---|---|---|---|---|---|---|
| A0 | off | 1.0 | | | | |
| A1 | on | 1.0 | | | | |
| A2 | off | 0.5/0.7 | | | | |
| A3 | on | 0.5/0.7 | | | | |

### 4.3 cycle 时间 + 延迟

| 组 | cycle 时间(s) | 往返延迟 p50(ms) | inference_delay(步) | 备注 |
|---|---|---|---|---|
| A0 | | | | |
| A1 | | | | |
| A2 | | | | |
| A3 | | | | |

### 4.4 结论 (待填)

- RTC 单独降速: A1 vs A0 = ____%  (手臂) / ____% (夹爪)
- EMA 单独降速: A2 vs A0 = ____%  (手臂) / ____% (夹爪)
- 二者是否叠加 (A3 ≈ A1×A2?): ____
- 主项是 RTC 还是 EMA: ____  → 决定先动哪个 (降延迟 / 调 guidance / 关 EMA / 提 speed_factor)

---

## 5. 注意 / 陷阱

- **EMA 改完必核实**: `ros2 param get /policy_inference publish_smooth_alpha` (ros2 param set 对它无效, 只能重启)。
- **RTC 热改后清 prev_chunk**: 节点已自动处理, 翻转瞬间可能有 1 个 chunk 过渡, 丢弃头几秒再统计。
- **控制场景**: 折叠任务不同阶段速度差异大 (approach 快、精细操作慢), A/B 必须同起始场景 + 同里程碑截断, 否则速度均值不可比。
- **speed_factor 固定 1.0**: 别在 A/B 里动油门, 否则混淆。
- **别用开环离线复述判速度**: `action==state` 会让开环复述观测频率, 速度差要真机闭环测 (见 memory 真机晃动条目)。

---

## 6. 相关

- `docs/deployment/inference/rtc_implementation.md` — RTC 机制与历史 sweep
- `start_scripts/kai/rtc_apply.sh` — RTC 预设助手
- policy_inference_node.py: `inference_delay` 换算 (~line 2701), EMA (~line 2978), 参数回调 (~line 967)
