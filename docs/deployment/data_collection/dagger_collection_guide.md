# DAgger 数据采集 — 操作指南 + 架构 + 规划

> **场景**: 把已部署的 VLA ckpt 拉到真机上跑, 操作员在策略失败时用双臂 master 接管补示范, 数据按 Form C 双 dataset 落盘 (`inference/` + `dagger/`), 喂回训练闭环。
>
> **本文角色**: as-built 运行手册 (SOP) + 系统架构 + 后续规划。算法层面的 master plan (为什么走 DAgger、失败模式路由、RECAP/RLT 升级路径) 见 [`../strategy/dagger_implementation_plan.md`](../strategy/dagger_implementation_plan.md)。纯遥操作录数据 (无策略) 见 [`teleoperation_guide.md`](teleoperation_guide.md)。

---

## 1. 一句话流程

```
启动 infra (脚本) → 浏览器打开 web → 选 ckpt (v0/v1) → 点 Start 加载模型
   → 策略跑 (录 inference/) → 失败时拨柔性开关接管 → 「开始」录 dagger/
   → 「保存」/「丢弃」→ 关开关交还策略 → 循环 → Ctrl-C 收摊 (web 同生命周期退出)
```

三件事由 web 控制: **① 选 ckpt ② Start/Stop 模型会话 ③ 录制 开始/保存/丢弃 (与 `start_data_collect.sh` 一致)**。其余全部只读监控。

---

## 2. 系统架构 (as-built)

### 2.1 两阶段启动 — infra 与 session 解耦

模型加载 (~22–30s) 推迟到操作员点 Start 之后, 不在脚本启动时阻塞。

| 阶段 | 入口 | 起什么 | 谁拉起 |
|---|---|---|---|
| **Infra** | `start_dagger_collect.sh` | CAN + 3 相机 + 2 从臂 + 2 master_servo + dagger_recorder + dagger_pedal + **web** (`enable_policy:=false`, 不起 policy) | 操作员在终端 |
| **Session** | `start_dagger_session.sh` | 仅 `policy_inference_node` (v0 JAX 进程内 / v1 Triton serve + ws client) | web 后端 fork (点 Start) |

`start_dagger_collect.sh` → `start_autonomy.sh --dagger` → `dagger_launch.py` (IncludeLaunchDescription `autonomy_launch.py` + 额外 dagger 节点)。

### 2.2 组件拓扑

```
 ┌─────────────── 终端: start_dagger_collect.sh (Ctrl-C 关全部) ───────────────┐
 │                                                                            │
 │  ROS2 infra (dagger_launch.py)            web/dagger_manager (生命周期绑定)  │
 │  ├ multi_camera_node (3×RealSense)        ├ backend  :8788 (FastAPI+rclpy)  │
 │  ├ piper_left / piper_right (从臂)         └ frontend :5174 (Vite+React)     │
 │  ├ piper_master_left / _right (master)            │                         │
 │  ├ dagger_recorder  ← 状态机 + 双 writer          │ 点 Start → fork ↓        │
 │  └ dagger_pedal     ← evdev HID→/dagger/pedal     │                         │
 │                                                   ▼                         │
 │                              start_dagger_session.sh (policy_inference)     │
 │                              ├ v0: mode=ros2  进程内 JAX                     │
 │                              └ v1: start_serve_v1.sh :8002 + ws/shm client   │
 └────────────────────────────────────────────────────────────────────────────┘
```

端口: web backend **8788** / frontend **5174** (与 data_manager 的 8787/5173 错开, 可并存); v1 serve **8002**。

---

## 3. 完整操作流程 (SOP)

### Step 0 — 前置
- ckpt 已 pack (`train_config.json` + `_CHECKPOINT_METADATA` + `assets/<asset_id>/norm_stats.json`), 放在 `/data1/DATA_IMP/checkpoints/ckpt_v0/` 或 `ckpt_v1/`。
- v1 ckpt 额外需要 `v1_p200.pkl` (自包含于 ckpt 目录, 或 `optimize/results/<name>_v1_p200.pkl`)。

### Step 1 — 起 infra
```bash
cd /data1/tim/workspace/deepdive_kai0
bash start_scripts/start_dagger_collect.sh            # 默认基线 ckpt + 自动起 web
# 可选: --task Task_A  --prompt "..."  --subset dagger  --rerun
```
> ⚠️ **不要用 `| tee` 管道**。要存日志用重定向: `bash start_dagger_collect.sh > /tmp/dagger.log 2>&1`。tee 在 Ctrl-C 时会先死, 留下孤儿进程。

看到 `[dagger] starting web (background)...` + ROS2 节点起来即可。此时**还没有加载模型**。

### Step 2 — web 选 ckpt + Start
浏览器 → `http://<sim01-ip>:5174/`:
1. **System** 卡: `Infra ● ready (shell-managed)`, `Session ○ no policy loaded`。
2. Checkpoint 列表显示 `ckpt_v0` (绿 V0) 和 `ckpt_v1` (蓝 V1)。勾一个有效 ckpt (✓)。
3. 点 **Start** → 后端 fork `start_dagger_session.sh --variant <v0|v1>` → 加载模型。
   - v0 ≈ 22s (JAX 进程内); v1 ≈ 30s (serve build + CUDA graph + ws/shm 连接)。
4. `Session ● loaded (pid …)` 亮起, 策略开始跑, `inference/` 自动录制。

### Step 3 — 接管补示范 (失败时)
1. 拨开柔性开关 (master 按键) → `dagger_recorder` 收到 takeover → **策略停 + finalize 当前 inference ep** → 进 `HUMAN_RECORD`, master 进拖动模式。
2. **开始** 录制 — web **Controls** 卡点「● 开始」, 或踩一下硬踏板 → 开 dagger episode (`recording=True`, State 卡红色 REC)。
3. 双臂 master 拖动把这条轨迹做对。
4. 结束当前 episode (与 `start_data_collect.sh` 一致, 二选一):
   - 「✓ 保存」(或再踩一下踏板) → finalize 并保留。
   - 「✕ 丢弃」→ abort, 删除半成品文件 (轨迹做坏了不想要时)。
   - 同一接管内可多次 开始→保存/丢弃 = 多个 ep。
5. 关柔性开关 → `RETURNING` → master 软件复位 → 策略 resume → 开新 inference ep。
> 三按钮发 `/dagger/record_cmd` (start/save/discard); 硬/软踏板发 `/dagger/pedal_toggled`, 是 开始↔保存 切换 (无丢弃)。

### Step 4 — 收摊
终端 `Ctrl-C`: trap 关 web (`[dagger] stopping web...`) + ros2 launch 退出 + 端口级兜底查杀 8788/5174。`pgrep -af "dagger|piper|uvicorn|vite"` 应为空。

---

## 4. 状态机 + 触发语义 (踏板与状态解耦)

四态 + 一个独立的 `_recording` flag。**柔性开关控制"谁在驾驶", 踏板控制"录不录 dagger ep", 两者正交。**

```
POLICY_RUN ──柔性开关 ON──▶ ALIGNING ──(对齐完)──▶ HUMAN_RECORD ──柔性开关 OFF──▶ RETURNING ──┐
 策略驾驶                    停策略 +              双 master 拖动                 关 dagger ep   │
 录 inference/(=0)          finalize inf ep       ├ 踏板 → 开/关 dagger writer    复位 master    │
                                                  └ State 不变 (始终 HUMAN_RECORD)              │
      ▲────────────────────────────────────────────────────────────────────────────────────────┘
                                          回 POLICY_RUN: 开新 inference ep
```

| State | 谁驱动从臂 | 录什么 | 退出条件 |
|---|---|---|---|
| `POLICY_RUN` | policy `/master/joint_*` | `inference/` (intervention=0), 连续 | 任一柔性开关 ON |
| `ALIGNING` | 从臂 hold | (finalize inference ep, 不录新数据) | 对齐完成 |
| `HUMAN_RECORD` | 双 master 拖动 | `dagger/` (intervention=1), **仅当 `_recording=True`** | 柔性开关 OFF |
| `RETURNING` | master 软件复位 | (close dagger ep if open) | 复位完成 → POLICY_RUN |

**录制语义** (只在 HUMAN_RECORD 生效, 不改状态机):

| 来源 | 动作 | 效果 |
|---|---|---|
| `/dagger/record_cmd` = `start` (web「开始」) | 开 episode | `_open_episode()` + `_recording=True` |
| `/dagger/record_cmd` = `save` (web「保存」) | finalize 保留 | `_close_episode()` + `_recording=False` |
| `/dagger/record_cmd` = `discard` (web「丢弃」) | abort 删除 | `_discard_episode()` (`writer.abort()`) + `_recording=False` |
| `/dagger/pedal_toggled` (硬/软踏板) | start↔save 切换 | 按当前 `_recording` 翻转 (无丢弃) |

- 三动作复用同一组 helper (`_start/_save/_discard_recording`), 与 `start_data_collect.sh` 的 recorder.start/save/discard 逻辑一致 (`finalize` vs `abort`)。
- 太短的 episode (< `min_ep_sec`) 在 save 时也会被自动 drop (abort)。

> 设计要点 (历史踩坑): 早期把第二次踩踏板做成切回 policy, 操作员困惑 → 改为踏板**只**开关 dagger writer, 状态机由柔性开关独占控制。

---

## 5. ckpt 变体: v0 (JAX) vs v1 (Triton)

`start_dagger_session.sh` 按 ckpt 所在分组目录自动判定 (`ckpt_v1/*` → v1, 其余 → v0), 也可 `--variant {v0|v1|auto}` 强制。

| | **v0** (`ckpt_v0/*`) | **v1** (`ckpt_v1/*`) |
|---|---|---|
| 推理 | JAX 进程内 (`mode=ros2`) | V1 Triton `serve_policy_v1.py` (:8002) + `policy_inference_node` ws/shm 客户端 |
| 加载耗时 | ~22s | ~30s (含 serve build + CUDA graph capture + warmup) |
| 额外资产 | 仅 orbax + norm_stats | + `v1_p200.pkl` (自包含 / `optimize/results/`) |
| RTC 调参 | JAX legacy (3Hz/k=8) | 生产配置 (20Hz / k=6 / exec_h=12 / shm / fast_obs / pipelined) |
| 入口对照 | `start_autonomy_from_ckpt.sh` | `start_autonomy_from_ckpt_v1.sh` |
| delta 模式 | — | 自动检测 (`base_config` 含 `delta` 或 `use_delta_joint_actions=True`) → serve `--delta-joint-actions` |

v1 缺 `v1_p200.pkl` 时: web 列表标红 `· no v1_p200.pkl`, 后端 Start 返回 404。需先用 `optimize/v1_triton/convert_kai0_to_v1.py` + `expand_v1_pkl_for_phase2.py` 转换 (见 `start_autonomy_from_ckpt_v1.sh` 提示)。

> **Prompt 大小写**: v1 serve 的 `--prompt` / 客户端 prompt 必须与训练一致, **不能 lowercase** (PaligemmaTokenizer 保留大写)。窄分布 ckpt 大小写不符会静默退化到近乎不动。脚本从 sidecar 取 prompt 原样透传, `--prompt` 可覆盖。

---

## 6. 数据落盘 (Form C 双 dataset)

```
/data1/DATA_IMP/KAI0/<task>/
├── inference/<date>-v2/             ← intervention=0, 策略 rollout (POLICY_RUN 连续录)
│   ├── data/chunk-000/episode_*.parquet
│   ├── videos/chunk-000/{top_head,hand_left,hand_right}/episode_*.mp4
│   └── meta/{episodes.jsonl, tasks.jsonl}
└── dagger/<date>-v2/                ← intervention=1, 人类接管 (HUMAN_RECORD + 踏板录)
    └── (同结构)
```

- 双 dataset 路径分离, episode 索引独立 — 与官方 KAI0 100% 兼容, 是 RECAP advantage estimator 的必需输入 (见 strategy §8.2)。
- 一次"接管 → 补示范 → 交还" cycle 通常产: 多个 `inference/` ep (策略段被 takeover 截断) + 0..N 个 `dagger/` ep (取决于踩了几次踏板)。
- 30Hz 同步采集 14 维 state/action; 三路视频并行编码; parquet 流式落盘 (复用 `EpisodeWriter`)。

---

## 7. Web dagger_manager 详解

| 区域 | 控制? | 内容 |
|---|---|---|
| **相机预览** 卡 | 只读 | 三路 RealSense 实时画面 (左腕 / 头部 / 右腕), 每格叠加 `fps · latency · drop`; 与 `start_data_collect.sh` 的 data_manager 同款 |
| **双臂状态** 卡 | 只读 | 14 维 obs 关节角度条 (双臂各 6 关节 + 夹爪), 取自 `/puppet/joint_*` (从臂真实位姿, 即模型所见) |
| **System** 卡 | ✅ ② ckpt + Start/Stop | Infra LED (shell-managed) + Session LED + ckpt 选择 (v0/v1 徽章, ✓/! 校验) |
| **Controls** 卡 | ✅ ③ 录制启停 | 开始 / 保存 / 丢弃 三按钮 (= `/dagger/record_cmd`) + 硬踏板状态; 与 `start_data_collect.sh` 一致 |
| **State** 卡 | 只读 | 状态机当前态 + REC 指示 + Stack/ROS bridge/policy execute/按键 L,R/last pedal |
| **Episodes** 卡 | 只读 | `inference/` + `dagger/` 磁盘 parquet 实时计数 |

**画面/关节数据通路**: backend ROS bridge 订阅 3 路 `/camera_*/camera/color/image_raw` (BEST_EFFORT sensor QoS) → JPEG 编码 (stride 2 / q60, 可用 `KAI0_JPEG_STRIDE` / `KAI0_JPEG_QUALITY` 调) → `GET /api/camera/<tile>/mjpeg` (multipart MJPEG, `<img src>` 直连); 关节走 `GET /api/joints` (前端 5Hz 轮询)。相机健康 (fps/latency/drop) 随 5Hz WS snapshot 推送, 用于 tile 的 live 门控。

**生命周期绑定**: web 由 `start_dagger_collect.sh` 拉起、随其 Ctrl-C 退出 (trap + 端口级查杀)。Start 只加载 **session** (policy), infra 归 shell 管 — 点 Start 不会重启系统。`SKIP_WEB=1` 可跳过 web 自管 (开发热重载时手动 `web/dagger_manager/run.sh`)。

---

## 8. ROS2 topics 速查 (dagger_recorder)

| 方向 | topic | 类型 | 含义 |
|---|---|---|---|
| sub | `/master_button_left,right` | Bool | 柔性开关 → takeover 触发 |
| sub | `/dagger/takeover` | Bool | web/外部强制接管 |
| sub | `/dagger/pedal_toggled` | Empty | 踏板 (硬/软) → 翻转 `_recording` (start↔save) |
| sub | `/dagger/record_cmd` | String | web 三按钮 `start` / `save` / `discard` |
| sub | `/puppet/joint_left,right` · `/master/joint_left,right` | JointState | 从臂 / 主臂关节 |
| pub | `/dagger/state` | String (latched) | 状态机当前态 |
| pub | `/dagger/recording` | Bool (latched) | dagger writer 是否在录 |
| pub | `/policy/execute` | Bool | 策略执行使能 (接管时 halt) |
| pub | `/master_controled/joint_*` · `/teach/*` | — | master 拖动 / 配置 / teach 模式 |

---

## 9. 常见故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动后机械臂不动, 直接 OBSERVE | 启动时柔性开关已 ON → JAX 加载期间被误判为 takeover | 把两个开关都拨 OFF (handback) 再操作; 已加 boot-gate 要求先看到一次全 OFF |
| 跑完 dagger 后 teleop 读不到接口 | master 残留 drag 模式 / 没释放 CAN | 已修: 节点 shutdown 自动 `DisconnectPort()` + SIGTERM handler; 关闭即释放 |
| rerun 仍自动打开 | `start_autonomy.sh` CLI `enable_rerun:=true` 盖过 dagger_launch 的 include 覆盖 | `start_dagger_collect.sh` 默认 `--no-rerun` 在 start_autonomy 层关掉 |
| Ctrl-C 后 web 仍在跑 / 点 Start 像重启了系统 | tee 死掉留孤儿 + web 误 fork 整栈 | 已修: trap + 端口级查杀; web Start 只管 session, infra 归 shell |
| v1 Start 报 404 no pkl | 该 ckpt 没转换出 `v1_p200.pkl` | 先转换 (§5) |

---

## 10. 规划 / Roadmap

### 已实现 (本轨)
- ✅ 两阶段启动 (infra / session 解耦) + 延迟加载模型
- ✅ 4 态状态机 + 踏板 `_recording` 解耦 (Form C 双 dataset 落盘)
- ✅ web dagger_manager (3 控制 + 只读监控, 生命周期绑定)
- ✅ v0 (JAX) + v1 (Triton serve) ckpt 双路径, 按分组自动判定
- ✅ web 实时监控: 三路相机 MJPEG 预览 + 双臂 14 维关节角度条 (与 data_manager 同款)
- ✅ 录制 开始 / 保存 / 丢弃 三按钮 (`/dagger/record_cmd`), 与 `start_data_collect.sh` 逻辑一致 (finalize vs abort)

### 待办
| # | 项 | 优先级 | 备注 |
|---|---|---|---|
| R1 | **v1 真机 5-cycle 验证** | 高 | 代码已通过 colcon/tsc, 待真机跑 (你来测) |
| R3 | 状态机转移加 `GetArmStatus` 健康检查 | 中 | strategy Phase D4 / piper_review O2 |
| R4 | ALIGNING/RETURNING 改 `MotionCtrl_1` 软件拖动路径 | 中 | strategy Phase D3 |

### 自动化 Gap (引用 strategy 主计划, 不在本采集轨)
- 失败自动检测 (G1)、best ckpt 自动选择/pack (G2)、failure auto-trigger (G3) → 见 [`../strategy/dagger_implementation_plan.md`](../strategy/dagger_implementation_plan.md) §4。

---

## 11. 跨文档跳转

- 算法 master plan / 失败模式路由 / RECAP·RLT 升级 → [`../strategy/dagger_implementation_plan.md`](../strategy/dagger_implementation_plan.md)
- 纯遥操作录数据 SOP → [`teleoperation_guide.md`](teleoperation_guide.md)
- data_manager UI 设计 (采集端) → [`data_manager_plan.md`](data_manager_plan.md)
- replay / 三栈检查 → [`replay_and_stacks_usage.md`](replay_and_stacks_usage.md)
- ckpt 命名 / 目录规范 → [`../training_ops/checkpoints_layout.md`](../training_ops/checkpoints_layout.md)
- 跨服务器数据 sync (录完上传) → [`../training_ops/data_sync_tos.md`](../training_ops/data_sync_tos.md)
- sim01 部署 (DAgger 评估端) → [`../inference/sim01_deployment.md`](../inference/sim01_deployment.md)
- V1 Triton 推理栈 → `../inference/` + `start_scripts/kai/start_autonomy_v1.sh`
