# V3 采集管线改造 — 在线前端裁剪 + 夹爪取主臂 action

> **目标**: 让**采集时**直接产出 V3 数据，免去事后 `build_no_release.py --per-date` 转换。两项改动：
> 1. **在线前端裁剪 (front-trim)**: 录制落盘时自动裁掉每条 episode 开头的"投放等待"静止段 (14D state 长时间不变)，保留 onset 前 `MARGIN=15` 帧 lead-in，**不全删**。语义与 `build_no_release.motion_onset` 完全一致。
> 2. **夹爪 action 取主臂 (gripper-from-master)**: 新 `action==state` 约定 = 12 个手臂关节 action = 从臂 state (不变)；2 个夹爪维 (dim6/dim13) 的 action = **主臂(摇操臂)** 夹爪指令。state 仍全是从臂。
>
> 关联: [`docs/training/future_plans/plans/idle_data_trimming_experiments.md`](../../training/future_plans/plans/idle_data_trimming_experiments.md) §3.6 已结论 **front-trim 有效、middle-trim(v3.2) 退化**——本改造只做 front-trim。

---

## 0. 现状 (改造前)

采集落盘的单一收口是 `web/data_manager/backend/app/dataset_writer.py::EpisodeWriter.write_tick`，被三处复用：

| 入口 | 文件 | 备注 |
|---|---|---|
| 遥操采集 | `recorder.py::_capture_loop` → `write_tick` | 走 `ros_bridge.get_state_action()` 取 state/action |
| DAgger 采集 | `dagger_recorder_node.py::_on_record_tick` → `write_tick` | 自建 state/action (从臂 state, action=state) |
| 自主部署录制 | `autonomy_recorder_node.py` | **部署侧诊断录制，不属"数据集生成"** |

- `write_tick` **逐帧即时编码** mp4 (PyAV) + 累积 parquet 行，无任何裁剪。
- `action==state` 现状: 遥操 (`ros_bridge.get_state_action`, `KAI0_ACTION_EQ_STATE=1`) 与 dagger 均令 **action 全 14 维 = 从臂 state**。

---

## 1. 设计

### 1.1 在线前端裁剪 (EpisodeWriter 内置，滚动缓冲)

事后裁剪要重编码 3 路 mp4；在线裁剪用**滚动缓冲 + 延迟编码**，零重编码、内存有界。

算法 (与 `build_no_release` 同口径常量: `THR=3e-3`, `WIN=10`, `MARGIN=15`, `ARM_DIMS=[0..5,7..12]`):

- 维护一个最多 `MARGIN+WIN=25` 帧的**待定缓冲** (raw RGB(已 resize) + depth + state/action/ts/intervention)。
- 每来一帧: 算 `da = mean|Δaction[ARM_DIMS]|` vs 上一帧；`moving = da>THR`；连续 `moving` 计数 `run`。
  - `run>=WIN` → **onset 命中** (= `i-WIN+1`)。此刻缓冲恰好持有 `[onset-MARGIN .. i]` (证明见下) → **整桶 flush 编码**，之后转**直通模式** (即时编码，等同现状)。
  - 未命中 → 缓冲超过 `MARGIN+WIN` 就丢最旧 (这些帧必早于任何未来 cut，安全)。
- `finalize` 时仍未命中 onset (从未运动的异常 ep): 只保留缓冲最后 `MARGIN` 帧 (对齐 `build_no_release`: `cut=len-MARGIN`)。

**缓冲恰为待保留帧的证明**: onset 在绝对帧 `O` 命中于 `i=O+WIN-1`；缓冲容量 `MARGIN+WIN` → `buf_start = max(0, i-(MARGIN+WIN)+1) = max(0, O-MARGIN) = cut`。∴ flush 整桶 = parquet 行 `[cut:]`，与离线版逐位一致。内存上界 = `25 帧 × 3 cam ≈ 70MB`，瞬态。

**pts/frame_index**: 用单独的"已编码计数" `self._frame_idx`，只在 emit 时自增 → 裁剪后 mp4 pts 与 parquet `frame_index` 都从 0 连续 (避免 lerobot timestamp 容差崩)。

**开关**: `EpisodeWriter(front_trim=None)` → `None` 时读 env `KAI0_FRONT_TRIM` (**默认 "0" 关**)。
**仅采集入口显式开 (`=1`)**，autonomy_recorder 不导出 → 保持关 (否则 idle-table 诊断录制会被裁到 15 帧)。

### 1.2 夹爪 action 取主臂

`action = state` 基础上，把 2 个夹爪维覆盖为主臂夹爪指令：

- **遥操** (`ros_bridge.get_state_action`): `KAI0_ACTION_EQ_STATE=1` 路径里，若 `KAI0_GRIPPER_FROM_MASTER=1` (默认开) 且主臂在线 → `action[6]=left_master[6]`, `action[13]=right_master[6]`；主臂缺失则回退从臂夹爪 (= state)。
- **DAgger** (`dagger_recorder_node._on_record_tick`): 节点已订阅 `/master/joint_{left,right}` (`_q_master_*`)。同样 `action[6]/[13]` 取主臂夹爪，加 `_got_master_*` 收到标志做回退保护。
- 12 个手臂维不动 (仍 = 从臂 state)；`state` 全 14 维仍 = 从臂。

**开关**: env `KAI0_GRIPPER_FROM_MASTER` (默认 "1")。

---

## 2. 改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `web/data_manager/backend/app/dataset_writer.py` | `EpisodeWriter` 加 front-trim 常量 + 滚动缓冲；`write_tick` 拆出 `_emit_tick`/`_prep_rgb`/`_prep_depth`；构造读 `front_trim`/env；`finalize` 处理未命中 onset |
| 2 | `web/data_manager/backend/app/ros_bridge.py` | `get_state_action` 加 `KAI0_GRIPPER_FROM_MASTER` 路径 (夹爪取主臂 + 回退) |
| 3 | `ros2_ws/src/piper/scripts/dagger_recorder_node.py` | `_on_master` 设 `_got_master_*`；`_on_record_tick` action 夹爪维取主臂 (env 开关 + 回退) |
| 4 | `start_scripts/start_data_collect.sh` | 导出 `KAI0_FRONT_TRIM=1` + `KAI0_GRIPPER_FROM_MASTER=1` (+ info 日志) |
| 5 | `start_scripts/start_dagger_collect.sh` | 同上导出 (env 透传到 dagger_recorder_node) |

不改: 训练管线 / norm_stats / config.py / autonomy_recorder / `build_no_release.py` (离线版保留)。

---

## 3. 验证

1. **离线对拍 (核心)**: 造一条带长前端静止 + 运动的合成 action 序列，喂滚动缓冲版的 onset/cut 逻辑，断言 cut 与 `build_no_release.motion_onset` 逐位相等 (含: 正常、onset<MARGIN、从不运动 三种)。
2. `python -c "import ast; ast.parse(...)"` / `bash -n` 语法检查全部改动。
3. **冒烟**: mock bridge 跑 EpisodeWriter front_trim=1，喂"前 100 帧静止 + 后 200 帧运动"，断言落盘 parquet 行数 ≈ `200+MARGIN`、mp4 帧数==parquet 行数、`frame_index` 从 0 连续。
4. **夹爪**: 单测 `get_state_action` 在有/无主臂两种情形下 `action[6]/[13]` 取值正确、12 手臂维 == state。
5. **真机** (用户): 采一条 → 回放确认开头静止被裁、夹爪 action 跟手、训练 loader 不报 timestamp 容差。

> ⚠️ 部署类改动 (start_*.sh / ROS2 node) 真机验证通过前**不 commit** (见 memory `commit_after_verify`)。
