# 🦾 Piper 机械臂学习手册

> 面向新手的 **Agilex Piper 双臂** 系统学习路径，从 CAN 硬件接入 → SDK 直接控制 → ROS2 集成 → 遥操作 → 数据采集 → VLA 自主推理，逐层递进。
> 每一项都标注了本仓库里**对应的代码文件**，建议边读代码边动手。
>
> 适用环境：**sim01**（双 RTX 5090，负责推理 + IPC：相机 / CAN / ROS2 / Piper SDK）。
> 先决条件：在仓库根目录 `source setup_env.sh`（自动识别机器 profile），ROS2 Jazzy 已安装。

---

## 0 · 学习路线总览

```
第0层 基础概念 ──► 第1层 CAN总线 ──► 第2层 SDK直接控制 ──► 第3层 运动学FK/IK
                                                              │
   第7层 自主推理 ◄── 第6层 数据采集/回放 ◄── 第5层 遥操作 ◄── 第4层 ROS2集成
                                                              │
                                                       第8层 排障（随时回看）
```

| 阶段 | 内容 | 建议用时 |
|---|---|---|
| Day 1 | 第 0~1 层：CAN 激活 + 拓扑认知 | 1 天 |
| Day 2-3 | 第 2 层：SDK 直接控制（核心）| 2 天 |
| Day 4 | 第 3 层：FK/IK 运动学 | 1 天 |
| Day 5-7 | 第 4~5 层：ROS2 节点 + 遥操作 | 3 天 |
| Week 2 | 第 6~7 层：数据采集 → 自主推理 | 1 周 |

---

## 第 0 层 · 基础概念（先建立心智模型）

先搞清楚"谁是谁"，后面所有代码都围绕这套拓扑。

- [ ] **双臂拓扑**：4 个臂 —— `left_master` / `left_slave` / `right_master` / `right_slave`。master = 示教手（人拖动），slave = 执行手（跟随运动）。
- [ ] **固件角色（写入 NVM，需重启生效）**：`0xFA` = master（可拖动示教）、`0xFC` = slave（执行运动）。
- [ ] **控制模式 `ctrl_mode`**：`0x00` STANDBY / `0x01` CAN_CTRL / `0x02` TEACHING / `0x06` LINKAGE_INPUT / `0x07` OFFLINE_TRAJECTORY。
- [ ] **观测/动作维度**：state 14 维 = 双臂关节角 + 夹爪开合。

📂 参考：
- [config/pipers.yml](../config/pipers.yml) —— 4 臂的 CAN 口、波特率、反馈 Hz、ROS2 话题
- [docs/deployment/piper_arm_id_and_mode_review.md](deployment/piper_arm_id_and_mode_review.md) —— 角色 / 模式 / 软件拖动触发详解

---

## 第 1 层 · CAN 总线与硬件接入

目标：能把 4 个 USB-CAN dongle 正确激活并映射到对应的臂。

- [ ] **理解 CAN 激活流程**：扫描 USB 适配器 → 设波特率 → 重命名为符号名（`can_left_mas` 等）
  - [piper_tools/setup_can.sh](../piper_tools/setup_can.sh) —— 主激活管线
  - [piper_tools/activate_can.sh](../piper_tools/activate_can.sh) / [find_all_can_port.sh](../piper_tools/find_all_can_port.sh)
- [ ] **端口↔臂 标定（HITL「摇臂」）**：摇动某只臂，看哪个 CAN 口帧数变化，自动建立映射
  - [piper_tools/calibrate_can_mapping.py](../piper_tools/calibrate_can_mapping.py) —— 交互式标定，写入 `config/pipers.yml`
  - [piper_tools/verify_can_mapping.py](../piper_tools/verify_can_mapping.py) —— 实时验证映射正确性
- [ ] **CAN 健康诊断**：帧率 / bus-off / 错误码
  - [piper_tools/can_health_snap.sh](../piper_tools/can_health_snap.sh) —— 健康快照（支持 `--loop N`）
  - [piper_tools/diagnose_can.sh](../piper_tools/diagnose_can.sh) —— 轻量接口检查
- [ ] **一键硬件自检**：相机 + CAN 臂一起验证
  - [piper_tools/test_hardware.py](../piper_tools/test_hardware.py)（`--cam-only` / `--arm-only`）

> ⚠️ **已知坑**：dongle 重新插拔会导致 USB 重枚举、映射错乱。可用 [export_dongle_serials.py](../piper_tools/export_dongle_serials.py) 导出序列号做位置无关激活（`activate_can_v2.sh`）。

**动手练习**：
```bash
bash piper_tools/setup_can.sh        # 激活 + 标定
bash piper_tools/can_health_snap.sh  # 看总线是否健康
```

---

## 第 2 层 · SDK 直接控制 ⭐（最核心，必学）

目标：脱离 ROS2，直接用 Python SDK 控制单臂运动 + 夹爪。这是理解一切上层逻辑的基础。

SDK 入口类：
```python
from piper_sdk import C_PiperInterface, C_PiperInterface_V2
from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
```

| 功能 | SDK API | 学习样例 |
|---|---|---|
| 连接 / 使能 | `ConnectPort()` / `EnableArm(7)` | [piper_ctrl_go_zero.py](../piper_tools/piper_ctrl_go_zero.py) |
| 关节控制 | `JointCtrl(j0..j5, speed)` | 同上 |
| 夹爪控制 | `GripperCtrl(pos_mm, speed, mode)` | [piper_ctrl_gripper.py](../piper_tools/piper_ctrl_gripper.py) |
| 模式切换 | `ModeCtrl` / `MotionCtrl_1`(拖动) / `MotionCtrl_2`(CAN+MOVE_J) | piper_ctrl_go_zero.py |
| 状态读取 | `GetArmStatus` / `GetArmJointMsgs` / `GetArmGripperMsgs` | [verify_can_mapping.py](../piper_tools/verify_can_mapping.py) |
| 主从配置 | `MasterSlaveConfig(0xFA/0xFC)` | [calibrate_can_mapping.py](../piper_tools/calibrate_can_mapping.py) |
| 回零 / 复位 | 6 关节 + 夹爪归零 | piper_ctrl_go_zero.py |

- [ ] 精读 [piper_ctrl_go_zero.py](../piper_tools/piper_ctrl_go_zero.py)：完整走一遍「连接 → 使能 → 切 CAN 模式 → JointCtrl 归零」
- [ ] 精读 [piper_ctrl_gripper.py](../piper_tools/piper_ctrl_gripper.py)：夹爪开合循环 + 状态反馈
- [ ] 理解 **平滑回到 demo 起始位**：[go_to_pose.py](../piper_tools/go_to_pose.py)（VLA 推理前的预对齐）

> 💡 **关键区分**：`MotionCtrl_1(grag_teach_ctrl=0x01)` 是**软件触发拖动**；`MasterSlaveConfig` 改的是**固件角色**（NVM 持久化，要重启）。

---

## 第 3 层 · 运动学 FK / IK

目标：能在关节空间和末端位姿空间之间互相转换。

- [ ] **正运动学（FK）**：6 关节角 → 末端 6D 位姿
  - [calib/piper_fk.py](../calib/piper_fk.py)（封装 SDK `C_PiperForwardKinematics`，带 DH 偏置修正）
- [ ] **逆运动学（IK）**：目标位姿 + seed → 关节解
  - [calib/piper_ik.py](../calib/piper_ik.py)（ikpy + URDF，处理关节限位）
  - [calib/piper_dh_ik.py](../calib/piper_dh_ik.py)（DH 参数版，参考实现）
- [ ] **手眼标定**：相机 ↔ 臂基座的 4×4 变换
  - [config/calibration.yml](../config/calibration.yml)（`T_world_camF` / `T_world_baseL/R` 等）
  - 标定流程：[calib/](../calib/) 下 `capture_handeye.py` → `solve_calibration.py` → `verify_calibration.py`

---

## 第 4 层 · ROS2 集成（节点 + 话题）

目标：理解整套 ROS2 节点如何协作，话题怎么连。

**自定义消息** [ros2_ws/src/piper_msgs/msg/](../ros2_ws/src/piper_msgs/msg/)：
- `PiperStatusMsg.msg` —— 臂状态反馈（ctrl_mode / teach_status / 各关节限位）
- `PosCmd.msg` —— 末端位姿命令（xyz + rpy + gripper）

**核心节点** [ros2_ws/src/piper/scripts/](../ros2_ws/src/piper/scripts/)：

| 节点 | 作用 |
|---|---|
| `arm_reader_node.py` | mode0=读状态发布 `/puppet/joint_states`；mode1=订阅 `/master/*` 驱动 slave |
| `arm_teleop_node.py` | 示教手执行侧（主从同步） |
| `arm_master_servo_node.py` | master 臂状态发布 + 按钮切换拖动模式 |
| `multi_camera_node.py` | 3× RealSense 驱动（D435 head + 2× D405 wrist）|
| `policy_inference_node.py` | 订阅图像/关节 → JAX 推理 → 发 `/master/joint_*` |

- [ ] 画出**话题数据流图**（autonomy 模式）：
  ```
  3× camera_node ─► /camera_*/color/image_raw ┐
  arm_reader(mode1) ─► /puppet/joint_*         ├─► policy_inference_node
                                               │         │
                                               │         ▼ /master/joint_*
                                               └──► arm_reader(mode1) ─► JointCtrl() ─► slave臂
  ```

---

## 第 5 层 · 遥操作（Teleoperation）

目标：人拖动 master 臂，slave 臂实时跟随。

- [ ] **一键启动**：[start_scripts/kai/start_teleop.sh](../start_scripts/kai/start_teleop.sh)
- [ ] **Launch 编排**：[ros2_ws/src/piper/launch/teleop_launch.py](../ros2_ws/src/piper/launch/teleop_launch.py)（2 master servo + 2 slave teleop）
- [ ] **上手指南 + 工作流**：[docs/deployment/data_collection/teleoperation_guide.md](deployment/data_collection/teleoperation_guide.md)

**数据流（teleop 模式）**：
```
arm_master_servo(mode0) ─► /master/joint_* ─► arm_teleop(mode1) ─► JointCtrl()+GripperCtrl() ─► slave臂
```

---

## 第 6 层 · 数据采集 & 回放

目标：录制示教轨迹（LeRobot v2.1 格式），并能回放验证。

- [ ] **采集**：遥操作同时录制 图像/关节/夹爪
  - [session_launch.py](../ros2_ws/src/piper/launch/session_launch.py) + `autonomy_recorder_node.py`
  - 数据落到 `kai0/data/Task_{A,B,C}/{base,dagger}/`（parquet + mp4）
- [ ] **回放**：parquet 轨迹回放到真臂
  - [playback_launch.py](../ros2_ws/src/piper/launch/playback_launch.py) + `replay_node.py`
  - [start_scripts/kai/start_replay_test.sh](../start_scripts/kai/start_replay_test.sh)
- [ ] **DAgger（人工介入式采集）**：
  - [dagger_launch.py](../ros2_ws/src/piper/launch/dagger_launch.py) + `dagger_pedal_node.py`（脚踏切换自主/接管）

---

## 第 7 层 · 自主推理（VLA 上臂）

目标：跑通「相机 + 臂 + 策略节点」整栈，让模型自主控制机械臂。

- [ ] **启动自主栈**：[start_scripts/kai/start_autonomy.sh](../start_scripts/kai/start_autonomy.sh)
  - `--mode` ros2/websocket/both，`--execute` 启用实际臂控制，`--sim` 仅回放不连 CAN
- [ ] **Launch 编排**：[ros2_ws/src/piper/launch/autonomy_launch.py](../ros2_ws/src/piper/launch/autonomy_launch.py)
- [ ] **推理节点契约**：`policy_inference_node.py`（输入/输出话题、execution-mode joint vs ee_pose）
- [ ] **独立推理节点**（其它节点已起时）：[start_policy_node.sh](../start_scripts/kai/start_policy_node.sh)
- [ ] **运行时切执行开关**：[toggle_execute.sh](../start_scripts/kai/toggle_execute.sh)
- [ ] **并行测试**：[test_inference_parity.py](../start_scripts/kai/test_inference_parity.py)（验证本地 JAX 与 WebSocket serve 一致）

> ⚠️ **RTX 5090 已知坑**：JAX 0.5.3 的 XLA autotuner 会 SIGSEGV，需 `XLA_FLAGS=--xla_gpu_autotune_level=0`。详见 `start_autonomy_from_ckpt_v2.sh`（Blackwell workaround）。

---

## 第 8 层 · 排障（出问题时回来看）

- [ ] **遥操作排障决策树**：[docs/deployment/piper_arm_teleop_troubleshooting.md](deployment/piper_arm_teleop_troubleshooting.md)
  - 5 大根因：SDK 线程崩溃 / bus-off / JointCtrl 失败 / 示教模式锁死 / dongle 重枚举
- [ ] **臂 ID 与模式审查**：[docs/deployment/piper_arm_id_and_mode_review.md](deployment/piper_arm_id_and_mode_review.md)
- [ ] **CAN 健康循环监控**：`bash piper_tools/can_health_snap.sh --loop 10`

---

## 附录 A · 常用命令速查

```bash
# —— 硬件接入 ——
bash piper_tools/setup_can.sh                 # CAN 激活 + 标定
bash piper_tools/can_health_snap.sh           # 总线健康快照
python piper_tools/test_hardware.py           # 相机 + 臂自检

# —— SDK 直控 ——
python piper_tools/piper_ctrl_go_zero.py      # 关节 + 夹爪归零
python piper_tools/piper_ctrl_gripper.py      # 夹爪开合测试

# —— ROS2 应用 ——
bash start_scripts/kai/start_teleop.sh        # 遥操作（4 臂主从）
bash start_scripts/kai/start_autonomy.sh --mode ros2 --execute   # 自主推理
bash start_scripts/kai/toggle_execute.sh      # 切换执行开关
```

## 附录 B · 关键文件地图

| 类别 | 文件 |
|---|---|
| CAN / 硬件 | `piper_tools/{setup_can.sh, calibrate_can_mapping.py, verify_can_mapping.py, can_health_snap.sh}` |
| SDK 直控 | `piper_tools/{piper_ctrl_go_zero.py, piper_ctrl_gripper.py, go_to_pose.py, test_hardware.py}` |
| 运动学 | `calib/{piper_fk.py, piper_ik.py, piper_dh_ik.py}` |
| ROS2 节点 | `ros2_ws/src/piper/scripts/{arm_reader_node, arm_teleop_node, arm_master_servo_node, multi_camera_node, policy_inference_node}.py` |
| Launch | `ros2_ws/src/piper/launch/{autonomy, teleop, session, playback, dagger}_launch.py` |
| 配置 | `config/{pipers.yml, cameras.yml, calibration.yml, intrinsics.yaml}` |
| 启动脚本 | `start_scripts/kai/{start_teleop.sh, start_autonomy.sh, start_policy_node.sh, toggle_execute.sh}` |
| 文档 | `docs/deployment/{piper_arm_teleop_troubleshooting.md, piper_arm_id_and_mode_review.md, data_collection/teleoperation_guide.md}` |

---

*生成于 2026-06-10 · 基于本仓库实际代码整理。若文件结构变动，以仓库实际为准。*
