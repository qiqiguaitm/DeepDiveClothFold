# Piper 双臂遥操偶发中断 — 排查手册

> 现象: 数据采集 (`start_data_collect.sh`) 进行中, **某一条** Piper 臂突然不响应
> 遥操指令，其他 3 条臂正常。重启服务一般能恢复。
>
> 本文档定位用，从最小成本动作开始，逐层向下定根因。

---

## 0. 30 秒决策树

```
出现"某条臂不动"时立刻
    ↓
1. 跑 can_health_snap.sh 抓快照            ← 半分钟
    ↓
2. 看快照里 "## 2. CAN frame rate" 段:
    ┌─────────────────────────────────────────────────┐
    │ 死的 iface 1s frames < 100  → 走 (A) 总线/物理   │
    │ 死的 iface 1s frames ≥ 1000 → 走 (B) SDK/软件   │
    └─────────────────────────────────────────────────┘
        ↓                              ↓
    (A) ip -s link rx/tx errors?  (B) ros2 topic hz /puppet/joint_states?
        ↓                              ↓
    暴增→bus-off(H2)              hz<10→piper_sdk死(H1)
    全0→dongle失联(H5)            正常→JointCtrl失败(H3)
```

---

## 1. 现象描述与已知根因 hypothesis

| 代号 | 根因 | 触发条件 | 自愈 | 现场特征 |
|---|---|---|---|---|
| **H1** | piper_sdk 后台读 CAN 线程异常 break 后无重启 | 总线短暂错误/抖动 | 否 | candump 正常但 `ros2 topic hz` ≈ 0 |
| **H2** | CAN 总线 bus-off (rx/tx errors 累积) | 电气干扰 / 线缆松 / 双臂同总线高负载 | 否 (除非启 `restart-ms`) | `ip -s link` errors 暴增 + state 异常 |
| **H3** | piper_sdk 关键调用 (JointCtrl/EnableArm/MotionCtrl_2) 静默失败 | 关节限位/通信错/总线短暂占用 | 否 | 总线 + topic 都正常, 但臂物理不动; `err_code` 非零 |
| **H4** | 主臂物理按钮误触发 → 进入 teach mode → 不再 forward 指令 | 衣袖蹭按钮 / 多人作业 | 是 (按一次复位) | `ctrl_mode = TEACHING_MODE(0x02)` 或 `teach_status != 0` |
| **H5** | USB-CAN dongle 短暂重枚举 | USB 供电不稳 / hub 闪退 | 否 (iface 名变了) | `ip link show` 找不到原 iface, `lsusb` device 编号变 |

---

## 2. 立刻可用的排查工具

### 2.1 自动后台监控 ⭐ 默认开启
**`start_data_collect.sh` 启动后, `can_diag` 服务会自动跑** (随采集服务管理). 行为:
- 每 30s 写一份快照到 `/tmp/can_diag/snap_<ts>.txt`
- 实时检测每份 snap 的故障 marker (DEAD iface / bus-off / err_code 非零)
- **一旦命中, 自动打包**到 `/tmp/can_diag/INCIDENT_<ts>/`, 包含:
  - `_INCIDENT.txt` (触发原因 + 时间戳)
  - 最近 5 个 snap 文件
  - `arms.log.tail500` + `health_lines.log` (Phase-B `[health]` log 精华)
  - `ros_health_lines.log` (ROS native log 里 `[health]` 关键字)
  - `dmesg.tail` (最近 5 min `gs_usb` / USB disconnect 内核日志)
- 1 分钟冷却避免同一故障重复打包

**关闭/调整**:
- `SKIP_CAN_DIAG=1 bash start_scripts/start_data_collect.sh` — 不起监控
- `CAN_DIAG_INTERVAL=10 bash ...` — 改为 10s 间隔

**查看状态**:
```bash
bash web/data_manager/run.sh status            # 看 can_diag 是否 RUNNING
bash web/data_manager/run.sh logs can_diag     # 跟踪监控日志
ls -lt /tmp/can_diag/ | head                   # 看最新快照 / INCIDENT 目录
```

**出事直接拉 incident bundle**:
```bash
ls -d /tmp/can_diag/INCIDENT_* 2>/dev/null | tail -1   # 最新一次故障的 bundle
# bundle 自带 _INCIDENT.txt 写明触发原因, 直接对照 §3 找对应 H 处理
```

### 2.2 手动快照（需要补充信息时）

```bash
bash start_scripts/kai/diag/can_health_snap.sh > /tmp/snap_$(date +%H%M%S).txt
```

输出 8 段:
1. 4 条 CAN iface 的 `ip -s link`（errors / dropped）
2. 1s candump 帧数（健康基线 ≈ 3200/iface）
3. ROS topic hz（健康基线 ≈ 200 Hz）
4. `/puppet/arm_status` 最近一条
5. ROS2 节点列表
6. piper_sdk 进程
7. lsusb gs_usb dongle 列表
8. 最近 200 行 arms.log

**最关键 3 行（从快照里挑）**:
- "## 2." 的 `[DEAD]` / `[LOW]` 标记直接指认是哪条死了
- "## 1." 死掉 iface 的 errors 列
- "## 4." 的 `err_code` 与 `communication_status_joint_*`

### 2.3 ROS log 里的 Phase-B 健康度

`arm_teleop_node.py` 已加 2 项主动 logging（**只在异常时打印**，正常无噪声）:

| 关键字 | 含义 | 看到这条意味着 |
|---|---|---|
| `[health] can=can_xxx ... isOk=False` | piper_sdk 后台 CAN 读线程死 | **H1** 命中 |
| `[health] can=can_xxx ... err_code=0xXXXX` (非 0) | 臂内部错误（关节限位/通信） | **H3** 命中, 查 byte bit 找具体关节 |
| `[health] ... ctrl_mode=0x02 teach=0x01` | 进入了 teach mode | **H4** 命中 |
| `[health] can=can_xxx master_msg_gap=NNNms` | `/master/joint_states` 断流 100ms+ | 上游 master 节点问题, 不是 slave 问题 |

捞日志:
```bash
grep "\[health\]" /data1/tim/workspace/deepdive_kai0/web/data_manager/logs/arms.log | tail -40
# 或 ROS native log:
find ~/.ros/log -name "*.log" -mmin -60 -exec grep -l "\[health\]" {} \;
```

---

## 3. 各 hypothesis 的现场判读 + 临时恢复

### H1 — piper_sdk 后台线程死

**判读**:
- `candump <iface>` 1s 输出几千行（总线活的）
- `ros2 topic hz /puppet/joint_states` ≈ 0
- arms.log 里出现 `[health] ... isOk=False`

**临时恢复**:
```bash
# 最小: 重启那一个 arm 服务
bash start_scripts/start_data_collect.sh restart
# (后续: 由 Fix 1 watchdog 自愈)
```

### H2 — CAN bus-off

**判读**:
- `ip -s link show <iface>` 显示 `RX: errors > 0` 或 `TX: errors > 0`
- 或 state 不再是 `UP, LOWER_UP` (而是 `BUSOFF` 或 `ERROR-PASSIVE`)
- `candump <iface>` 1s 输出 0 帧

**临时恢复**:
```bash
sudo ip link set <iface> down
sudo ip link set <iface> type can bitrate 1000000 restart-ms 100
sudo ip link set <iface> up
bash start_scripts/start_data_collect.sh restart
```

**根本预防**: 在 `setup_can_v2.sh` / `activate_can_v2.sh` 起 CAN 时永远带 `restart-ms 100`（内核自动 bus-off 恢复）。

### H3 — SDK 静默失败 / 关节通信错

**判读**:
- `candump` + `ros2 topic hz` 都正常
- arms.log `[health] ... err_code=0xXXXX` 非零
- bit 解码:
  - byte 6 bit[0..5]: 关节 1~6 角度超限
  - byte 7 bit[0..5]: 关节 1~6 通信异常

**临时恢复**:
```bash
# 软复位 (清错误)
ros2 service call /can_left_slave/restore_ms_mode std_srvs/srv/Trigger
# 或重启数据采集
bash start_scripts/start_data_collect.sh restart
```
如果是关节通信异常 → 物理检查那个关节的扁平线 / 整臂下电再上电。

### H4 — Teach mode 误触发

**判读**:
- `[health] ... ctrl_mode=0x02` 或 `teach=0x01..0x07`
- 主臂上的按钮亮绿色 (LED on)

**临时恢复**: 再按一次按钮关掉 teach mode。

**根本预防**: 操作员训练 + 主臂上贴胶布盖住按钮 (物理 cover).

### H5 — Dongle 重枚举

**判读**:
- `lsusb` device 编号跟之前对比变了
- `ip link show` 找不到原 iface 名 (变成 canX)
- dmesg 有 `gs_usb` disconnect/connect 日志:
  ```bash
  dmesg | grep -iE "gs_usb|usb.*disconnect" | tail -20
  ```

**临时恢复** (利用 v2 工具):
```bash
bash piper_tools/activate_can_v2.sh    # 按 serial 重命名 + bitrate up
bash start_scripts/start_data_collect.sh restart
```

**根本预防**: 装 udev rules（serial → 永久 kernel-level naming），见 `piper_arm_id_and_mode_review.md` §四·A.3。

---

## 4. 健康基线参考（在正常静止 sim01 上 idle 实测）

| 指标 | 正常值 | 异常阈值 |
|---|---|---|
| `ip -s link` errors | 0 | >0 即异常 |
| `candump` 1s 帧数 | 3000-3500 | <1000 LOW, <100 DEAD |
| `ros2 topic hz /puppet/joint_states` | ~200 Hz | <50 Hz |
| `ros2 topic hz /master/joint_states` | ~50-100 Hz | <20 Hz |
| `arm_status.err_code` byte 6/7 | 0x0000 | 任意 bit 1 |
| `arm_status.ctrl_mode` (slave) | 0x01 CAN_CTRL | 0x02 TEACHING 异常 |
| Phase-B `[health] gap` | 没看到 | 出现 >100ms gap = master 断流 |

---

## 5. 长期修复 roadmap（待落地）

| Fix | 解决 | 风险 | 状态 |
|---|---|---|---|
| **Fix 1** arm_teleop_node 加 1Hz watchdog → isOk()=False 连续 3s 触发 DisconnectPort + ConnectPort + EnableArm | H1, 部分 H5 | 中 (自愈期间会丢约 0.5s 帧) | 待做 |
| **Fix 2** 包装 `_safe_call()` + 周期检查 err_code → 写 ROS topic | H3 全部 | 极低 | 待做 |
| **Fix 3** `setup_can_v2.sh` 起 iface 时强制带 `restart-ms 100` | H2 | 极低 | 待做 |
| **Fix 4** 装 udev rules (`/etc/udev/rules.d/99-piper-can.rules`) | H5 | 低 | 待做 (见 piper_arm_id_and_mode_review.md §A.3) |
| **Fix 5** `run.sh` 加 process-level supervisor (节点 crash 自动 respawn) | 兜底 | 中 | 待做 |

**建议执行顺序**: 先 Phase A/B 抓到 1-2 次真实数据 → 锁定哪个 H 命中频率最高 → 优先做对应的 Fix。盲目实施 Fix 1 可能修错地方（若真正原因是 H2，watchdog 再多次 reconnect 也救不回 dongle）。

---

## 6. 相关文件索引

| 文件 | 作用 |
|---|---|
| `start_scripts/kai/diag/can_health_snap.sh` | 快照 / loop 监控 |
| `start_scripts/start_data_collect.sh` | 数据采集入口 |
| `web/data_manager/run.sh` | 进程管理 (arms / cameras / backend / frontend) |
| `ros2_ws/src/piper/scripts/arm_teleop_node.py` | 遥操节点 (含 Phase-B logging) |
| `piper_sdk/interface/piper_interface_v2.py:609-650` | 后台 CAN 读线程 (H1 来源) |
| `piper_tools/setup_can_v2.sh` / `activate_can_v2.sh` | CAN iface 激活 (serial 路) |
| `docs/deployment/piper_arm_id_and_mode_review.md` | dongle 标识 / master-slave 模式总览 |

---

## 7. 现场处理脚本卡片（贴在采集工位旁）

```
出问题 → 拉最新自动 incident bundle:
  ls -dt /tmp/can_diag/INCIDENT_* | head -1
  cat <该目录>/_INCIDENT.txt        # 自动写好的触发原因

如果没 incident bundle (问题发生太快 / can_diag 没起来):
  bash start_scripts/kai/diag/can_health_snap.sh > /tmp/snap_$(date +%H%M%S).txt

看 _INCIDENT.txt / 快照 "## 2. CAN frame rate":
  □ 死的 iface < 100 帧/s → 看 "## 1." errors;
       0 errors → H5 (dongle 失联) → bash piper_tools/activate_can_v2.sh
       errors 暴增 → H2 (bus-off) → sudo ip link set <iface> down; up; restart
  □ 死的 iface > 1000 帧/s → 看 "## 3." topic hz;
       hz ≈ 0 → H1 (piper_sdk 死) → start_data_collect.sh restart
       hz 正常 → H3 (SDK 失败) 或 H4 (teach) → 看 "## 4." err_code & ctrl_mode

最后兜底: bash start_scripts/start_data_collect.sh restart
```
