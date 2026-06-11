# incidents/ — 事件 + Debug log + 硬件 Issue

> **场景**: 时间戳事件 / 安全事件 / 真机操作 debug log / 硬件 issue (相机 / USB).
> **命名约定**: 时间戳事件用 `YYYY-MM-DD_<name>.md` 前缀, `ls` 按时间序自然排序。

## 文件清单 (按时间倒序)

| 日期 | 文件 | 主题 |
|---|---|---|
| 2026-05-16 | [`../../backup/2026-05-16_uc_security_incident_and_backup.md`](../../backup/2026-05-16_uc_security_incident_and_backup.md) (uc 已停用) | uc 集群挖矿木马入侵 (Ravencoin Rigel) + 重装备份记录 |
| 2026-04-27 | [`2026-04-27_realsense_anti_flicker.md`](2026-04-27_realsense_anti_flicker.md) | RealSense 抗闪烁修复 — 分水岭 |
| (无日期) | [`task_a_real_robot_grasp_corner_debug_log.md`](task_a_real_robot_grasp_corner_debug_log.md) | Task A 真机叠衣 "夹不到衣角反复尝试" 排查日志 |
| (无日期) | [`usb_camera_layout.md`](usb_camera_layout.md) | USB Camera Layout Issue — 相机识别次序问题 |

## 按需求找文件

| 你怀疑什么 | 去 |
|---|---|
| uc 节点跑慢 / SSH 异常登录 / 矿机 | ../../backup/2026-05-16_uc_security_incident_and_backup.md (含 IoC + 检测命令) |
| RealSense 摄像头条纹 / 闪烁 | 2026-04-27_realsense_anti_flicker.md |
| 真机抓衣角失败 / 夹不到 / 反复尝试 | task_a_real_robot_grasp_corner_debug_log.md |
| USB 摄像头识别错 / /dev/video* 编号乱 | usb_camera_layout.md |

## 跨场景跳转

- 当前推理 / 真机部署文档 → `../inference/`
- 安全后续: SSH 凭据修订 → `../training_ops/ssh_and_credentials.md`
- 相机选型 (硬件层而非 issue) → `../strategy/cross_embodiment_strategy.md` §1.1 (D405 vs D435 对比)

## 添加新 incident 约定

新事件文件用 `YYYY-MM-DD_<short_topic>.md` 命名, 在本 README 表格里加一行。文档结构推荐:
- Executive summary
- 时间线
- 原因分析 (root cause)
- IoCs (如安全事件) / 复现步骤 (如 debug)
- 已处置 / Action Items
- 经验教训
