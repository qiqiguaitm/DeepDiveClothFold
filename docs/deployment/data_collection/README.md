# data_collection/ — 数据采集与遥操作

> **场景**: 用人操作机器人录新数据 / 遥操作 (teleop) / data_manager UI / replay 历史数据。

## 文件清单

| 文件 | 用途 |
|---|---|
| [`teleoperation_guide.md`](teleoperation_guide.md) | 遥操作指南 — 双臂 Piper teleop 录数据的完整流程 |
| [`dagger_collection_guide.md`](dagger_collection_guide.md) | DAgger 数据采集 — as-built SOP + 架构 + 规划 (策略跑 + 接管补示范 + 双 dataset) |
| [`data_manager_plan.md`](data_manager_plan.md) | 双臂 VLA 数据采集 UI 设计计划 |
| [`replay_and_stacks_usage.md`](replay_and_stacks_usage.md) | Replay 与三栈 (录制/replay/分析) 使用指南 |

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 上手 teleop 录数据 (从零) | teleoperation_guide.md |
| 跑 DAgger 采集 (策略部署 + 失败接管补示范) | dagger_collection_guide.md |
| 看 data_manager UI 是怎么设计 (新功能开发) | data_manager_plan.md |
| 看一次录好的数据 / replay 检查 / 分析 stack | replay_and_stacks_usage.md |

## 跨场景跳转

- data_manager 后端 venv 构建 → `../inference/build_web_venv.md`
- 录完数据上传 TOS → `../training_ops/data_sync_tos.md` (sim01 → TOS)
- 训练机数据集路径约定 → `../training_ops/storage_and_env.md`
- 真机推理 / 部署 (与采集不同阶段) → `../inference/`
