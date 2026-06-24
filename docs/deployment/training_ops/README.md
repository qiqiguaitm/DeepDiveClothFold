# training_ops/ — 训练集群运维

> **场景**: 提交训练任务 / 集群知识库 / 文件结构与环境 / 跨服务器数据同步 / Ckpt 管理。
> ⚠️ **uc01/02/03 集群已彻底停用 (2026-05-18 退役)**:uc 专属文档 (`uc_cluster_jobs.md`、`uc_cluster_data_sharing_analysis.md`、SSH 互信/数据源 uc 段) 已移到 [`../../backup/`](../../backup/README.md)。

## 目录结构

```
training_ops/
├── README.md                          ← 你在这里
├── overview.md                        服务器全景 + 单机启动 + 性能基线
├── storage_and_env.md                 文件结构 + ckpt 规范 + Python 栈 + env vars
├── ssh_and_credentials.md             SSH 速查 + 用户 + TOS 凭据
├── data_sync_tos.md                   TOS 枢纽 + 跨服务器 sync + ckpt 回流
├── checkpoints_layout.md              kai0/checkpoints/ 目录规范
└── submission/                        训练任务提交 (2 路径)
    ├── README.md
    ├── volc_ml_platform.md            Volc ML Platform YAML/SDK 提交
    └── gf0_control_plane.md           gf0 作为统一控制平面 (⭐ 推荐)
```

## 文件清单 + 一句话用途

| 文件 | 行数 | 用途 |
|---|---|---|
| [`overview.md`](overview.md) | ~235 | 服务器全景表 + 单机直接启动模板 (gf 通用 + smoke test) + 各机用途分工 + 性能基线 + 修订历史 |
| [`storage_and_env.md`](storage_and_env.md) | ~280 | 工作目录路径 / ckpt 本地存储规范 / 数据集源 / 临时存储 / Python 栈 / 环境变量 / 训练实验命名约定 |
| [`ssh_and_credentials.md`](ssh_and_credentials.md) | ~55 | SSH 连接命令 / 用户体系 / TOS 凭据 |
| [`data_sync_tos.md`](data_sync_tos.md) | ~268 | TOS 中心枢纽 / sim01 上传 / 训练服务器拉 / 跨服务器 sync / ckpt 训练→sim01 部署 |
| [`checkpoints_layout.md`](checkpoints_layout.md) | ~271 | ckpt 目录文件命名 / params / train_state / assets / EMA |
| [`submission/`](submission/README.md) | — | 2 种提任务路径 (Volc / gf0 control plane) |

## 按需求找文件 (Agent 探索表)

| 你想做什么 | 去 |
|---|---|
| 知道 gf0/3 + Robot-North-H20 各是什么型号 / 在哪个 region | overview.md |
| 单机直接启动一个训练 (gf 机, ssh 上去) | overview.md (§5.1-5.5b 通用启动模板) |
| 查 ckpt 应该存到哪个绝对路径 / 数据集放哪 | storage_and_env.md |
| 设置 SSH 别名 / 用 ubuntu vs tim / TOS bucket 凭据 | ssh_and_credentials.md |
| 数据上传 TOS / 从 TOS 拉到训练机 / ckpt 回流 sim01 | data_sync_tos.md |
| 看 ckpt 目录内文件命名 (params, train_state, EMA, ...) | checkpoints_layout.md |
| 提 Volc ML Platform 任务 (Robot-North/robot-task) | submission/volc_ml_platform.md |
| 用 gf0 统一管理所有训练资源 | submission/gf0_control_plane.md |
| data_manager 后端 venv 构建 | ⚠️ 这在 [`../inference/build_web_venv.md`](../inference/build_web_venv.md) (推理 web 服务通用 venv) |
| ⚠️ uc01-03 历史 (直连启动 / 24 GPU HSDP / 数据共享) | [`../../backup/`](../../backup/README.md) (已停用) |

## 跨场景跳转

- 战略与跨本体方向 → `../strategy/`
- 真机推理部署 / 时延优化 → `../inference/`
- 数据采集前置 (teleop / data_manager) → `../data_collection/`
- 历史训练 incident → `../incidents/`
- 训练实验历史与计划 → `../../training/`

- [数据集裁剪 & PTS 归零](dataset_trimming_and_pts.md) — 裁视频头必归零 PTS,否则 lerobot 时间戳解码静默取错帧→真机失败/offline MAE 盲
