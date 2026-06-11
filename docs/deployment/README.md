# docs/deployment — 部署文档总索引

> **目的**: 为 deepdive_kai0 项目的真机部署 / 训练运维 / 数据采集 / 推理优化 / 可视化 / 事件复盘相关文档提供导航入口。Agent (人或 AI) 按需层层下钻找到所需文档。
> **更新**: 2026-05-25 重构成"6 场景多叉树"结构。

---

## 目录结构 (多叉树)

```
docs/deployment/
├── README.md                          ← 你在这里 (顶层分诊节点)
├── strategy/                          战略与上游对照 (3)
├── training_ops/                      训练集群运维
│   ├── (5 主文档)
│   └── submission/                    任务提交子目录 (3)
├── inference/                         推理与真机部署
│   ├── (4 主文档)
│   └── realtime_vla/                  实时优化 series (4)
├── data_collection/                   数据采集与遥操作 (3)
├── visualization/                     可视化 (3)
└── incidents/                         事件 + Debug log (4)
```

---

## 1. 6 大场景一句话路由 (第 1 步分诊)

| 你的目的 | 去 | 关键入口 |
|---|---|---|
| 看跨本体战略 / Task A roadmap / 与官方 fork 差异 / DAgger 主轨 / AWBC 升级 / RLT 精修轨 | [`strategy/`](strategy/README.md) | cross_embodiment_strategy.md / dagger_implementation_plan.md / awbc_implementation_plan.md / rlt_implementation_plan.md |
| 提训练任务 / 集群知识库 / 数据 sync | [`training_ops/`](training_ops/README.md) | overview.md → submission/ |
| ckpt 部署真机 / 推理时延优化 / sim01 / RTC | [`inference/`](inference/README.md) | realtime_vla/ / sim01_deployment.md |
| 录数据 / 遥操作 / replay 看数据 | [`data_collection/`](data_collection/README.md) | teleoperation_guide.md |
| 在线 viz / mesh / rerun | [`visualization/`](visualization/README.md) | inference_visualization.md |
| 历史 incident / 真机 debug / 硬件 issue | [`incidents/`](incidents/README.md) | (按日期排序) |

---

## 2. 跨场景常见任务"探索路径" (第 2 步: 复合任务的多文档串联)

### 任务 A: 把训完的 ckpt 部署到真机, 推理时延高

```
1. inference/README.md
2. → realtime_vla/README.md (实时优化 series)
3. → strategy.md (战略决策点) → roadmap.md (5 阶段) → v1_triton_log.md (V1 已实施)
4. 并行: training_ops/data_sync_tos.md (取 ckpt 到 sim01)
5. 并行: inference/sim01_deployment.md (部署步骤)
6. 如果需要算法细节: inference/rtc_implementation.md
```

### 任务 B: 提一个新 Volc 训练任务 (Robot-North-H20 16 卡)

```
1. training_ops/README.md
2. → overview.md (服务器全景, 确认 region)
3. → storage_and_env.md (数据/ckpt 路径约定)
4. → submission/README.md
5. → submission/volc_ml_platform.md (YAML + SDK)
6. 或经 gf0 统一: submission/gf0_control_plane.md (mlp CLI 速查)
7. 数据未就位: training_ops/data_sync_tos.md (TOS → cnsh/cnbj)
```

### 任务 C: uc01-03 集群 24 GPU HSDP/FSDP 训练

```
1. ../backup/uc_cluster_jobs.md (3-host HSDP 配置, uc 已停用)
2. → ssh_and_credentials.md (uc 互信拓扑)
3. → storage_and_env.md (RDMA NCCL 环境变量)
```

### 任务 D: 录新数据并训练

```
1. data_collection/teleoperation_guide.md (遥操作录数据)
2. → data_collection/data_manager_plan.md (UI + meta)
3. → training_ops/data_sync_tos.md (sim01 → TOS)
4. → training_ops/storage_and_env.md (训练机数据集路径约定)
5. → training_ops/submission/<...>.md (提任务)
```

### 任务 E: 实时推理出问题排查

```
1. incidents/ (先查历史 incident 是否有同款问题)
2. → inference/realtime_vla/v1_triton_log.md (V1 实施日志, 常见 P50 异常)
3. → inference/ipc_inference_deployment_review.md (IPC 架构)
4. → inference/ros2_image_inference_validation_review.md (ROS2 校验)
```

### 任务 F: uc 节点跑慢, 怀疑入侵

```
1. ../backup/2026-05-16_uc_security_incident_and_backup.md (IoC + 检测脚本, uc 已停用)
2. → training_ops/ssh_and_credentials.md (检查 SSH 状态 + 互信)
```

---

## 3. 大文档拆分映射 (旧 deep link 还原)

| 旧文件 (已删) | 新位置 |
|---|---|
| `cross_embodiment_data_reuse_plan.md` | `strategy/cross_embodiment_strategy.md` |
| `realtime_vla_optimization_analysis.md` | `inference/realtime_vla/{strategy,roadmap,v1_triton_log,layer_b_plan}.md` (4 文件) |
| `training_servers_knowledge_base.md` | `training_ops/{overview,storage_and_env,ssh_and_credentials,data_sync_tos}.md` + `training_ops/submission/{volc_ml_platform,gf0_control_plane}.md` (uc_cluster_jobs → backup/) |
| `analysis_kai0_xvla.md` | 不保留, 主要内容已合并到 `strategy/cross_embodiment_strategy.md` §1, 历史可 `git log` 查 |

旧路径中 `docs/deployment/<file>.md` 现在都需要加场景目录前缀 (`docs/deployment/<scenario>/<file>.md`)。

---

## 4. 文档约定 (写新文档时遵循)

- **归类**: 新文档归到合适场景目录, 找不到再询问加新场景
- **README**: 每个目录至少一个 README (即使内部 1 文件也写), README 不重复正文, 只列+决策表
- **拆 vs 不拆**: 单文件超 800 行 + 内部有 ≥3 个独立主题, 才拆
- **跨场景跳转**: 子目录 README 末尾标"如果你想看 X → 去 `../other/`"
- **时间戳文件**: incident/debug log 用 `YYYY-MM-DD_<name>.md` 格式
- **多叉树深度**: 默认 2 层 (deployment/scenario/file); 内容多/同主题成 series 时下钻第 3 层 (deployment/scenario/series/file)

---

## 5. 上游与平行文档

| 文档 | 位置 | 用途 |
|---|---|---|
| 训练实验历史与计划 | `../training/` (training/README.md) | future_plans + history 两层 |
