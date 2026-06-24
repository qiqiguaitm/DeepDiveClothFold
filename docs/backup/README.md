# docs/backup — 归档 (停用/历史文档)

> **用途**: 存放已停用的服务器 / 已退役的方案相关文档,从 active 文档树移出但保留以备查证。**这里的内容不再维护,不代表当前生产状态。**

## uc 系列集群 (uc01 / uc02 / uc03) — 2026-05-18 重装,后彻底停用

> uc 集群已**彻底停用**。当前生产训练只用 **Volc ML Platform**(cn-beijing Robot-North-H20 / cn-shanghai robot-task)+ gf0/gf3 跳板机。下列文档为 uc 时期的完整记录,迁移自 `docs/deployment/` 各处。

| 文件 | 原位置 | 内容 |
|---|---|---|
| [`uc_cluster_jobs.md`](uc_cluster_jobs.md) | `deployment/training_ops/submission/` | uc01-03 直连启动 + 单机 8 GPU + 三机 24 GPU RDMA HSDP/FSDP 集群训练 + uc 特有踩坑 |
| [`uc_cluster_data_sharing_analysis.md`](uc_cluster_data_sharing_analysis.md) | `deployment/training_ops/` | uc01 NFS export → uc02/03 数据共享拓扑分析 |
| [`2026-05-16_uc_security_incident_and_backup.md`](2026-05-16_uc_security_incident_and_backup.md) | `deployment/incidents/` | uc 集群入侵事件复盘 + 备份处置 |
| [`uc_cluster_reference.md`](uc_cluster_reference.md) | `deployment/training_ops/{storage_and_env,ssh_and_credentials}.md` | 从 active 文档剪出的 uc 段:数据集源路径(NFS 共享布局)+ SSH 互信拓扑 |

**与现状的差异提示**(查阅旧文档时注意):
- uc 路径 `/data/shared/ubuntu/workspace/deepdive_kai0/...` 已不存在。
- uc 的 `1-min git pull cron 镜像 GitHub main` 已随机器停用;现仅 gf3 仍有该 cron。
- "num_workers = 16 × 节点数"等 uc 调参经验对 Volc/gf 机仍部分适用,但路径/拓扑全部失效。
