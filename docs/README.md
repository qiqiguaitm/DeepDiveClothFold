# docs/ — deepdive_kai0 文档总索引 (顶层导航)

> **用途**: 全项目文档的**唯一顶层入口**。本文件描述 `docs/` 下每个文件的作用 + 检索方式。
> **结构原则**: 多叉树分诊 —— 顶层 README(本文件)→ 域 README → 子目录 README → 具体文档。每层只做"路由",叶子才是内容。
> **更新**: 2026-06-11(uc 集群停用归档 + 生成本顶层索引)。

---

## 0. 顶层结构

```
docs/
├── README.md                  ← 你在这里 (顶层分诊)
├── project_complete_guide.md  项目完整指南 (架构/代码/数据/部署全景, 新人首读)
├── deployment/                部署 · 训练运维 · 数据采集 · 推理 · 可视化 · 事件 (6 场景)
├── training/                  训练实验: 分析 · 未来计划 · 历史结果 (3 子域)
└── backup/                    归档: 已停用/历史文档 (当前 = uc 集群)
```

| 顶层 | 何时来这 | 入口 |
|---|---|---|
| **`project_complete_guide.md`** | 第一次接触项目 / 要全局观 | 单文件 |
| **`deployment/`** | 真机部署 / 提训练任务 / 集群运维 / 采数据 / 推理优化 / 可视化 / 复盘事故 | [`deployment/README.md`](deployment/README.md) |
| **`training/`** | 看某次训练结果 / 找实验计划 / 读结果分析归因 | [`training/README.md`](training/README.md) |
| **`backup/`** | 查已停用资源的历史 (uc 集群) | [`backup/README.md`](backup/README.md) |

---

## 1. 检索方式 (3 种, 按场景选)

1. **README 逐层下钻 (首选)** — 从本文件 → 域 README → 子目录 README。每个 README 都有"按需求找文件"表。适合"我想做 X,该读哪篇"。
2. **关键词 grep** — 知道术语时最快:
   ```bash
   # 标题里搜 (定位文档)
   grep -rn "^#.*<关键词>" docs/ --include="*.md"
   # 全文搜 (定位段落); 例: 找某 config / 某踩坑 / 某机器
   grep -rin "norm_stats\|pi05_flatten_fold_awbc\|cn-beijing\|vision-blind" docs/ --include="*.md"
   ```
3. **本文件 §2 全量文件清单** — 想要"每个文件一句话作用"的总表时直接往下看。

> **命名约定**(辅助检索):`*_plan.md`/`future_plans/` = 计划;`*_results.md`/`history/` = 已完成结果;`analysis/` = 归因分析;`incidents/` = 事故复盘;各目录 `README.md` = 该层分诊。

---

## 2. 全量文件清单 (每文件一句话作用)

### 2.1 顶层
| 文件 | 作用 |
|---|---|
| [`project_complete_guide.md`](project_complete_guide.md) | deepdive_kai0 项目完整指南(架构/代码/数据/部署全景) |

### 2.2 `deployment/` — 部署与运维 (索引: [`deployment/README.md`](deployment/README.md))

**根级**
| 文件 | 作用 |
|---|---|
| [`deployment/README.md`](deployment/README.md) | 部署文档总索引 (6 场景多叉树分诊) |
| [`deployment/multimodal_inference_protocol.md`](deployment/multimodal_inference_protocol.md) | 多模态推理协议 (Depth + EE Pose) |
| [`deployment/pi05_inference_backend_benchmark_plan.md`](deployment/pi05_inference_backend_benchmark_plan.md) | Pi0.5 推理后端对比测试 plan |
| [`deployment/piper_arm_id_and_mode_review.md`](deployment/piper_arm_id_and_mode_review.md) | Piper 双臂标识 & 主从模式调研 |
| [`deployment/piper_arm_teleop_troubleshooting.md`](deployment/piper_arm_teleop_troubleshooting.md) | Piper 双臂遥操偶发中断排查手册 |
| [`deployment/tos_access_guide.md`](deployment/tos_access_guide.md) | TOS 对象存储数据拉取指南 |

**`training_ops/` — 训练集群运维** (索引: [`training_ops/README.md`](deployment/training_ops/README.md))
| 文件 | 作用 |
|---|---|
| [`training_ops/overview.md`](deployment/training_ops/overview.md) | 服务器全景 + 单机启动模板 + 各机用途分工 + 性能基线 |
| [`training_ops/storage_and_env.md`](deployment/training_ops/storage_and_env.md) | 工作目录/ckpt 存储规范/数据集源/Python 栈/env vars/实验命名 |
| [`training_ops/ssh_and_credentials.md`](deployment/training_ops/ssh_and_credentials.md) | SSH 连接/用户体系/TOS 凭据 |
| [`training_ops/data_sync_tos.md`](deployment/training_ops/data_sync_tos.md) | TOS 枢纽/sim01 上传/训练机拉/跨机 sync/ckpt 回流 |
| [`training_ops/checkpoints_layout.md`](deployment/training_ops/checkpoints_layout.md) | ckpt 目录文件命名 (params/train_state/assets/EMA) |
| [`training_ops/submission/README.md`](deployment/training_ops/submission/README.md) | **训练任务提交 (2 路径)** 分诊 + 提交前检查清单 |
| [`training_ops/submission/volc_ml_platform.md`](deployment/training_ops/submission/volc_ml_platform.md) | Volc ML Platform YAML/SDK 提交 + region/queue + Volc 特有坑 |
| [`training_ops/submission/gf0_control_plane.md`](deployment/training_ops/submission/gf0_control_plane.md) | gf0 统一控制平面 (mlp CLI/queue/镜像/批量提交) ⭐推荐 |
| [`training_ops/submission/training_pitfalls_common.md`](deployment/training_ops/submission/training_pitfalls_common.md) | 跨集群共性踩坑 + 新数据集→提交 7 步前置链 |

**`strategy/` — 战略与上游对照** (索引: [`strategy/README.md`](deployment/strategy/README.md))
| 文件 | 作用 |
|---|---|
| [`strategy/task_a_master_plan.md`](deployment/strategy/task_a_master_plan.md) | Task A 主规划 (训练+部署 roadmap) |
| [`strategy/awbc_implementation_plan.md`](deployment/strategy/awbc_implementation_plan.md) | AWBC / RECAP advantage 升级实施方案 (⭐当前 AWBC 执行计划) |
| [`strategy/dagger_implementation_plan.md`](deployment/strategy/dagger_implementation_plan.md) | DAgger / SFT 迭代实施方案 |
| [`strategy/rlt_implementation_plan.md`](deployment/strategy/rlt_implementation_plan.md) | RLT (Physical Intelligence RL Token) 实施方案 |
| [`strategy/cross_embodiment_strategy.md`](deployment/strategy/cross_embodiment_strategy.md) | Cross-embodiment 战略与数据分析 |
| [`strategy/official_diff_and_risk_analysis.md`](deployment/strategy/official_diff_and_risk_analysis.md) | deepdive_kai0 vs 官方 kai0 差异/风险/修复记录 |

**`inference/` — 推理与真机部署** (索引: [`inference/README.md`](deployment/inference/README.md))
| 文件 | 作用 |
|---|---|
| [`inference/sim01_deployment.md`](deployment/inference/sim01_deployment.md) | sim01 部署文档 |
| [`inference/xvla_inference_bringup.md`](deployment/inference/xvla_inference_bringup.md) | X-VLA 推理 bring-up (ckpt → 真机跑通) |
| [`inference/xvla_upstream_vs_local_consistency.md`](deployment/inference/xvla_upstream_vs_local_consistency.md) | X-VLA 上游 vs 本地一致性分析 |
| [`inference/rtc_implementation.md`](deployment/inference/rtc_implementation.md) | RTC (Real-Time Chunking) 实现方案 |
| [`inference/fixed_noise_inference_fix.md`](deployment/inference/fixed_noise_inference_fix.md) | Fixed-noise 推理修复 (真机 oscillation 诊断) |
| [`inference/ipc_inference_deployment_review.md`](deployment/inference/ipc_inference_deployment_review.md) | IPC & 推理服务部署 review |
| [`inference/ros2_image_inference_validation_review.md`](deployment/inference/ros2_image_inference_validation_review.md) | ROS2 图像处理与推理校验策略 review |
| [`inference/build_web_venv.md`](deployment/inference/build_web_venv.md) | data_manager 后端 venv 构建 |
| [`inference/realtime_vla/README.md`](deployment/inference/realtime_vla/README.md) | 实时 VLA 推理优化 series 索引 |
| [`inference/realtime_vla/strategy.md`](deployment/inference/realtime_vla/strategy.md) | 实时优化战略 |
| [`inference/realtime_vla/roadmap.md`](deployment/inference/realtime_vla/roadmap.md) | 5 阶段实施路线图 + 真机测试 |
| [`inference/realtime_vla/ee_stability_layer1.md`](deployment/inference/realtime_vla/ee_stability_layer1.md) | EE 末端稳定性优化 Layer 1 |
| [`inference/realtime_vla/layer_b_plan.md`](deployment/inference/realtime_vla/layer_b_plan.md) | Layer B 系统级优化 plan |
| [`inference/realtime_vla/v1_triton_log.md`](deployment/inference/realtime_vla/v1_triton_log.md) | V1 Triton 推理优化实施日志 |
| [`inference/realtime_vla/flash_future_research.md`](deployment/inference/realtime_vla/flash_future_research.md) | FLASH 深度研究方向 |

**`data_collection/` — 数据采集与遥操作** (索引: [`data_collection/README.md`](deployment/data_collection/README.md))
| 文件 | 作用 |
|---|---|
| [`data_collection/teleoperation_guide.md`](deployment/data_collection/teleoperation_guide.md) | 遥操作指南 |
| [`data_collection/dagger_collection_guide.md`](deployment/data_collection/dagger_collection_guide.md) | DAgger 数据采集操作指南 + 架构 |
| [`data_collection/data_manager_plan.md`](deployment/data_collection/data_manager_plan.md) | 双臂 VLA 数据采集 UI 设计计划 |
| [`data_collection/replay_and_stacks_usage.md`](deployment/data_collection/replay_and_stacks_usage.md) | Replay 与三栈使用指南 |

**`visualization/` — 可视化** (索引: [`visualization/README.md`](deployment/visualization/README.md))
| 文件 | 作用 |
|---|---|
| [`visualization/inference_visualization.md`](deployment/visualization/inference_visualization.md) | 在线推理可视化与交互执行控制 |
| [`visualization/inference_visualization_mesh.md`](deployment/visualization/inference_visualization_mesh.md) | 点云→Mesh 化升级方案 |
| [`visualization/rerun_mesh_transparency_lesson.md`](deployment/visualization/rerun_mesh_transparency_lesson.md) | Rerun mesh 透明度 debug postmortem |

**`incidents/` — 事件 + Debug log** (索引: [`incidents/README.md`](deployment/incidents/README.md))
| 文件 | 作用 |
|---|---|
| [`incidents/task_a_real_robot_grasp_corner_debug_log.md`](deployment/incidents/task_a_real_robot_grasp_corner_debug_log.md) | 真机叠衣"夹不到衣角"排查日志 |
| [`incidents/2026-04-27_realsense_anti_flicker.md`](deployment/incidents/2026-04-27_realsense_anti_flicker.md) | RealSense 抗闪烁修复 |
| [`incidents/usb_camera_layout.md`](deployment/incidents/usb_camera_layout.md) | USB camera layout issue |

### 2.3 `training/` — 训练实验 (索引: [`training/README.md`](training/README.md))

**`analysis/` — 结果分析与归因** (索引: [`analysis/README.md`](training/analysis/README.md))
| 文件 | 作用 |
|---|---|
| [`analysis/xvla_vs_official_gap_rootcause.md`](training/analysis/xvla_vs_official_gap_rootcause.md) | X3.C vs 官方 X-VLA vs pi05 真机差根因 |
| [`analysis/x3c_realrobot_trace_20260601.md`](training/analysis/x3c_realrobot_trace_20260601.md) | X3.C 真机 trace 震荡/折返实证 |
| [`analysis/xvla_dataset_vs_official.md`](training/analysis/xvla_dataset_vs_official.md) | 我们 EE6D 数据 vs 官方 Agilex 对齐审计 |
| [`analysis/xvla_innovation_directions.md`](training/analysis/xvla_innovation_directions.md) | X-VLA 优化/创新方向研究 |
| [`analysis/pi05_cross_embodiment_training_deep_dive.md`](training/analysis/pi05_cross_embodiment_training_deep_dive.md) | pi0.5 跨本体训练深度研究 (官方 vs 我们) |
| [`analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md`](training/analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md) | 数据规模 vs 质量 (vis_v2_full vs pure_200) |
| [`analysis/vis_v2_full_data_audit.md`](training/analysis/vis_v2_full_data_audit.md) | vis_v2_full 数据侧 audit (oscillation 根因) |
| [`analysis/base_dataset_preprocess_assessment.md`](training/analysis/base_dataset_preprocess_assessment.md) | base 数据集预处理价值全维度扫描 |
| [`analysis/tac_v2_effectiveness_pure_200.md`](training/analysis/tac_v2_effectiveness_pure_200.md) | TAC v2 有效性 (pure_200) |
| [`analysis/pytorch_vs_jax_eval_postmortem.md`](training/analysis/pytorch_vs_jax_eval_postmortem.md) | PyTorch vs JAX eval 方法论 + 踩坑 |

**`future_plans/plans/` — 未来训练计划** (索引: [`future_plans/README.md`](training/future_plans/README.md))
| 文件 | 作用 |
|---|---|
| [`xvla_proprio_shortcut_openloop_fix.md`](training/future_plans/plans/xvla_proprio_shortcut_openloop_fix.md) | X-VLA vision-blind 开环根因认证 + 修复 (E0/E1/E2/E3) |
| [`xvla_track_x_curriculum.md`](training/future_plans/plans/xvla_track_x_curriculum.md) | Track X — X-VLA 官方架构 native 训练 (X3.A/B/C) |
| [`xvla_camera_robust_grasp_final.md`](training/future_plans/plans/xvla_camera_robust_grasp_final.md) | X-VLA 相机鲁棒精确抓取最终方案 |
| [`xvla_domain_slot_init_ablation.md`](training/future_plans/plans/xvla_domain_slot_init_ablation.md) | XVLA domain 槽位 warm-init ablation |
| [`idle_data_trimming_experiments.md`](training/future_plans/plans/idle_data_trimming_experiments.md) | idle(静止/投放)数据裁剪影响 (v3/v3.2) |
| [`gripper_action_clip_experiment.md`](training/future_plans/plans/gripper_action_clip_experiment.md) | 夹爪 action 裁剪对真机夹持稳定性影响 |
| [`corrected_plan_a_conditioning_premerge.md`](training/future_plans/plans/corrected_plan_a_conditioning_premerge.md) | Plan A — embodiment conditioning + per-DS norm |
| [`dagger_validity_and_finetune_comparison.md`](training/future_plans/plans/dagger_validity_and_finetune_comparison.md) | DAgger 数据有效性验证 + 训练方式对比 |
| [`data_root_cause_probe_experiments.md`](training/future_plans/plans/data_root_cause_probe_experiments.md) | 数据问题排查实验系列 |
| [`advantage_pipeline_and_visual_subgoal.md`](training/future_plans/plans/advantage_pipeline_and_visual_subgoal.md) | Advantage pipeline + 视觉 subgoal 增强 |
| [`awbc_viva_value_comparison_plan.md`](training/future_plans/plans/awbc_viva_value_comparison_plan.md) | AWBC × ViVa value model 对比 |
| [`awbc_v2_training_plan.md`](training/future_plans/plans/awbc_v2_training_plan.md) | awbc_v2 训练计划 |
| [`awbc_pi07style_experiment.md`](training/future_plans/plans/awbc_pi07style_experiment.md) | AWBC π0.7-style 实验 (已 superseded) |
| [`gf2_advantage_awbc_plan.md`](training/future_plans/plans/gf2_advantage_awbc_plan.md) | Advantage estimator + AWBC 复现 (原 uc01) |
| [`stage_classifier_plan.md`](training/future_plans/plans/stage_classifier_plan.md) | Stage classifier 训练方案 |
| [`pytorch_native_vis_v2_full.md`](training/future_plans/plans/pytorch_native_vis_v2_full.md) | PyTorch 原生训练 pi05 (R1 abs + R2 delta) |
| [`pytorch_ema_patch_and_retrain.md`](training/future_plans/plans/pytorch_ema_patch_and_retrain.md) | PyTorch EMA patch + 重训 |
| [`A_mirror200_pi05_pytorch.md`](training/future_plans/plans/A_mirror200_pi05_pytorch.md) | A_mirror200 PyTorch 原生训练对照 |
| [`A_0423_0527_excl_calibration_drift.md`](training/future_plans/plans/A_0423_0527_excl_calibration_drift.md) | A_0423_0527 双 init 训练 (排校准漂移段) |
| [`pi05_from_paligemma_base_training_plan.md`](training/future_plans/plans/pi05_from_paligemma_base_training_plan.md) | 从 PaliGemma base 自训 pi0.5 |
| [`ssl_phase_pretrain_pipeline.md`](training/future_plans/plans/ssl_phase_pretrain_pipeline.md) | Track A — SSL phase 0-3 预训练 pipeline |
| [`cosmos3_policy_droid_fold_adapt_plan.md`](training/future_plans/plans/cosmos3_policy_droid_fold_adapt_plan.md) | Cosmos3-Policy-DROID → 叠衣 AC-WM 适配 |
| [`cosmos3_three_model_i2v_eval_plan.md`](training/future_plans/plans/cosmos3_three_model_i2v_eval_plan.md) | Cosmos3 三模型 I2V 世界预测评测 plan |
| [`tau0_fold_visrobot01_joint_finetune.md`](training/future_plans/plans/tau0_fold_visrobot01_joint_finetune.md) | τ₀-WM 叠衣微调 (关节空间 visrobot01) |
| [`task_e_master_plan.md`](training/future_plans/plans/task_e_master_plan.md) | Task E 主规划 (扶起倒箱) |
| [`multinode_distributed_training_plan.md`](training/future_plans/plans/multinode_distributed_training_plan.md) | 双节点分布式训练部署 |
| [`parallel_execution_plan.md`](training/future_plans/plans/parallel_execution_plan.md) | 双机并行执行计划 |
| [`reproduction_plan.md`](training/future_plans/plans/reproduction_plan.md) | kai0 完整复现训练计划 |
| [`training_plans.md`](training/future_plans/plans/training_plans.md) | kai0 复现训练方案 |
| [`gf0_normal_training_plan.md`](training/future_plans/plans/gf0_normal_training_plan.md) | GF0 normal pi0.5 fine-tune 记录与计划 |
| [`gf1_training_plan.md`](training/future_plans/plans/gf1_training_plan.md) | run_gf1.sh 实验计划 |

**`history/` — 训练历史结果** (索引: [`history/README.md`](training/history/README.md))
| 文件 | 作用 |
|---|---|
| [`history/experiments/00_training_history.md`](training/history/experiments/00_training_history.md) | ⭐ 训练实验历史汇总主表 (pi05/kai0/X-VLA) |
| [`history/experiments/training_reproduction_log.md`](training/history/experiments/training_reproduction_log.md) | kai0 Task A 训练复现日志 |
| [`history/experiments/training_paradigm_comparison.md`](training/history/experiments/training_paradigm_comparison.md) | 单阶段 vs 两阶段 (mixed_1 中介) 范式对比 |
| [`history/experiments/training_cli_notes.md`](training/history/experiments/training_cli_notes.md) | Training CLI notes |
| [`history/experiments/wandb_monitoring.md`](training/history/experiments/wandb_monitoring.md) | wandb 训练监控指南 |
| [`history/experiments/dynamic_dataset_workflow.md`](training/history/experiments/dynamic_dataset_workflow.md) | 动态数据集训练流程 |
| [`history/experiments/dataset_diagnostic_report.md`](training/history/experiments/dataset_diagnostic_report.md) | Task_A 数据集诊断报告 |
| [`history/experiments/kai0_task_a_opensource_analysis.md`](training/history/experiments/kai0_task_a_opensource_analysis.md) | kai0 Task_A 开源情况梳理 |
| [`history/experiments/*_results.md` (~20 篇)](training/history/experiments/) | 各次训练结果 (pure_200 NEW SOTA / smooth_800 / mixed_1 / X3 / τ₀ / AWBC baseline / conditioning ablation 等) — 按 config 名 grep |

### 2.4 `backup/` — 归档 (已停用) (索引: [`backup/README.md`](backup/README.md))
| 文件 | 作用 |
|---|---|
| [`backup/uc_cluster_jobs.md`](backup/uc_cluster_jobs.md) | uc01-03 任务提交 + 3-host HSDP/FSDP 集群 (已停用) |
| [`backup/uc_cluster_data_sharing_analysis.md`](backup/uc_cluster_data_sharing_analysis.md) | uc NFS 数据共享拓扑分析 (已停用) |
| [`backup/uc_cluster_reference.md`](backup/uc_cluster_reference.md) | 从 active 文档剪出的 uc 段 (数据源/SSH 拓扑) |
| [`backup/2026-05-16_uc_security_incident_and_backup.md`](backup/2026-05-16_uc_security_incident_and_backup.md) | uc 集群入侵事件复盘 + 备份 (已停用) |

---

## 3. 常见任务 → 入口 (快速路由)

| 我想… | 去 |
|---|---|
| **提一个训练任务** | [`deployment/training_ops/submission/README.md`](deployment/training_ops/submission/README.md)(提交前检查清单 + Volc/gf0 两路径) |
| 看某 config / 某次训练的结果 MAE | [`training/history/experiments/00_training_history.md`](training/history/experiments/00_training_history.md) → grep config 名 |
| 找某实验的计划/动机 | [`training/future_plans/README.md`](training/future_plans/README.md) |
| 真机部署 ckpt / 推理优化 | [`deployment/inference/README.md`](deployment/inference/README.md) |
| 采数据 / 遥操作 | [`deployment/data_collection/README.md`](deployment/data_collection/README.md) |
| 排查真机/相机/夹爪问题 | [`deployment/incidents/README.md`](deployment/incidents/README.md) |
| 服务器型号/路径/SSH/TOS | [`deployment/training_ops/README.md`](deployment/training_ops/README.md) |
| 新人通读项目 | [`project_complete_guide.md`](project_complete_guide.md) |
| 查 uc 集群历史 (已停用) | [`backup/README.md`](backup/README.md) |
