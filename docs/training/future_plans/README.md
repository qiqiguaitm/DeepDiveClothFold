# Future Training Plans — 未来训练计划汇总

> **范围**: 列出所有"待启动 / 进行中"的训练计划, 每行简表 + 链接到单文件详细计划。
> **更新**: 2026-05-25。
> **上游战略文档**: [`../../deployment/strategy/cross_embodiment_strategy.md`](../../deployment/strategy/cross_embodiment_strategy.md) (3 robots / 4-层 ROI / Tri-track 架构 / 决策点)。

---

## 当前优先级 plan

### 🌍 Cosmos3 世界模型评测 (2026-06-05 新)

| 优先级 | 计划 | 状态 | 资源 | ETA | 目的 |
|---|---|---|---|---|---|
| ⭐⭐ P1 | [**Cosmos3 三模型 I2V 世界预测评测**](plans/cosmos3_three_model_i2v_eval_plan.md) | 📝 待评审 (设计定稿, P0 env 已解) | 当前主机(b2)+b1 = 16×A100 | ~7–9h | 叠衣 val 上横比 Nano/Super/Super-I2V；3 ep × 3 cam × horizon(1s/3s/7s) **滑窗覆盖整段** teacher-forced rollout，metric-vs-horizon + PSNR/SSIM/LPIPS/temporal/FVD |
| ⭐⭐⭐ P0 | [**Cosmos3 FD 叠衣世界模型**](plans/cosmos3_wam_fold_world_model_plan.md) | 🔄 执行中 (2026-06-12 启动, 本机先行: baseline ✅ / FD 数据通路 ✅ / smoke 运行中) | 本机 8×A100 验证 → AIHC 4n8g 正式 | ~2-3 周到可用评测器 | forward_dynamics 后训练 Nano → 动作可控柔性衣物世界模型; 终极门禁 = 策略评测相关性 r≥0.8; 代码 `cosmos/wam_fold_wm/`, 状态见 plan §7 |

### 🔥 v7/v8 真机失败验证实验 (2026-05-27 新)

| 优先级 | 计划 | 状态 | 资源 | ETA | 目的 |
|---|---|---|---|---|---|
| ⭐⭐⭐ P0 | [**v4 数据可用性验证 (AE AWBC)**](plans/pi05_v4_awbc_validation_plan.md) | 📋 配置定档(待集群) | 8× GPU | — | 全 v4 base(1207ep)+dagger(789ep) 跑 KAI0 AE AWBC, 验证 v4 新框架(前裁+尾裁+夹爪取主臂 action≠state)可用性, 重点真机夹持稳定性 vs 旧 AWBC |
| ⭐⭐⭐ P0 | [**叠衣 SOP 范式基线小实验**](plans/pi05_fold_sop_paradigm_baselines.md) | 📋 配方定档, 逐范式推进 | 8× GPU | — | 同一训练配方在不同折法 SOP 各做 pi05 基线: Vertical Fold v1(Task_AV1 200ep)+ Horizontal Fold v1(Task_AH1 200ep已落地), 跨范式对比哪种更易学 |
| ⭐⭐⭐ P0 | [**Task_A + Task_AV1 混合 1:1 co-train**](plans/pi05_task_a_av1_mixed_1to1_plan.md) | 📋 定稿待实施 | BJ 8× H20 | — | 横向折(1033ep)+竖向折新SOP(304ep冻结)pre-merge + domain_weights=(1,3.256) frame-1:1 过采样, JAX, 50k, warm-start mixed_1_clean |
| ⭐⭐⭐ P0 | [**AWBC 完整流程 on vis Task_A**](plans/awbc_vis_task_a_full_pipeline_plan.md) | 📝 待确认 | 8× GPU | ~1 周 | vis-native 重建打标走完整 Stage 0→4(标注→estimator→打标→discretize→AWBC),对照复用版/SFT |
| ⭐⭐⭐ P0 | [**AWBC × ViVa value model 对比**](plans/awbc_viva_value_comparison_plan.md) | 📝 待评审 | (见 plan) | (见 plan) | 只换 advantage label 来源 (pi0-AdvEst → ViVa) 的受控 A/B |
| ⭐⭐ P1 | [**A_mirror200_pi05_pytorch** (pure_200 PyTorch 对照)](plans/A_mirror200_pi05_pytorch.md) | ✅ done (见 §8 results + postmortem) | 8× GPU | — | 已完成: PyTorch 同协议比 JAX 差 4.1× (@50), EMA 假说证伪 |
| ~~A_0423_0527 双 init~~ | [(plan)](plans/A_0423_0527_excl_calibration_drift.md) | ❌ 取消 | — | — | 用户 2026-05-31 决定不做 |
| ~~PyTorch EMA patch + 重训~~ | [(plan)](plans/pytorch_ema_patch_and_retrain.md) | ❌ 作废 | — | — | EMA 假说被 model-soup 证伪, 见 postmortem |

### 之前优先级 plan (2026-05-23 PM 之后新提的)

| 优先级 | 计划 | 状态 | 资源 | ETA | 关联 task |
|---|---|---|---|---|---|
| ⭐ P1 | [PyTorch 原生训练 pi05 vis_v2_full R1+R2](plans/pytorch_native_vis_v2_full.md) | ⏳ pending (config 未加) | Robot-North-H20 16 H20 串行 | ~4-5 day | `#18` |
| 🔴 P0 | [Track X X-VLA 官方架构 Curriculum](plans/xvla_track_x_curriculum.md) | smooth_800 三件套训练+eval 完成 (X3.C offline 健康); **X3.C 真机失败** → 根因 R1 缺 ImageNet 归一化 (analysis/xvla_vs_official_gap_rootcause.md) → **🔄 P0 重训 `X3C_smooth800_p0` (修 R1+对齐官方 60k, uc01 运行中, §0.NEW.6)**; ⚠️ **2026-06-09 p0+d5anchor 真机仍失败, 根因实为 proprio 捷径 vision-blind, R1 只是表层 → 见下行** | uc01 8 A800 | 运行中 | `#17` |
| 🔴 P0 | [**X-VLA proprio 捷径 / vision-blind 开环修复**](plans/xvla_proprio_shortcut_openloop_fix.md) | 📝 根因已认证 (离线 ablation: p0/d5anchor 视觉影响比 **0.000**, 非数据非部署); 待跑 E1 `use_proprio=False` 确诊 | 8 GPU | (见 plan) | `#17` |
| P2 | [Track A SSL Phase 0-3 Pretrain Pipeline](plans/ssl_phase_pretrain_pipeline.md) | 🔄 Phase 0 部分 done (kai0 base+dagger CoTracker3+SAM2), 余下 pending | uc02 8 GPU + Robot-North-H20 | ~3-5 day Phase 0, 后续待评估 | `#11/12/13/14` |

---

## 历史遗留 plan (按主题归档, 多数已部分执行或被 supersede)

> 这些是早期阶段写的训练计划文档, 实验大多已部分或全部跑完, 结果在 `history/experiments/` 中可查。保留作为**设计依据**与**复现指南**。

### kai0 复现系列

| 计划 | 文件 | 备注 |
|---|---|---|
| kai0 复现 (mixed_1 + full) | [training_plans.md](plans/training_plans.md) | 总入口, 定义 `kai0_mixed_1` (基础版) 与 `kai0_full` (完整版) |
| kai0 复现执行 plan | [reproduction_plan.md](plans/reproduction_plan.md) | 实验编号 + 步骤 |

### AWBC 系列

| 计划 | 文件 | 备注 |
|---|---|---|
| AWBC v2 训练 plan | [awbc_v2_training_plan.md](../history/experiments/awbc_v2_training_plan.md) 🗄️归档 | 第二代 AWBC 实验设计 |
| AWBC pi0.7-style 实验 | [awbc_pi07style_experiment.md](../history/experiments/awbc_pi07style_experiment.md) 🗄️归档 | 仿 pi0.7 风格 AWBC 变体 |
| Advantage Estimator + visual subgoal pipeline | [advantage_pipeline_and_visual_subgoal.md](plans/advantage_pipeline_and_visual_subgoal.md) | AWBC 上游 advantage 训练 + visual subgoal |
| Stage classifier 训练 plan | [stage_classifier_plan.md](plans/stage_classifier_plan.md) | 阶段分类器 (AWBC 状态分段) |

### Hardware-specific 训练 plan

| 计划 | 文件 | 备注 |
|---|---|---|
| gf0 normal training plan | [gf0_normal_training_plan.md](plans/gf0_normal_training_plan.md) | gf0 8×A100 baseline 训练流程 |
| gf1 training plan | [gf1_training_plan.md](plans/gf1_training_plan.md) | gf1 8×A100 训练流程 |
| gf2 advantage + AWBC plan | [gf2_advantage_awbc_plan.md](../history/experiments/gf2_advantage_awbc_plan.md) 🗄️归档 | gf2 跑 advantage estimator + AWBC |
| Multinode distributed training | [multinode_distributed_training_plan.md](plans/multinode_distributed_training_plan.md) | 多节点分布式训练设计 |
| Parallel execution plan | [parallel_execution_plan.md](plans/parallel_execution_plan.md) | 多机并发执行调度 |

### Task E 系列

| 计划 | 文件 | 备注 |
|---|---|---|
| Task E master plan | [task_e_master_plan.md](plans/task_e_master_plan.md) | Phase 0 baseline + Phase 1 差异化 2×2 + Phase 1-FT + Phase 2 vision LoRA. ⭐ Task E 系列总入口 |

### WAM (τ₀ / GigaWorld) 叠衣服系列

| 计划 | 文件 | 备注 |
|---|---|---|
| τ₀-WM 叠衣服微调 (关节空间 · visrobot01) | [tau0_fold_visrobot01_joint_finetune.md](plans/tau0_fold_visrobot01_joint_finetune.md) | 📝 待评审. 在 tau0 框架内真·复用预训练 (复用 action_blocks+视频主干, 仅重置 14 维投影), 关节空间 P1 暖启→P2 专精, visrobot01 关节直控免 FK/IK. P1 ablation 为 go/no-go 分水岭 (否则退 GigaWorld 关节-14) |

---

## 何时新加 plan? 何时移到 history?

- **新加 plan**: 当 cross_embodiment 战略文档 §10.X 出现新的"⏳ pending"实验, 或本地有新 idea 已成形可执行时 → 加到 `plans/`, 同步更新本表。
- **移到 history**: 当 plan 中所有实验跑完并出 eval 结果 → `git mv plans/foo.md ../history/experiments/foo_results.md`, 同步从本表删除、加到 `history/README.md` 表。
- **保留**: 即便实验完成, 如果 plan 文档本身有**长期复现价值** (如 reproduction_plan.md), 可双份保留 plan + results。

---

## 链路图: future_plans → 上游战略

```
docs/deployment/strategy/cross_embodiment_strategy.md  (战略层: 3 robots / Tri-track / Conditioning / TAC / 决策点)
   ├── §3 4-层 ROI         ──→ plans/ssl_phase_pretrain_pipeline.md
   ├── §5/§7 Track X       ──→ plans/xvla_track_x_curriculum.md
   └── §6 RTC/TAC + §7     ──→ plans/pytorch_native_vis_v2_full.md
                                (+ docs/deployment/inference/realtime_vla/strategy.md §1.4 选项 X)
```
