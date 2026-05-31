# Future Training Plans — 未来训练计划汇总

> **范围**: 列出所有"待启动 / 进行中"的训练计划, 每行简表 + 链接到单文件详细计划。
> **更新**: 2026-05-25。
> **上游战略文档**: [`../../deployment/strategy/cross_embodiment_strategy.md`](../../deployment/strategy/cross_embodiment_strategy.md) (3 robots / 4-层 ROI / Tri-track 架构 / 决策点)。

---

## 当前优先级 plan

### 🔥 v7/v8 真机失败验证实验 (2026-05-27 新)

| 优先级 | 计划 | 状态 | 资源 | ETA | 目的 |
|---|---|---|---|---|---|
| ⭐⭐⭐ P0 | [**AWBC × ViVa value model 对比**](plans/awbc_viva_value_comparison_plan.md) | 📝 待评审 | (见 plan) | (见 plan) | 只换 advantage label 来源 (pi0-AdvEst → ViVa) 的受控 A/B |
| ⭐⭐ P1 | [**A_mirror200_pi05_pytorch** (pure_200 PyTorch 对照)](plans/A_mirror200_pi05_pytorch.md) | ✅ done (见 §8 results + postmortem) | 8× GPU | — | 已完成: PyTorch 同协议比 JAX 差 4.1× (@50), EMA 假说证伪 |
| ~~A_0423_0527 双 init~~ | [(plan)](plans/A_0423_0527_excl_calibration_drift.md) | ❌ 取消 | — | — | 用户 2026-05-31 决定不做 |
| ~~PyTorch EMA patch + 重训~~ | [(plan)](plans/pytorch_ema_patch_and_retrain.md) | ❌ 作废 | — | — | EMA 假说被 model-soup 证伪, 见 postmortem |

### 之前优先级 plan (2026-05-23 PM 之后新提的)

| 优先级 | 计划 | 状态 | 资源 | ETA | 关联 task |
|---|---|---|---|---|---|
| ⭐ P1 | [PyTorch 原生训练 pi05 vis_v2_full R1+R2](plans/pytorch_native_vis_v2_full.md) | ⏳ pending (config 未加) | Robot-North-H20 16 H20 串行 | ~4-5 day | `#18` |
| ⭐ P1 | [Track X X-VLA 官方架构 X3.A/X3.B Curriculum](plans/xvla_track_x_curriculum.md) | ✅ **控制变量三件套训练+eval 完成** (全 A_0423_0527 vis): **X3.C vis-only 各 horizon 全胜** (@1=0.0142), 加 kai +78% / 加 xvla 再 +9% — 但属 fit 非泛化, **真机待做**才是域贡献终判 (§0.1) | uc01+02+03 各 8 A800 | done | `#17` |
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
| AWBC v2 训练 plan | [awbc_v2_training_plan.md](plans/awbc_v2_training_plan.md) | 第二代 AWBC 实验设计 |
| AWBC pi0.7-style 实验 | [awbc_pi07style_experiment.md](plans/awbc_pi07style_experiment.md) | 仿 pi0.7 风格 AWBC 变体 |
| Advantage Estimator + visual subgoal pipeline | [advantage_pipeline_and_visual_subgoal.md](plans/advantage_pipeline_and_visual_subgoal.md) | AWBC 上游 advantage 训练 + visual subgoal |
| Stage classifier 训练 plan | [stage_classifier_plan.md](plans/stage_classifier_plan.md) | 阶段分类器 (AWBC 状态分段) |

### Hardware-specific 训练 plan

| 计划 | 文件 | 备注 |
|---|---|---|
| gf0 normal training plan | [gf0_normal_training_plan.md](plans/gf0_normal_training_plan.md) | gf0 8×A100 baseline 训练流程 |
| gf1 training plan | [gf1_training_plan.md](plans/gf1_training_plan.md) | gf1 8×A100 训练流程 |
| gf2 advantage + AWBC plan | [gf2_advantage_awbc_plan.md](plans/gf2_advantage_awbc_plan.md) | gf2 跑 advantage estimator + AWBC |
| Multinode distributed training | [multinode_distributed_training_plan.md](plans/multinode_distributed_training_plan.md) | 多节点分布式训练设计 |
| Parallel execution plan | [parallel_execution_plan.md](plans/parallel_execution_plan.md) | 多机并发执行调度 |

### Task E 系列

| 计划 | 文件 | 备注 |
|---|---|---|
| Task E master plan | [task_e_master_plan.md](plans/task_e_master_plan.md) | Phase 0 baseline + Phase 1 差异化 2×2 + Phase 1-FT + Phase 2 vision LoRA. ⭐ Task E 系列总入口 |

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
