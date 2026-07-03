# LMWM

Latent Milestone World Model (LMWM) is the CRAVE-side project for training a
recurrence state world model over task-aware milestone states.

The repository layout follows the local `kai0` convention:

- `src/lmwm/`: first-party Python package.
- `scripts/`: runnable data, training, evaluation, and export entrypoints.
- `configs/`: YAML configs for datasets, models, and training runs.
- `docs/`: design notes and experiment plans.
- `data/`: local manifests or lightweight derived metadata. Large datasets stay
  in the shared workspace data roots.
- `checkpoints/`: local model outputs.
- `logs/`: training and evaluation logs.
- `vendor/LaWAM/`: unmodified upstream LaWAM reference implementation.

The first milestone is not a full VLA policy. It is a standalone recurrence
state world model:

```text
CRAVE demo videos -> DINO/state features -> milestone ids/prototypes
-> milestone sequence dataset -> next-state / path-to-completion model
-> latent milestone subgoal interface for VLA/PI-style policies
```

See `docs/recurrence_state_world_model_plan.md` for the current plan.

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/lmwm_final_architecture_20260703.md](docs/lmwm_final_architecture_20260703.md) | **★★最终架构(定稿,单一事实源)** — 感知/milestone预测器/VLA subgoal接口、参数量IO、指标vs baseline、收紧决策、跨外观泛化 |
| [docs/pitfalls_and_lessons_20260703.md](docs/pitfalls_and_lessons_20260703.md) | **试错与踩坑记录** — 建模负结果 + gf3 工程坑 + 验证方法库(防重复踩坑) |
| [docs/architecture_research_20260703.md](docs/architecture_research_20260703.md) | milestone+1 架构调研(JEPA/层级/LaWM/扩散)+ 起步方案 |
| [docs/lmwm_architecture_20260703.md](docs/lmwm_architecture_20260703.md) | 架构框架(前一版收紧) — augin 输入、UnifiedLMWM、subgoal 双损失、集成/蒸馏 |
| [docs/lmwm_technical_report.md](docs/lmwm_technical_report.md) | **技术报告** — 方法 + 配图结果 + 解码预测可视化 |
| [docs/lmwm_stage_overview.md](docs/lmwm_stage_overview.md) | 阶段总览 — 所有阶段、结果和诚实局限的单页导览 |
| [docs/phase_a_real_future_20260702.md](docs/phase_a_real_future_20260702.md) | Phase A — 真实未来标签 + 图无关评估(0.94→0.23 诚实重塑) |
| [docs/phase_b_calibration_prior_20260702.md](docs/phase_b_calibration_prior_20260702.md) | Phase B — 校准(ECE 0.10→0.005)+ 图作软先验融合(λ=0.3) |
| [docs/phase_c_history_20260702.md](docs/phase_c_history_20260702.md) | Phase C — 帧历史条件(负结果:无增益) |
| [docs/vla_integration_20260702.md](docs/vla_integration_20260702.md) | Phase D — VLA 集成接口(`lmwm.vla_interface.VLALMWMPredictor`) |
| [docs/next_milestone_vla_validation_plan.md](docs/next_milestone_vla_validation_plan.md) | **Phase E plan** — next-milestone 提示对 VLA 是否有效:调研+决定性验证(GT 子目标先行, kill criteria) |
| [docs/lawm_reference_20260702.md](docs/lawm_reference_20260702.md) | LaWM/LAM 训练配方参考 + 在 LMWM 数据上的损失测试 + 优化建议 |
| [docs/ceiling_analysis_20260702.md](docs/ceiling_analysis_20260702.md) | 是否到瓶颈?深度分析(kNN 上界 + 标签/混叠诊断)+ 突破方法 |
| [docs/mean_variance_research_20260702.md](docs/mean_variance_research_20260702.md) | 均值+方差双降研究(7B否决;ensemble+CVaR+code因子化 → top1 0.453/std −40%) |
| [docs/optimization_plan_20260702.md](docs/optimization_plan_20260702.md) | **自动迭代优化(L1–L7 滚动日志)** — patch-token/多假设/容量/蒸馏/温度/异构;终态 top1 0.459、部署单模型蒸馏 0.449、frame-only 天花板论证 |
| [docs/episode_medoid_target_analysis_20260702.md](docs/episode_medoid_target_analysis_20260702.md) | 提议分析:用 episode-local medoid latent 作 milestone 目标(实测 0.877 > 簇心 0.836) |
| [docs/recurrence_state_world_model_plan.md](docs/recurrence_state_world_model_plan.md) | 设计规划 — 设计原理、LaWAM 参考点、分阶段路线 |
| [docs/automatic_iteration_log_20260701.md](docs/automatic_iteration_log_20260701.md) | 自动迭代日志 — 每一步实验的详尽记录 |
| [docs/stage1_dinov3h_run_20260701.md](docs/stage1_dinov3h_run_20260701.md) | Stage-1 DINOv3-H 训练记录 |
| [docs/stage1_smoke_run_20260701.md](docs/stage1_smoke_run_20260701.md) | Stage-1 smoke 运行记录 |

## Terminology Lock

Use these names consistently in docs, figures, configs, and runtime summaries:

- `Greedy`: one-step local prediction, `argmax P(stage_{t+1} | stage_t)`.
- `Max-product`: finite-horizon dynamic programming / max-product search toward
  the terminal milestone; the reported milestone is the next step on that path.

Do not use `Max-Probability Milestone` for the one-step rule. That wording is
ambiguous and was retired to avoid swapping the two meanings.

## Current Best LMWM Artifact

Current best checkpoint:

`lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt`

It maps current DINOv3-H frame features to:

- recurrence transition probability row;
- Greedy next milestone: one-step `argmax P(stage_{t+1} | stage_t)`;
- Max-product next milestone: finite-horizon dynamic programming / max-product
  search toward the terminal milestone, then take the next step on that path;
- Greedy latent prototype subgoal;
- max-product latent prototype subgoal;
- confidence and entropy signals for downstream VLA gating.

Validated neural inference output:

`lmwm/outputs/stage3_unified_inference/20260701_best/summary.json`

Recommended hybrid inference config:

`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_recommended.yaml`

Validated recommended hybrid output:

`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/summary.json`

Validation-selected safe hybrid config:

`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_validation_selected.yaml`

This policy selects weak milestones from held-out per-milestone metrics. It has
higher fallback use and should be treated as a safer but more graph-prior-heavy
option than the recommended balanced config.

Learned uncertainty prototype config:

`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_learned_uncertainty.yaml`

This policy uses a lightweight learned error-risk model instead of a hard weak
milestone list. It is a prototype for reducing manual fallback rules; keep the
balanced config as the default until the learned policy is validated on broader
held-out data.

Tuned learned uncertainty config:

`lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_learned_tuned.yaml`

This version overrides learned error thresholds to cover the previous weak
milestone #34 while keeping fallback around 21%. It is the current strongest
learned-fallback candidate, but still needs broader held-out validation before
replacing the balanced default.

Full 200k-pair policy comparison:

`lmwm/outputs/runtime_eval/20260702_policy_comparison_summary.json`

On the full exported kai0_base DINOv3-H pair set, the balanced recommended
policy remains the default tradeoff. The validation-selected safe policy is more
accurate but relies much more on graph fallback; the tuned learned policy is the
current learned-fallback candidate but still trails the balanced policy on full
set mean top1.

The hybrid interface keeps neural predictions and adds graph fallback fields:

- `hybrid_greedy`
- `hybrid_max_product`
- `hybrid_greedy_subgoal_latent`
- `hybrid_max_product_subgoal_latent`
- `greedy_fallback_mask`
- `max_product_fallback_mask`

The graph fallback is a planning prior, not an independent learned correction.
Use fallback masks and confidence scores when connecting to VLA policy inputs.

## Runtime API

Use `lmwm.runtime.UnifiedLMWMPredictor` for online VLA-facing inference.

Smoke-tested output:

`lmwm/outputs/runtime_smoke/20260701_hybrid_recommended/summary.json`

Batch/runtime consistency check:

`lmwm/outputs/runtime_smoke/20260701_hybrid_recommended/batch_runtime_compare.json`

Confidence calibration:

`lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/summary.json`

Minimal usage:

```python
from lmwm.runtime import UnifiedLMWMPredictor

predictor = UnifiedLMWMPredictor.from_yaml(
    "lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_recommended.yaml"
)
result = predictor.predict(current_features, current_milestones)
```

`current_milestones` is optional. If omitted, the predictor assigns the current
stage by nearest DINOv3-H milestone prototype.
