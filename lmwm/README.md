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

## Documentation

| Doc | Purpose |
|---|---|
| [docs/lmwm_stage_overview.md](docs/lmwm_stage_overview.md) | **Start here** — single-page tour of all stages, results, and honest limitations |
| [docs/phase_a_real_future_20260702.md](docs/phase_a_real_future_20260702.md) | Phase A — real-future labels + graph-independent evaluation (the 0.96→0.23 honest reframing, and the real-future win) |
| [docs/phase_b_calibration_prior_20260702.md](docs/phase_b_calibration_prior_20260702.md) | Phase B — calibration (ECE 0.10→0.005 via T=1.30) + graph-as-soft-prior fusion (λ=0.3: top1 0.38→0.42, NLL 1.98→1.80) |
| [docs/phase_c_history_20260702.md](docs/phase_c_history_20260702.md) | Phase C — frame-history conditioning (negative result: no gain, single-frame is at ceiling) |
| [docs/vla_integration_20260702.md](docs/vla_integration_20260702.md) | **Phase D — VLA integration interface** (`lmwm.vla_interface.VLALMWMPredictor`); LMWM is VLA-ready |
| [docs/recurrence_state_world_model_plan.md](docs/recurrence_state_world_model_plan.md) | Design rationale, LaWAM reference points, staged plan |
| [docs/automatic_iteration_log_20260701.md](docs/automatic_iteration_log_20260701.md) | Exhaustive step-by-step record of every iteration (artifacts, commands, metrics) |
| [docs/stage1_dinov3h_run_20260701.md](docs/stage1_dinov3h_run_20260701.md) | Stage-1 DINOv3-H prototype run record |
| [docs/stage1_smoke_run_20260701.md](docs/stage1_smoke_run_20260701.md) | Stage-1 smoke run record (one-hot representation) |

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
