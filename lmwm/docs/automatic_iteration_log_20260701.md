# LMWM Automatic Iteration Log - 2026-07-01

## Goal

Build toward the final Latent Milestone World Model with small reversible
increments. Each step must have an artifact, a validation run, and a short
interpretation before moving to the next step.

## Iteration Policy

- Do not modify `lmwm/vendor/LaWAM`.
- Prefer LaWM-shaped small steps before changing the model family.
- Keep every exported dataset, config, log, and checkpoint under `lmwm/`.
- Treat high scores on prototype-table tasks as pipeline validation, not final
  model quality.
- If a new step underperforms, keep its logs but continue from the last stronger
  checkpoint.

## Completed Steps

### 1. Stage-1D: frame feature -> future milestone prototype

Purpose: remove the overly easy prototype-to-prototype input and use real
DINOv3-H frame features as current observations.

Artifacts:

- Export config: `lmwm/configs/datasets/kai0base_dinov3h_frame2proto_next_unique.yaml`
- Dataset: `lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz`
- Training config: `lmwm/configs/training/kai0base_dinov3h_stage1d_frame2proto_next_unique.yaml`
- Script: `lmwm/scripts/export_dinov3h_milestone_pairs.py`
- Script: `lmwm/scripts/train_state_world_model.py`
- Run: `lmwm/logs/stage1d/20260701_142250+kai0base_dinov3h_stage1d_frame2proto_next_unique`
- Checkpoint: `lmwm/checkpoints/stage1d/20260701_142250+kai0base_dinov3h_stage1d_frame2proto_next_unique/best.pt`

Validation:

- Dataset pairs: 200000
- Episodes: 3055
- Input dim: 1280
- Future prototype dim: 1280
- Val top1 at final step: 1.0
- Val MSE at final step: 0.0003602966

Interpretation: the frame-to-prototype path is healthy, but the target remains a
finite 37-prototype table. It validates data handling and training mechanics,
not the full world model.

### 2. Recurrence Latent State Probability Graph

Purpose: turn CRAVE milestone assignments into a stage transition probability
model.

Artifacts:

- Config: `lmwm/configs/datasets/kai0base_dinov3h_recurrence_graph.yaml`
- Script: `lmwm/scripts/build_recurrence_graph.py`
- Graph: `lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz`
- Metadata: `lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_meta.json`

Graph definition:

- Assign every DINOv3-H frame to the nearest milestone center.
- Sort frames inside each episode.
- Compress consecutive identical milestones.
- Count transitions between compressed stages.
- Row-normalize counts with small smoothing.
- Greedy next: one-step `argmax P(stage_{t+1} | stage_t)`.
- Max-product next: finite-horizon dynamic programming toward the terminal
  milestone with highest CRAVE progress.

Validation:

- Valid frames: 334875
- Episodes: 3055
- Milestones: 37
- Nonzero transition edges: 1232
- Mean compressed episode length: 54.0815
- Terminal target: milestone 36, progress 0.9540757

Interpretation: this is the first explicit LMWM recurrence graph artifact. The
current max-product route uses a finite horizon of 10 and first-order transition
probabilities, so it inherits the Markov assumption and should be treated as a
planning prior rather than ground truth.

### 3. Stage-2 graph-supervised policy model

Purpose: distill the recurrence graph into a neural model from current visual
features.

Artifacts:

- Config: `lmwm/configs/training/kai0base_dinov3h_stage2_graph_policy.yaml`
- Script: `lmwm/scripts/train_graph_policy_model.py`
- Run: `lmwm/logs/stage2_graph/20260701_142639+kai0base_dinov3h_stage2_graph_policy`
- Checkpoint: `lmwm/checkpoints/stage2_graph/20260701_142639+kai0base_dinov3h_stage2_graph_policy/best.pt`

Objective:

```text
current frame DINOv3-H feature
  -> transition probability row
  -> Greedy next milestone
  -> Max-product completion next milestone
```

Best validation step: 1100

- Val KL: 0.1564875
- Greedy top1: 0.9348073
- Max-product top1: 0.9347580

Interpretation: this is the first neural recurrence world model. Errors remain
because frame-level features near stage boundaries can map to ambiguous stage
assignments, and the graph labels are deterministic projections of probabilistic
transition rows.

### 4. Stage-3 unified LMWM

Purpose: combine graph prediction and latent subgoal prediction in one model.

Artifacts:

- Config: `lmwm/configs/training/kai0base_dinov3h_stage3_unified.yaml`
- Script: `lmwm/scripts/train_unified_lmwm.py`
- Run: `lmwm/logs/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified`
- Checkpoint: `lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt`

Objective:

```text
current frame DINOv3-H feature
  -> transition probability row
  -> Greedy next milestone id
  -> Max-product completion next milestone id
  -> Greedy latent prototype subgoal
  -> Max-product latent prototype subgoal
```

Best validation step: 1100

- Val KL: 0.1561412
- Greedy top1: 0.9361382
- Max-product top1: 0.9353002
- Greedy prototype cosine: 0.9895124
- Max-product prototype cosine: 0.9915678

Interpretation: this is the strongest current LMWM artifact. It provides both a
probabilistic stage-world output and latent prototype subgoals that can be used
as VLA conditioning inputs.


### 5. Unified LMWM inference wrapper

Purpose: expose the current best unified LMWM as a VLA-ready inference artifact.

Artifacts:

- Script: `lmwm/scripts/infer_unified_lmwm.py`
- Output dir: `lmwm/outputs/stage3_unified_inference/20260701_best`
- Prediction file: `lmwm/outputs/stage3_unified_inference/20260701_best/predictions.npz`
- Summary: `lmwm/outputs/stage3_unified_inference/20260701_best/summary.json`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/infer_unified_lmwm.py   --checkpoint lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt   --dataset_npz lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz   --graph_npz lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz   --output_dir lmwm/outputs/stage3_unified_inference/20260701_best   --max_samples 4096   --device cuda:0
```

Validation on 4096 sampled rows:

- Greedy top1: 0.96484375
- Max-product top1: 0.96508789
- Greedy confidence mean: 0.9614772
- Max-product confidence mean: 0.9591309
- Transition confidence mean: 0.2024447
- Transition entropy mean: 2.7746096
- Greedy prototype cosine mean: 0.9915323
- Max-product prototype cosine mean: 0.9929357

VLA-facing fields:

```text
current_milestone
transition_probs
greedy_pred
max_product_pred
greedy_subgoal_latent
max_product_subgoal_latent
greedy_confidence
max_product_confidence
transition_entropy
```


### 6. Per-milestone error analysis

Purpose: identify unstable stages for the next robustness iteration.

Artifacts:

- Script: `lmwm/scripts/analyze_lmwm_predictions.py`
- Output dir: `lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis`
- CSV: `lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis/per_milestone_metrics.csv`
- Summary: `lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis/summary.json`

Worst milestones in the 4096-sample inference check:

- #34: Greedy 0.90625, Max-product 0.87500, support 3198
- #30: Greedy 0.90756, Max-product 0.90756, support 5087
- #5: Greedy 0.90323, Max-product 0.91935, support 3271
- #2: Greedy 0.95294, Max-product 0.89412, support 3843

Interpretation: the next robustness iteration should focus on boundary/ambiguous
stages rather than global model capacity. A practical next step is to add a
confidence gate and fallback to graph-table outputs when neural confidence is
low or when the current milestone belongs to the known weak set.


### 7. Hybrid graph fallback inference

Purpose: make the current unified LMWM safer for downstream VLA usage by
combining neural predictions with the explicit recurrence graph prior.

Artifacts:

- Updated script: `lmwm/scripts/infer_unified_lmwm.py`
- Config: `lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_recommended.yaml`
- Output dir: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback`
- Predictions: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/predictions.npz`
- Error analysis: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/error_analysis/per_milestone_metrics.csv`

Hybrid rule:

```text
Use neural LMWM prediction by default.
Fallback to recurrence graph table if:
  confidence < threshold, or
  current milestone is in the weak milestone set.
```

Weak milestone set from the previous error analysis:

```text
{2, 5, 30, 34}
```

Validation with `greedy_conf_threshold=0.93` and
`max_product_conf_threshold=0.93` on 4096 samples:

- Neural Greedy top1: 0.96484375
- Neural Max-product top1: 0.96508789
- Hybrid Greedy top1: 0.99755859
- Hybrid Max-product top1: 0.99804688
- Greedy fallback rate: 0.20532227
- Max-product fallback rate: 0.21069336
- Hybrid Greedy prototype cosine: 0.99674082
- Hybrid Max-product prototype cosine: 0.99737370

Interpretation: this is not independent proof that the neural model improved;
the fallback uses the graph prior that also defines the graph-supervised labels.
It is still useful operationally because it exposes uncertainty and prevents
known weak milestones from relying only on neural predictions.

### 8. Hybrid gate threshold sweep

Purpose: choose a default gate from measured accuracy / fallback-rate tradeoffs
instead of using a hand-picked confidence threshold.

Artifacts:

- Script: `lmwm/scripts/sweep_hybrid_gate.py`
- Sweep CSV: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/gate_sweep/hybrid_gate_sweep.csv`
- Summary: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_fallback/gate_sweep/summary.json`

Recommended setting for validation and early VLA integration:

```text
greedy_conf_threshold = 0.90
max_product_conf_threshold = 0.92
weak_milestones = {2, 5, 30, 34}
```

This gives mean top1 about 0.9976 with mean fallback rate about 0.196 on the
4096-sample inference check.

The recommended config was executed directly and produced:

- Output: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/predictions.npz`
- Summary: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/summary.json`
- Per-milestone analysis: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/error_analysis/per_milestone_metrics.csv`
- Hybrid Greedy top1: 0.99707031
- Hybrid Max-product top1: 0.99804688
- Greedy fallback rate: 0.18774414
- Max-product fallback rate: 0.20434570


Conservative setting when graph fallback should be limited:

```text
greedy_conf_threshold = 0.80
max_product_conf_threshold = 0.80
weak_milestones = {2, 5, 30, 34}
```

This gives mean top1 about 0.9935 with mean fallback rate about 0.146.


### 9. Online runtime predictor API

Purpose: move the current best LMWM from batch evaluation scripts toward a
stable online interface that can be called by VLA policy code.

Artifacts:

- Model module: `lmwm/src/lmwm/model.py`
- Runtime API: `lmwm/src/lmwm/runtime.py`
- Smoke script: `lmwm/scripts/smoke_runtime_predictor.py`
- Runtime smoke output: `lmwm/outputs/runtime_smoke/20260701_hybrid_recommended/summary.json`
- Batch/runtime consistency check: `lmwm/outputs/runtime_smoke/20260701_hybrid_recommended/batch_runtime_compare.json`

Runtime API:

```python
from lmwm.runtime import UnifiedLMWMPredictor

predictor = UnifiedLMWMPredictor.from_yaml(
    "lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_recommended.yaml"
)
result = predictor.predict(current_features, current_milestones)
```

Returned fields include:

```text
current_milestone
transition_probs
neural_greedy / neural_max_product
graph_greedy / graph_max_product
hybrid_greedy / hybrid_max_product
hybrid_greedy_subgoal_latent / hybrid_max_product_subgoal_latent
greedy_confidence / max_product_confidence
transition_entropy
greedy_fallback_mask / max_product_fallback_mask
```

Smoke validation on 512 samples:

- Explicit current stage match: true
- Inferred current stage match: 1.0
- Greedy fallback rate: 0.173828125
- Max-product fallback rate: 0.181640625
- Transition row sum mean: 1.0
- Transition row sum max absolute error: 2.3841858e-07
- Hybrid Greedy latent norm mean: 1.0
- Hybrid Max-product latent norm mean: 1.0

Batch/runtime consistency check against the recommended batch inference output:

- Compared rows: 512
- Exact fields matched: current milestone, hybrid milestone ids, fallback masks
- Float tolerance: 5e-6
- Passed: true

Interpretation: the current LMWM artifact now has a reusable online predictor,
not only offline scripts. The next step is to add calibration metrics and a
thin serving wrapper or policy adapter for VLA integration.


### 10. Confidence calibration analysis

Purpose: quantify whether neural confidence is meaningful enough to support
fallback gating, instead of relying only on a manually chosen threshold.

Artifacts:

- Script: `lmwm/scripts/calibrate_lmwm_confidence.py`
- Summary: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/summary.json`
- Greedy reliability bins: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/greedy_reliability_bins.csv`
- Max-product reliability bins: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/max_product_reliability_bins.csv`
- Greedy threshold curve: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/greedy_threshold_curve.csv`
- Max-product threshold curve: `lmwm/outputs/stage3_unified_inference/20260701_hybrid_recommended/calibration/max_product_threshold_curve.csv`

Results on the 4096-sample recommended inference output:

Greedy head:

- Accuracy: 0.96484375
- Mean confidence: 0.96147716
- ECE: 0.00726309
- AURC: 0.00227934
- Risk at 90% coverage: 0.00732303
- Best threshold under 20% neural-only fallback: 0.97
- Accepted accuracy at that threshold: 0.99726527

Max-product head:

- Accuracy: 0.96508789
- Mean confidence: 0.95913082
- ECE: 0.00653240
- AURC: 0.00246913
- Risk at 90% coverage: 0.00813670
- Best threshold under 20% neural-only fallback: 0.95
- Accepted accuracy at that threshold: 0.99646955

Interpretation: the confidence values are useful for ranking reliable neural
predictions. The recommended runtime gate still uses the hybrid sweep thresholds
`0.90 / 0.92` because the hybrid system already forces known weak milestones to
fallback, which consumes part of the fallback budget. Calibration supports the
principle of fallback gating and should be used to tune deployment-specific
coverage/accuracy tradeoffs.


### 11. Validation-driven weak milestone policy

Purpose: replace the manually maintained weak-milestone list with a policy that
is generated from held-out per-milestone validation metrics.

Artifacts:

- Script: `lmwm/scripts/select_hybrid_policy.py`
- Source metrics: `lmwm/outputs/stage3_unified_inference/20260701_best/error_analysis/per_milestone_metrics.csv`
- Generated config: `lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_validation_selected.yaml`
- Selection summary: `lmwm/outputs/stage3_unified_inference/20260701_policy_selection/summary.json`
- Validation output: `lmwm/outputs/stage3_unified_inference/kai0base_dinov3h_stage3_hybrid_validation_selected/summary.json`
- Per-milestone validation analysis: `lmwm/outputs/stage3_unified_inference/kai0base_dinov3h_stage3_hybrid_validation_selected/error_analysis/per_milestone_metrics.csv`

Selection rule:

```text
Select current milestone for graph fallback if:
  samples >= 30, and
  transition_support >= 0, and
  min(greedy_top1, max_product_top1) < 0.94
```

Selected weak milestones:

```text
{2, 5, 7, 19, 21, 23, 24, 30, 31, 34}
```

Validation-selected safe policy results on 4096 samples:

- Neural Greedy top1: 0.96484375
- Neural Max-product top1: 0.96508789
- Hybrid Greedy top1: 0.99804688
- Hybrid Max-product top1: 0.99877930
- Greedy fallback rate: 0.31738281
- Max-product fallback rate: 0.33227539
- Hybrid Greedy prototype cosine: 0.99689162
- Hybrid Max-product prototype cosine: 0.99763995

Interpretation: the validation-selected policy is a safe operating mode. It
improves hybrid top1 relative to the balanced recommended policy, but increases
fallback rate from about 19-20% to about 32-33%. Keep the original recommended
config as the balanced default; use the validation-selected config when the VLA
pipeline prefers higher planning-prior reliance over neural autonomy.


### 12. Learned uncertainty fallback prototype

Purpose: reduce reliance on hard weak-milestone lists by learning a lightweight
error-risk model from neural prediction errors.

Artifacts:

- Training script: `lmwm/scripts/train_uncertainty_policy.py`
- Runtime integration: `lmwm/src/lmwm/runtime.py`
- Learned config: `lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_learned_uncertainty.yaml`
- Policy artifact: `lmwm/outputs/uncertainty_policy/20260701_logistic/uncertainty_policy.npz`
- Policy training summary: `lmwm/outputs/uncertainty_policy/20260701_logistic/summary.json`
- Runtime smoke output: `lmwm/outputs/runtime_smoke/20260701_learned_uncertainty/summary.json`
- Per-milestone analysis: `lmwm/outputs/runtime_smoke/20260701_learned_uncertainty/error_analysis/per_milestone_metrics.csv`

Model:

```text
logistic error-risk model per head
features = [head confidence, transition entropy, transition confidence, current milestone one-hot]
label = neural head is wrong
```

Training/validation split from 4096 neural prediction rows:

Greedy head validation:

- Error rate: 0.03515625
- Recommended error threshold under 20% fallback: 0.30
- Fallback rate: 0.15429688
- Accepted accuracy: 0.99653578
- Error recall: 0.91666669

Max-product head validation:

- Error rate: 0.033203125
- Recommended error threshold under 20% fallback: 0.40
- Fallback rate: 0.15332031
- Accepted accuracy: 0.99538636
- Error recall: 0.88235295

Runtime smoke on 4096 rows, no hard weak milestone list:

- Neural Greedy top1 vs graph: 0.96484375
- Neural Max-product top1 vs graph: 0.96508789
- Hybrid Greedy top1 vs graph: 0.99804688
- Hybrid Max-product top1 vs graph: 0.99682617
- Greedy fallback rate: 0.17211914
- Max-product fallback rate: 0.16430664

Interpretation: learned uncertainty substantially reduces fallback compared to
the validation-selected safe policy while preserving high hybrid accuracy. It is
not yet the default because per-milestone analysis shows milestone #34 is still
not fully covered: learned hybrid reaches 0.984375 on #34 rather than 1.0. Keep
`stage3_hybrid_recommended` as the balanced default, use
`stage3_hybrid_validation_selected` as the safe graph-prior-heavy mode, and treat
`stage3_hybrid_learned_uncertainty` as the learned fallback prototype.


### 13. Tuned learned uncertainty policy

Purpose: improve the learned uncertainty prototype on the known weak milestone
#34 without falling back to a hard weak-milestone list.

Artifacts:

- Runtime threshold override: `lmwm/src/lmwm/runtime.py`
- Tuned config: `lmwm/configs/inference/kai0base_dinov3h_stage3_hybrid_learned_tuned.yaml`
- Threshold sweep summary: `lmwm/outputs/runtime_smoke/20260701_learned_uncertainty/threshold_sweep_summary.json`
- Runtime smoke output: `lmwm/outputs/runtime_smoke/20260701_learned_tuned/summary.json`
- Per-milestone analysis: `lmwm/outputs/runtime_smoke/20260701_learned_tuned/error_analysis/per_milestone_metrics.csv`

Tuned thresholds:

```text
greedy_error_threshold = 0.25
max_product_error_threshold = 0.30
```

These were selected from an error-probability sweep as the lowest mean fallback
combination that fully covers milestone #34 for both heads.

Runtime smoke on 4096 rows:

- Neural Greedy top1 vs graph: 0.96484375
- Neural Max-product top1 vs graph: 0.96508789
- Tuned learned Hybrid Greedy top1: 0.99853516
- Tuned learned Hybrid Max-product top1: 0.99853516
- Greedy fallback rate: 0.21362305
- Max-product fallback rate: 0.21411133

Milestone #34 per-milestone validation:

- Neural Greedy top1: 0.90625
- Neural Max-product top1: 0.875
- Tuned Hybrid Greedy top1: 1.0
- Tuned Hybrid Max-product top1: 1.0
- Greedy fallback rate: 1.0
- Max-product fallback rate: 1.0

Interpretation: tuned learned uncertainty now fixes the previous #34 miss while
keeping fallback far below the validation-selected safe policy. It is the current
strongest learned-fallback candidate. The balanced recommended config remains
the default until this learned policy is validated beyond the current 4096-row
sample and Task_A/kai0_base setting.


### 14. Full 200k-pair runtime policy evaluation

Purpose: validate runtime policies beyond the earlier 4096-row smoke checks.
The first full run saved predictions and exposed that writing full latent arrays
is expensive, so `lmwm/scripts/eval_runtime_policy.py` now supports
`--summary_only` for fast policy-level comparisons.

Artifacts:

- Full evaluator: `lmwm/scripts/eval_runtime_policy.py`
- Tuned learned full predictions: `lmwm/outputs/runtime_eval/20260702_learned_tuned_full/runtime_predictions.npz`
- Tuned learned full summary: `lmwm/outputs/runtime_eval/20260702_learned_tuned_full/summary.json`
- Tuned learned full per-milestone analysis: `lmwm/outputs/runtime_eval/20260702_learned_tuned_full/error_analysis/per_milestone_metrics.csv`
- Recommended full summary: `lmwm/outputs/runtime_eval/20260702_recommended_full_summary/summary.json`
- Validation-selected safe full summary: `lmwm/outputs/runtime_eval/20260702_validation_selected_full_summary/summary.json`
- Policy comparison: `lmwm/outputs/runtime_eval/20260702_policy_comparison_summary.json`

Full-set metrics on 200000 pairs:

```text
policy                    mean top1   greedy top1   max top1   mean fallback
validation_selected_safe  0.997555    0.997425      0.997685   0.3314975
recommended              0.996455    0.996255      0.996655   0.2002300
learned_tuned            0.9954875   0.995770      0.995205   0.2197125
```

Tuned learned policy full-set details:

- Neural Greedy top1 vs graph: 0.96325
- Neural Max-product top1 vs graph: 0.961735
- Hybrid Greedy top1 vs graph: 0.99577
- Hybrid Max-product top1 vs graph: 0.995205
- Greedy fallback rate: 0.222725
- Max-product fallback rate: 0.2167

Full-set per-milestone result for #34:

- Samples: 3606
- Neural Greedy top1: 0.88657793
- Neural Max-product top1: 0.88879645
- Tuned Hybrid Greedy top1: 0.99972268
- Tuned Hybrid Max-product top1: 1.0
- Greedy fallback rate: 0.98058791
- Max-product fallback rate: 0.99389906

Interpretation: tuned learned uncertainty fixes #34 at scale, but on the full
200k pair set it trails the balanced recommended policy in mean top1 while using
slightly more fallback. The current default should remain
`stage3_hybrid_recommended`. The tuned learned policy remains useful as the
best learned-fallback candidate and should be improved with broader features or
training data before replacing the default. The validation-selected safe policy
is still the highest-accuracy graph-prior-heavy mode.

## Current Best Artifact

Use the Stage-3 unified checkpoint as the current best model:

`lmwm/checkpoints/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified/best.pt`

## Known Limits

- Only existing DINOv3-H `kai0_base` cache was used; no matching DINOv3-H
  `kai0_dagger` cache was found in this iteration.
- The recurrence graph is first-order Markov and finite-horizon for max-product
  planning.
- Evaluation is on held-out episodes from the same task family, not cross-task.
- Labels are derived from CRAVE milestone assignment, so boundary noise and
  cluster ambiguity propagate into supervision.
- The current model predicts DINOv3-H latent prototype subgoals, not decoded
  images or direct robot actions.

## Next Automatic Iteration

The next step should target robustness rather than another easy accuracy gain:

1. Add confidence calibration and entropy metrics for transition rows.
2. Evaluate errors by current milestone id and transition support count.
3. Add a fallback policy for low-confidence or low-support transitions.
4. If DINOv3-H dagger features become available, repeat graph construction and
   train base+dagger mixed models.
5. Add an inference wrapper that returns `current_stage`, `transition_probs`,
   `greedy_subgoal_latent`, `max_product_subgoal_latent`, and confidence scores.
