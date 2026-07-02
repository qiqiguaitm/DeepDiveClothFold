# ICRA-Oriented LMWM Experiment Snapshot

Generated from existing local artifacts. This is a paper-table snapshot, not a
new training run.

## Terminology

- **Greedy**: one-step local prediction, `argmax P(stage_{t+1} | stage_t)`.
- **Max-product**: finite-horizon dynamic programming / max-product search
  toward the terminal milestone; report the next step on that path.

## Recurrence Graphs

| Graph | Frames/Pairs | Episodes | Milestones | Edges | Mean compressed len | Terminal | Terminal progress |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kai0base_dinov3h | 334875 | 3055 | 37 | 1232 | 54.0815 | #36 | 0.9541 |
| kai0bd_feature_stage1 | 44967 | 501 | 64 | 1562 | - | #63 | 0.9616 |

## Training Stages

| Stage | Pairs | Input dim | Milestones | Step | Top1 | Greedy top1 | Max-product top1 | KL | Greedy proto cos | Max-product proto cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1ab_fixed | 200000 | 1280 | 37 | 1200 | 1.0000 | - | - | - | - | - |
| stage1c_next_unique | 200000 | 1280 | 37 | 1200 | 1.0000 | - | - | - | - | - |
| stage1d_frame2proto | 200000 | 1280 | 37 | 1200 | 1.0000 | - | - | - | - | - |
| stage2_graph_policy | 200000 | 1280 | 37 | 1200 | - | 0.9271 | 0.9234 | 0.1604 | - | - |
| stage3_unified | 200000 | 1280 | 37 | 1200 | - | 0.9274 | 0.9246 | 0.1595 | 0.9888 | 0.9908 |
| kai0bd_stage1ab_fixed | 41655 | 796 | 64 | 1200 | 1.0000 | - | - | - | - | - |
| kai0bd_stage1c_next_unique | 44967 | 796 | 64 | 1200 | 1.0000 | - | - | - | - | - |
| kai0bd_stage2_graph_policy | 44967 | 796 | 64 | 1200 | - | 1.0000 | 1.0000 | 0.0007 | - | - |
| kai0bd_stage3_unified | 44967 | 796 | 64 | 1200 | - | 1.0000 | 1.0000 | 0.0006 | 1.0000 | 1.0000 |

## Runtime Policies

| Policy | Samples | Mean top1 | Greedy top1 | Max-product top1 | Mean fallback | Greedy fallback | Max-product fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected_safe | 200000 | 0.9976 | 0.9974 | 0.9977 | 0.3315 | 0.3265 | 0.3365 |
| recommended | 200000 | 0.9965 | 0.9963 | 0.9967 | 0.2002 | 0.1943 | 0.2062 |
| learned_tuned | 200000 | 0.9955 | 0.9958 | 0.9952 | 0.2197 | 0.2227 | 0.2167 |
| kai0bd_recommended | 44967 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## Paper Read

- `recommended` remains the balanced default: high mean top1 with lower fallback
  than the graph-prior-heavy safe policy.
- `validation_selected_safe` is the highest-accuracy safety mode, but it relies
  more heavily on graph fallback.
- `learned_tuned` is the strongest learned-fallback candidate, but it does not
  replace the default until it wins on broader held-out or cross-task data.

## Current Gap To ICRA-Ready Evidence

- Promote the new `kai0bd` evidence from Stage-1/2/3 pipeline validation into
  an independent held-out or cross-task evaluation.
- Add independent criteria beyond graph-table labels.
- Add VLA-side ablations showing whether latent milestone subgoals improve
  downstream execution or data selection.
