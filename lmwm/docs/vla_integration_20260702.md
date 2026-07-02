# LMWM ↔ VLA Integration Interface (Phase D)

> Date: 2026-07-02. Packages the real-future-validated LMWM (Phases A/B/C) into a
> single online predictor for use as a VLA planning prior.

## Why now

Phase C showed the standalone single-frame world model is at its ceiling: frame
history does not improve real-future prediction, and the remaining levers (action
conditioning, less-jittery labels) live on the VLA / data side. So the right next
move is to expose the current, honestly-validated LMWM to a policy — not to keep
tuning the world model alone.

## The recipe (validated vs the real observed next milestone)

```
p_cal   = softmax(greedy_logits / T),         T = 1.30
p_prior = transition_probs[current_milestone]
p_fused ∝ p_cal^(1-λ) · p_prior^λ,            λ = 0.30
```

`T=1.30` and `λ=0.30` are the held-out optima from Phase B. On held-out episodes,
vs the real observed next milestone: top1 ≈ **0.42**, top3 ≈ **0.73**, top5 ≈
**0.86**, NLL ≈ **1.80**, calibrated ECE ≈ **0.005**. These are honest numbers —
scored against reality, not the circular graph-lookup metric.

## API

```python
from lmwm.vla_interface import VLALMWMPredictor

predictor = VLALMWMPredictor.from_yaml(
    "lmwm/configs/inference/kai0base_dinov3h_vla_realfuture.yaml"
)
out = predictor.predict(current_features)          # (B, feature_dim) or (feature_dim,)
# or predictor.predict_one(current_feature)
```

`current_milestones` is optional; if omitted it is assigned by nearest DINOv3-H
prototype (uses the last `frame_dim` dims, so history-augmented inputs also work).

### Returned fields (per sample)

| field | shape | meaning |
|---|---|---|
| `next_milestone` | () | argmax of the fused distribution (top-1 next milestone) |
| `next_milestone_probs` | (M,) | frame-conditional distribution with soft graph prior — the main output |
| `topk_milestones` / `topk_probs` | (k,) | ranked next-milestone candidates + probabilities |
| `subgoal_latent` | (latent_dim,) | L2-normalized prototype subgoal for the greedy next milestone |
| `confidence` | () | max fused prob (calibrated; usable for gating) |
| `entropy` | () | entropy of the fused distribution (task-branch uncertainty) |
| `calibrated_probs` | (M,) | neural-only temperature-scaled distribution (no prior) |
| `current_milestone` | () | assigned/observed current stage |

## Suggested VLA usage

- **Subgoal conditioning**: feed `subgoal_latent` (a DINOv3-H prototype) as a
  latent visual goal, LaWAM-style.
- **Candidate set, not a hard target**: prefer `topk_milestones` / probabilities
  over a single next milestone — the problem has ~13 branches, so top-1 is often
  wrong while top-5 covers 86%.
- **Gate on `confidence` / `entropy`**: defer to the policy's own priors when the
  LMWM is uncertain (high entropy / low confidence).
- Do **not** treat top-1 as ground truth, and do not use the old graph-hybrid
  `runtime.UnifiedLMWMPredictor` fallback numbers (0.997) — those were validated
  against the graph that defined their labels.

## Config

`configs/inference/kai0base_dinov3h_vla_realfuture.yaml`:
`checkpoint` (single-frame real-future best), `graph_npz`, `temperature: 1.30`,
`prior_weight: 0.30`, `topk: 5`.

## Honest limits carried into VLA

- Single `kai0_base` DINOv3-H, same-task held-out; `T` / `λ` re-fit per dataset.
- Output is a milestone distribution + latent prototype subgoal — not decoded
  images or robot actions.
- Absolute top-1 ≈ 0.42 reflects intrinsic ~13-branch ambiguity; rely on top-k +
  confidence, not point prediction.

## Status

**LMWM is ready for VLA integration.** The interface exposes a calibrated,
real-future-validated next-milestone distribution, ranked candidates, a latent
subgoal, and uncertainty signals through one online predictor.
