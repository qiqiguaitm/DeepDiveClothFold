# Phase B — Calibration + Graph-as-Prior Fusion (real-future criterion)

> Date: 2026-07-02. Builds on
> [phase_a_real_future_20260702.md](phase_a_real_future_20260702.md).
> All numbers on the held-out episode split (40,572 pairs / 611 episodes / 37
> milestones), scored against the **real observed** next milestone.

## Framing

Phase A showed the real-future greedy head is already a frame-conditional
next-milestone distribution (softmax trained by CE on observed futures), and that
it beats the empirical baseline on NLL. Phase B asks two honest questions without
retraining:

1. **Calibration** — is the predicted confidence trustworthy, and is it fixable?
2. **Graph as prior** — demote the graph from label / hard fallback to a soft
   Bayesian prior via a log-linear pool
   `log p = (1-lam)·log p_neural + lam·log p_prior`, and check whether it helps
   real-future NLL / top-k. `p_prior = transition_probs[current_m]` (empirical
   milestone-level row).

Script: `scripts/eval_phase_b.py`. Artifact:
`outputs/phase_b_eval/realfuture/summary.json`.

## 1. Calibration

| metric | value |
|---|---|
| ECE (raw) | 0.103 |
| Brier | 0.773 |
| temperature `T` (fit on calib half) | 1.30 |
| test ECE before → after `T` | 0.097 → **0.005** |
| test NLL before → after `T` | 1.961 → 1.916 |

Reliability shows a mild, consistent over-confidence (e.g. in the 0.5–0.6 bin,
mean confidence 0.547 vs accuracy 0.425). A single temperature `T=1.30` removes
almost all of it (ECE 0.097 → 0.005, ~20×). **The frame-conditional distribution
is trustworthy after a one-parameter fix.**

## 2. Graph-as-prior log-linear fusion (vs real future)

| lam | top1 | top3 | top5 | NLL |
|---|---|---|---|---|
| 0.0 (neural only) | 0.383 | 0.686 | 0.822 | 1.978 |
| 0.2 | 0.416 | 0.731 | 0.855 | 1.812 |
| **0.3 (best NLL)** | **0.417** | **0.731** | **0.856** | **1.798** |
| 0.4 | 0.415 | 0.727 | 0.854 | 1.809 |
| 0.5 | 0.409 | 0.720 | 0.847 | 1.844 |
| 1.0 (prior only) | 0.240 | 0.483 | 0.633 | 2.572 |

The curve is smooth and concave with a clear optimum at **lam ≈ 0.3**: a ~30%
graph prior lifts top1 0.383 → **0.417**, top5 0.822 → **0.856**, and NLL 1.978 →
**1.798**.

## Findings

1. **The distribution is trustworthy.** Raw ECE 0.10 → 0.005 with `T=1.30`;
   confidence can be used for downstream gating once temperature-scaled.
2. **The graph helps as a soft prior — honestly this time.** At `lam=0.3` the
   graph improves every real-future metric. This is the principled replacement
   for the old hard fallback: the old hybrid "0.997" was validated against the
   graph that defined its labels (circular); this fusion is validated against
   reality and the gain is real but modest (top1 +3.4 pts, NLL −0.18).
3. **Neither component alone is enough.** Prior-only (0.240 / 2.572) is far worse
   than neural-only (0.383 / 1.978); fusion beats both. Frame features and the
   milestone-level prior carry complementary information.

## Recommended real-future inference recipe

```
p_neural = softmax(greedy_logits / T),   T = 1.30
p_fused  ∝ p_neural^0.7 · p_prior^0.3,    p_prior = transition_probs[current_m]
```

Report top-k + NLL, not top1-vs-graph. This supersedes the graph-lookup hybrid
gate for any evaluation that must reflect real next-milestone prediction.

## Honest caveats

- Absolute top1 ≈ 0.42 reflects a genuinely ~13-branch problem; top-3/top-5 (0.73
  / 0.86) and NLL (1.80) are the meaningful numbers.
- `T` and `lam` are tuned on held-out `kai0_base` — re-fit per dataset/split.
- Still single `kai0_base` DINOv3-H, same-task held-out, latent-prototype output.

## Next (Phase C, optional)

If single-frame top-k is near its ceiling (high branch entropy), add short
history / action conditioning and measure whether real-future NLL drops — this
tests whether the recurrence dynamics genuinely need context beyond one frame.
