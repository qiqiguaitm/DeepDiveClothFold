# Phase C — Does frame-history help? (negative result)

> Date: 2026-07-02. Tests whether temporal context beyond the current frame
> improves real-future next-milestone prediction. All numbers on the held-out
> episode split (40,572 pairs / 611 episodes / 37 milestones), scored against the
> **real observed** next milestone. Same split and real-future targets as
> Phase A/B (history datasets are row-aligned augmentations of the same pairs).

## Setup

`scripts/export_history_pairs.py` replaces each pair's single `current` frame
with the concatenation of `[t-(H-1)·S, ..., t-S, t]` DINOv3-H frames from the raw
cache; all other fields (milestones, real future, episode_id) are copied
verbatim. Two spans trained with `label_source: real_future` (same config as
Phase A otherwise):

- H=4, stride=2 (spans ~6 frames back)
- H=6, stride=4 (spans ~20 frames back)

## Results (neural greedy head vs real future)

| model | in_dim | top1 | top3 | top5 | NLL | best fusion top1 / NLL |
|---|---|---|---|---|---|---|
| single-frame | 1280 | **0.383** | 0.686 | 0.822 | **1.978** | 0.417 / 1.798 |
| history H4 s2 | 5120 | 0.377 | 0.686 | 0.826 | 1.979 | 0.421 / 1.782 |
| history H6 s4 | 7680 | 0.367 | 0.671 | 0.815 | 2.024 | 0.417 / 1.809 |

## Findings

1. **History does not help.** All differences are within noise (±0.006 top1, NLL
   identical at H4). The longer span (H6 s4) is slightly *worse* on the neural
   head.
2. **Larger inputs overfit faster.** Both history models peak early (val top1 at
   step ~600) then decline while train loss keeps dropping — the 4–6× larger
   input over the same 200k pairs overfits.
3. **Single-frame is at its ceiling.** The current DINOv3-H frame already captures
   the milestone state; recent raw frames add nothing. The ~13-branch entropy of
   the next milestone is **intrinsic task ambiguity**, not a missing-context
   artifact.

Side note: the DINOv3-H milestone assignment changes roughly every ~2 frames
(mean 54 compressed stages over ~110 frames/episode), so "next unique milestone"
is inherently jittery — which is exactly why CRAVE needs temporal smoothing
(Viterbi / sym-adaptive-vote) for its own read-out. History over such jittery
per-frame labels does not add signal.

## Implication

Further single-frame / short-history LMWM modeling has diminishing returns. The
remaining levers are (a) action conditioning and (b) better/less-jittery labels —
both of which live on the VLA / data side. **The natural next step is to integrate
the current calibrated LMWM into a VLA as a planning prior**, not to keep tuning
the standalone world model.

## Artifacts

- Datasets: `data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_hist{4s2,6s4}.npz`
- Checkpoints: `checkpoints/stage3_realfuture_hist/*`
- Evals: `outputs/real_future_eval/realfuture_hist{4s2,6s4}/`,
  `outputs/phase_b_eval/realfuture_hist{4s2,6s4}/`
