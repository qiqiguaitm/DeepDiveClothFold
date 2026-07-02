# Phase A — Real-Future Labels and Graph-Independent Evaluation

> Date: 2026-07-02.
> Goal: break LMWM out of the circular "predict-the-graph-table" setup by (1)
> training against the **real observed next milestone** and (2) evaluating
> against reality on held-out episodes, independent of the graph.

## Motivation

The exported pair dataset already stores `future_milestone` — the actually
observed next-unique milestone after frame `t`. But every prior trainer ignored
it and trained the greedy / max-product heads on `greedy_next[current_m]`, a
deterministic **graph-table lookup** indexed by the discretized current state.

Two problems:

1. The graph-lookup label matches the real observed future only **24.2%** of the
   time (75.8% of pairs disagree). The target discards the real trajectory.
2. Given the current state, the real next milestone has entropy **~2.56 nats
   (~13 effective branches)** — highly multimodal. A single argmax cannot be
   right most of the time, so "top1 vs graph = 0.96" was measuring reproduction
   of the table, not prediction of reality.

## What changed

- `src/lmwm/data.py`: `load_graph_policy_data(..., label_source=...)` with
  `"graph_lookup"` (default, backward compatible) and `"real_future"` (greedy /
  max-product / prototype heads target `future_milestone`). The transition head
  always targets the empirical milestone-level distribution
  `transition_probs[current_m]`, so its real-future NLL is comparable across
  label sources.
- `scripts/train_unified_lmwm.py`: reads `label_source` from config, records it
  in run meta.
- `configs/training/kai0base_dinov3h_stage3_realfuture.yaml`: real-future run.
- `scripts/eval_real_future.py`: graph-independent evaluator. On the held-out
  episode split it scores predictions against `future_milestone` (top-1/3/5,
  NLL) plus non-neural baselines. Works on any `UnifiedLMWM` checkpoint.

## Results (held-out: 40,572 pairs / 611 episodes / 37 milestones)

| Model / baseline | vs graph (circular) top1 | vs REAL top1 | top3 | top5 | NLL |
|---|---|---|---|---|---|
| uniform | — | 0.024 | 0.057 | 0.109 | 3.61 |
| graph argmax (greedy) | — | 0.240 | — | — | — |
| empirical dist `P(next\|cur milestone)` | — | 0.240 | 0.483 | 0.633 | 2.57 |
| graph-trained · neural greedy head | **0.936** | 0.233 | 0.366 | 0.474 | 16.04 |
| graph-trained · neural transition head | — | 0.207 | 0.434 | 0.582 | 2.68 |
| **real-future-trained · neural greedy head** | 0.275 | **0.383** | **0.686** | **0.822** | **1.98** |
| real-future-trained · neural transition head | — | 0.201 | 0.438 | 0.594 | 2.67 |

Artifacts:

- Graph-trained eval: `outputs/real_future_eval/graph_trained/summary.json`
- Real-future-trained checkpoint:
  `checkpoints/stage3_realfuture/20260702_045756+kai0base_dinov3h_stage3_realfuture/best.pt`
- Real-future-trained eval: `outputs/real_future_eval/realfuture_trained/summary.json`

## Findings

1. **The 0.96 was circular.** The graph-trained greedy head scores 0.936 vs the
   graph table but only **0.233 vs reality** — barely above the non-neural
   baselines (graph argmax 0.240, empirical 0.240) and its NLL of **16.0** shows
   a pathologically overconfident point predictor.
2. **Real-future training genuinely helps.** vs reality, the greedy head goes
   from 0.233 → **0.383** top1, 0.474 → **0.822** top5, and NLL 16.0 → **1.98**.
3. **Frame features carry dynamics beyond the milestone id.** The real-future
   greedy head's NLL (1.98) now **beats the empirical milestone-level
   distribution baseline** (2.57). This is the first evidence that LMWM is
   learning something the current-milestone lookup does not already contain —
   i.e. moving from "table validation" toward a real world model.
4. **The transition head is capped by its target.** Both models' transition
   heads (~2.67 NLL) sit at or slightly above the empirical baseline (2.57),
   because they are trained to match exactly that milestone-level distribution.
   To exceed it, the distributional target must itself become frame-conditional
   (Phase B).

## Honest caveats

- Real-future top1 of 0.38 is not "good" in absolute terms — but the problem has
  ~13 branches, so top-3/top-5 and NLL are the meaningful metrics, and there the
  model is clearly informative.
- Still single `kai0_base` DINOv3-H, same-task held-out, latent-prototype output
  (not decoded images or actions).

## Next (Phase B)

Make the **distribution** frame-conditional: train a distributional next-milestone
target from real futures (soft labels / direct CE against observed), evaluate NLL
+ calibration, and use the graph only as a prior — not as ground truth. Re-derive
the fallback logic under the real-future criterion (the old hybrid 0.997 does not
survive it).
