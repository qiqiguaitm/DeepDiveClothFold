# LMWM Stage-1 DINOv3-H Run - 2026-07-01

## Purpose

Validate the first LaWM-shaped LMWM training path on real CRAVE DINOv3-H
milestone prototypes, instead of the earlier one-hot smoke representation.

This is still a small-step prototype-to-prototype experiment. It verifies that
the data export, episode split, two-GPU execution, checkpointing, and transition
model training path work on 1280D CRAVE features.

## Source Features

- Feature cache: `temp/crave_full_dinov3h`
- Encoder: DINOv3-H
- Feature dimension: 1280
- Valid frames: 334875
- Episodes: 3055
- Milestone file: `temp/crave_full_dinov3h/milestones_uniform_dinov3h.npz`
- Milestone prototypes: 37 DINOv3-H cluster centers

No matching DINOv3-H cache for `kai0_dagger` was found in this pass. This run
uses the existing full `kai0_base` DINOv3-H cache.

## Exported Datasets

Command:

```bash
python lmwm/scripts/export_dinov3h_milestone_pairs.py \
  --config lmwm/configs/datasets/kai0base_dinov3h_fixed.yaml

python lmwm/scripts/export_dinov3h_milestone_pairs.py \
  --config lmwm/configs/datasets/kai0base_dinov3h_next_unique.yaml
```

Outputs:

- `lmwm/data/crave_sequences/kai0base_dinov3h/pairs_fixed_h3.npz`
- `lmwm/data/crave_sequences/kai0base_dinov3h/pairs_next_unique.npz`

Both exports contain 200000 pairs. Each pair is LaWM-shaped:

```text
current DINOv3-H milestone prototype r_t
future DINOv3-H milestone prototype r_{t+h}
current/future milestone id
episode_id for episode-level train/val split
progress_t / progress_future as CRAVE progress metadata
```

## Model Shape

The Stage-1 model intentionally follows the LAWM first-stage pattern:

```text
r_t, r_future -> inverse transition code u_t
r_t, u_t      -> predicted future latent r_hat_future
r_hat_future  -> future milestone classifier
```

This keeps the first version close to LAWM and avoids jumping directly to the
final recurrence graph / planning model.

## Training Commands

GPU0, fixed-horizon Stage-1A/B:

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/kai0base_dinov3h_stage1ab_fixed.yaml
```

GPU1, next-unique Stage-1C:

```bash
CUDA_VISIBLE_DEVICES=1 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/kai0base_dinov3h_stage1c_next_unique.yaml
```

Both configs use `device: cuda:0` because each process sees only its assigned
physical GPU after `CUDA_VISIBLE_DEVICES`.

## Results

Fixed-horizon run:

- Run dir: `lmwm/logs/stage1ab/20260701_140401+kai0base_dinov3h_stage1ab_fixed`
- Checkpoints: `lmwm/checkpoints/stage1ab/20260701_140401+kai0base_dinov3h_stage1ab_fixed`
- Train pairs: 159428
- Val pairs: 40572
- Final step: 1200
- Val loss: 0.0082213071
- Val MSE: 0.0070061434
- Val CE: 0.0024303274
- Val top1: 1.0
- Val top3: 1.0

Next-unique run:

- Run dir: `lmwm/logs/stage1c/20260701_140401+kai0base_dinov3h_stage1c_next_unique`
- Checkpoints: `lmwm/checkpoints/stage1c/20260701_140401+kai0base_dinov3h_stage1c_next_unique`
- Train pairs: 159428
- Val pairs: 40572
- Final step: 1200
- Val loss: 0.0038437987
- Val MSE: 0.0021513989
- Val CE: 0.0033847997
- Val top1: 1.0
- Val top3: 1.0

## Interpretation

The high accuracy is expected for this step because the inputs and targets are
drawn from a finite table of 37 milestone prototypes. This run should be treated
as a pipeline and architecture validation, not as evidence that the final
Latent Milestone World Model is solved.

The next useful step is to make the prediction problem less table-like:

- train on frame-level DINOv3-H features while supervising milestone-prototype
  targets;
- add multi-candidate next milestone likelihood instead of only one target;
- evaluate by held-out episodes and by transition types, especially low-support
  transitions;
- only then introduce recurrence graph planning and Viterbi / max-product route
  supervision.
