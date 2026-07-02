# LMWM Stage-1 Smoke Run - 2026-07-01

## Resource Split

Machine has 2 x NVIDIA A100-SXM4-80GB.

- GPU0: fixed-horizon Stage-1A/B smoke run.
- GPU1: next-unique-milestone Stage-1C smoke run.

For this small model, two independent single-GPU runs are more efficient than
DDP because the model has only ~69k parameters and communication overhead would
dominate.

## Data Export

Source artifact:

```text
temp/crave_interp_ep2302_30hz_decoded/_cache.npz
```

Commands:

```bash
python lmwm/scripts/export_crave_sequences.py \
  --config lmwm/configs/datasets/ep2302_smoke.yaml

python lmwm/scripts/export_crave_sequences.py \
  --config lmwm/configs/datasets/ep2302_next_unique_smoke.yaml
```

Outputs:

- `lmwm/data/crave_sequences/ep2302_smoke/pairs_fixed_h30.npz`
  - 2930 pairs
  - 39 milestones
- `lmwm/data/crave_sequences/ep2302_smoke/pairs_next_unique.npz`
  - 2953 pairs
  - 39 milestones

Important limitation:

```text
prototype_source = one_hot_progress_smoke
```

This smoke run uses `[one_hot(milestone_id), progress]` as the prototype vector.
It verifies the LaWM-shaped training loop and file management, but it is not yet
the full CRAVE prototype-latent experiment.

## Training Runs

### Fixed Horizon h=30

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/ep2302_stage1ab_smoke.yaml
```

Run dir:

```text
lmwm/logs/stage1ab/20260701_135327+ep2302_stage1ab_smoke
```

Checkpoint dir:

```text
lmwm/checkpoints/stage1ab/20260701_135327+ep2302_stage1ab_smoke
```

Final metrics:

```json
{
  "step": 600,
  "train_loss": 0.04764937236905098,
  "val_loss": 0.040797843957008355,
  "val_mse": 0.014962628426557921,
  "val_ce": 0.05167043106090087,
  "val_top1": 0.9965870307167235,
  "val_top3": 0.9982935153583617
}
```

### Next Unique Milestone

Command:

```bash
CUDA_VISIBLE_DEVICES=1 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/ep2302_stage1c_next_unique_smoke.yaml
```

Note: when using `CUDA_VISIBLE_DEVICES=1`, the process sees the physical GPU1 as
`cuda:0`, so the config uses `device: cuda:0`.

Run dir:

```text
lmwm/logs/stage1c/20260701_135353+ep2302_stage1c_next_unique_smoke
```

Checkpoint dir:

```text
lmwm/checkpoints/stage1c/20260701_135353+ep2302_stage1c_next_unique_smoke
```

Final metrics:

```json
{
  "step": 600,
  "train_loss": 0.034386344254016876,
  "val_loss": 0.04841142873927421,
  "val_mse": 0.023584733324758897,
  "val_ce": 0.049653390331064584,
  "val_top1": 0.9983079526226735,
  "val_top3": 0.9983079526226735
}
```

## Interpretation

This validates:

- project structure;
- config-driven run management;
- CRAVE `_cache.npz` to pair dataset export;
- LaWM-shaped inverse transition encoder and forward decoder;
- milestone classification head;
- logging and checkpoint output.

This does not yet validate:

- learning over true CRAVE prototype latents;
- cross-episode generalization;
- action-conditioned dynamics;
- graph planning or max-completion path inference.

## Next Step

Replace the smoke prototype vector with real CRAVE milestone prototype latents.
The next export target should include:

```text
prototype_latent[milestone_id] -> r_t raw input
milestone_id / progress        -> auxiliary metadata only
episode-level train/val split  -> no random frame split leakage
```

