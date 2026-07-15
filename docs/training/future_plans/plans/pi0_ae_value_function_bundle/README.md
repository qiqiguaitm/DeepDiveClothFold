# pi0-AE Value Function Bundle

This directory bundles the pi0 Advantage Estimator value-function code and the
checkpoint used by the AWBC/ViVa comparison plan as the optional `pi0-AE` arm.

## Checkpoint

`checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_adv_est_v1_100000` is a symlink to:

```text
/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/100000
```

Expected files:

- `model.safetensors` - model weights
- `metadata.pt` - checkpoint metadata
- `optimizer.pt` - optimizer state

## Code

Key files copied from the source repo:

- `code/kai0/src/openpi/models_pytorch/pi0_pytorch.py`
  - `class AdvantageEstimator`
  - `AdvantageEstimator.sample_values(...)`
- `code/kai0/src/openpi/training/config.py`
  - config name: `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`
- `code/kai0/stage_advantage/annotation/eval.py`
  - labeling entrypoint
  - default `Flatten-Fold / KAI0` checkpoint points to `adv_est_v1/100000`
- `code/kai0/stage_advantage/annotation/evaluator.py`
  - converts value predictions into `relative_advantage`, `absolute_value`,
    and `absolute_advantage`
- `code/kai0/stage_advantage/README.md`
  - stage advantage pipeline notes

## Value Definition

The value function is `AdvantageEstimator.value_head`, an MLP on the transformer
state-token representation. `sample_values(...)` returns:

```text
value_pred = value_head(suffix_out[:, 0, :])
```

The evaluator writes:

```text
absolute_value[n] = V(frame_0, frame_n)
absolute_advantage[n] = absolute_value[n + 50] - absolute_value[n]
relative_advantage[n] = V(frame_n, frame_n+50)
```

## Default Labeling Command

Run from the full `kai0` repo environment, not from this minimal bundle:

```bash
python stage_advantage/annotation/eval.py Flatten-Fold KAI0 <dataset>
```

This bundle is a reference/package of the relevant code and checkpoint pointer;
the full repo is still needed to execute inference because the model depends on
the broader `openpi` package and runtime environment.
