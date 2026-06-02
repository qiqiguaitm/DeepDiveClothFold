# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GigaWorld-Policy is an action-centered World-Action Model (WAM) for robot policy learning. It is initialized from the **Wan2.2-TI2V-5B** video-diffusion backbone and jointly learns 2D pixel (video) dynamics and robot actions, with action decoding being the primary output and video generation optional.

## Setup

```bash
conda create -n gigaworld-policy python==3.11
conda activate gigaworld-policy
# Install the three vendored frameworks (order matters — train, then models, then datasets):
pip install ./third_party/giga-train
pip install ./third_party/giga-models
pip install ./third_party/giga-datasets
```

There is no test suite, linter config, or build step in this repo — work is run via the `scripts/` entry points below.

## Common Commands

All entry points are run as modules from the repo root.

```bash
# 1. Compute action normalization stats -> norm_stats_delta.json (required by the policy)
python -m scripts.compute_norm_stats --data_paths <dataset_dir> --output_path <out.json> \
  --embodiment_id <id> --delta-mask <mask> --sample-rate 1.0 --action-chunk 48

# 2. Pre-compute T5 text embeddings for language instructions
python -m scripts.compute_t5_embedding --repo_id <dataset_dir> --root <dataset_dir> \
  --wan_path <Wan2.2-TI2V-5B> --device cuda --text_len 512 --t5_folder_name t5_embedding

# 3. Train (config is a Python module path resolving to a `config` dict)
python -m scripts.train --config world_action_model.configs.example.config

# 4. Inference: start server, then run the open-loop client
python -m scripts.inference_server --model_id <hf_model_dir> --transformer_path <ckpt_dir> \
  --stats_path <norm_stats_delta.json> --t5_embedding_pkl <t5_embedding.pt>   # add --return_images for video viz
python -m scripts.inference_client --dataset_paths <dataset_dir> --save_dir ./vis
```

Before training, edit `world_action_model/configs/example.py`: set `models.pretrained`, `transform.norm_path` (the generated stats), and `data_dir`.

## Architecture

### Three-layer structure
- **`third_party/giga-train`** — generic training framework (`giga_train` package) built on HuggingFace Accelerate. Provides the `Trainer` base class, config loading (`load_config`), `launch_from_config`, and registries (`TRANSFORMS`, `SAMPLERS`, `OPTIMIZERS`, `SCHEDULERS`). `scripts/train.py` just calls `launch_from_config(config)`.
- **`third_party/giga-datasets`** — the `giga_datasets` package: `LeRobotDataset`, samplers, evaluators, and data structures.
- **`third_party/giga-models`** — the `giga_models` package, notably `RobotInferenceServer` (the inference socket server used by `scripts/inference_server.py`).
- **`world_action_model/`** — the actual model, trainers, transforms, and inference pipeline that sit on top of those frameworks.

Treat `third_party/*` as vendored dependencies; most policy work happens in `world_action_model/` and `scripts/`.

### Config-driven training
Training is fully config-driven. A config is a plain Python module exposing a `config` dict (see `world_action_model/configs/example.py`). Key fields: `runners` (dotted path to the trainer class, e.g. `world_action_model.trainer.wa_casual_trainer.CasualWATrainer`), `dataloaders` (with a `transform` referenced by registry `type` like `WATransformsLerobot`), `models`, `optimizers`, `schedulers`, `train`, and `launch` (GPU ids, `distributed_type`, DeepSpeed config — default `accelerate_configs/zero2.json`). The trainer/transform/sampler are resolved by string name through giga_train's registries, so a new trainer must be importable and the path put in `runners`.

### The core model: `CasualWorldActionTransformer`
`world_action_model/models/transformer_wa_casual.py` subclasses the Wan diffusion transformer and adds an **`action_encoder`** (MLP: action_dim→...→inner_dim) and **`action_decoder`** (inner_dim→...→action_dim), plus a 1D RoPE (`WanRotaryPosEmbed1D`) for the action/state tokens. The central idea is **token reordering**: state, ref-video (spatial), action (T tokens), and noisy-video tokens are concatenated into one sequence, and a **causal self-attention mask** enforces that state/ref and action tokens cannot attend to future noisy-video tokens. There are three forward paths selected at runtime:
- `_forward_train` — joint flow-matching loss on video latents and actions.
- `_forward_inference` — denoises video **and** decodes actions.
- `_forward_inference_action_only` — skips noisy-video tokens entirely for fast action-only decoding (the `action_only=True` path used at serve time).

### Training loop: `CasualWATrainer`
`world_action_model/trainer/wa_casual_trainer.py` (subclass of `giga_train.Trainer`). `forward_step` encodes images through the (frozen) `AutoencoderKLWan`, applies flow-matching noise to both video latents and actions with a shared `sigma`/timestep, builds the first-frame conditioning mask, and returns `{'visual_loss', 'action_loss'}`. Note it **re-creates** `action_encoder`/`action_decoder`/`action_rope` in `get_models` (with `action_dim` from config) and copies them onto the loaded transformer — so the action head dimension is set by the trainer/config, not by the checkpoint. `flow_shift` (default 5.0) and `expand_timesteps` control the noise schedule and the Wan2.2-TI2V timestep-per-token expansion.

### Inference pipeline: `WAPipeline`
`world_action_model/pipeline/wa_pipeline.py` (a diffusers `DiffusionPipeline`). `__call__` runs the flow-matching denoising loop with **two separate schedulers** (one for video latents, one for actions) that must stay in lockstep. It supports `action_only` to skip video decoding. `scripts/inference_server.py` wraps this pipeline: it loads pre-computed T5 embeddings and norm stats, monkey-patches an `inference(observation)` method onto the pipeline, and serves it via `RobotInferenceServer`. Action post-processing — denormalize, then `add_state_to_action` using a `--delta_mask` distinguishing delta vs. absolute action dims — happens in the server, mirroring the normalization done in the transform.

### Data conventions
Datasets are LeRobot-format. The default setup uses 3 camera views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), an action chunk of 48, and a robot-type→embodiment-id mapping (`robotype_to_embed_id`) so multiple embodiments share the model. `WATransformsLerobot` (`world_action_model/transformers/wa_transforms_lerobot.py`) handles per-embodiment normalization via the `norm_stats_delta.json` produced by `compute_norm_stats`.
