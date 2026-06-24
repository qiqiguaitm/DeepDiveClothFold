# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **official inference/deployment release of τ₀-World Model (τ₀-WM)** — *A Unified Video-Action World Model for Robotic Manipulation* (also referred to as **VAM**, the Video-Action Model). The repo serves a single trained policy over a WebSocket: given multi-view camera frames + dual-arm end-effector state + a language prompt, it returns an absolute-pose action chunk for two robot arms.

**This repo is inference-only.** Training code, the Simulator weights, and the Test-Time Computation code are *not* in this repo (per the README, the latter two "will be released soon"). Do not assume a training loop, dataset loader, or eval harness exists here — there is none.

This directory is its own git repository (`github.com/sii-research/tau-0-wm`), nested inside the larger `deepdive_kai0` workspace. The parent `../CLAUDE.md` describes the unrelated **kai0/openpi** project — its commands, configs, and architecture do **not** apply here. Treat this repo in isolation.

## Setup & Running

```bash
pip install -r requirements.txt          # torch 2.8+cu128, diffusers 0.35.2, transformers 5.0.0rc1, websockets 16

# Start the policy server (host + port are positional args)
bash run_infer_server.sh <HOST> <PORT>
#   → python -m web_infer_utils.server --config configs/deployment/wan_pretrain_rela_eef6d.yaml --host <HOST> --port <PORT>

# Smoke-test with a client that sends one random observation
python web_infer_utils.simple_client.py   # or: python -m web_infer_utils.simple_client
```

There is **no lint config, no test suite, and no build step.** The only `*_test.py` files (`web_infer_utils/openpi_client/*_test.py`) are vendored from openpi and test the msgpack/image utilities, not this project.

### Required weights (must be downloaded and wired into the YAML before serving)

Edit `configs/deployment/wan_pretrain_rela_eef6d.yaml` and replace every `/path/to/...` placeholder:
- `diffusion_model.model_path` → τ₀-WM checkpoint ([HF: sii-research/tau-0-wm](https://huggingface.co/sii-research/tau-0-wm))
- `vae_path` → `Wan2.2_VAE.pth` from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- `text_encoder.checkpoint_path` / `text_encoder.tokenizer_path` → UMT5-xxl encoder + tokenizer from the same Wan2.2 release

The server can be launched under `torchrun` (it reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK`); without those env vars it runs single-GPU on `cuda:0`.

## Architecture

The model is a **dual-branch diffusion transformer** built on Wan 2.2 TI2V-5B. One branch denoises *video* latents (the world model); a parallel branch denoises *actions* conditioned on the video branch's intermediate features. This is what makes it a "video-action world model": action prediction is grounded in a learned video prediction backbone.

### Request flow (server → policy → model)

1. **`web_infer_utils/server.py`** — `TauPolicyServer` subclasses `TauPolicy` and adds the openpi-style WebSocket loop. Each frame is `msgpack_numpy`-decoded into an `obs` dict, passed to `self.play(**obs)`, and the resulting action chunk is packed back. A `<reset>` substring anywhere in `prompt` triggers `self.reset()` (clears the cached text/world context) before being stripped out. Health check on `GET /healthz`. CLI flags expose many inference-perf knobs (`--compile`, `--attention-impl`, fused-QKV, KV/rope caches) — see `get_args()`.

2. **`web_infer_utils/TauPolicy.py`** — the core policy. `__init__` reads the YAML into an `argparse.Namespace`, dynamically loads the VAE / text-encoder / diffusion-model / pipeline classes via `utils.import_custom_class` (so swapping implementations is config-only), and loads `statistics.json` for mean/std normalization. **`play()`** is the entry point: it normalizes state, builds the EEF-6D state vector, runs the diffusion pipeline, then un-normalizes and converts relative predictions back to absolute poses. Holds `self.context` across calls so the world-model context can persist between steps unless `reset_context=True`.

3. **`models/wan_2_2_models/pipeline/textimage2video.py`** — `WanTI2V`, the diffusion sampler. Two methods:
   - **`infer(...)`** — action-only path used in deployment (`joint_denoising=False`). It encodes the observed frames to VAE latents once, runs the diffusion loop, and on the **first step** stores the video branch's per-layer activations into a `video_states_buffer` (and optionally an `action_context_kv_cache`); subsequent action-denoising steps reuse that buffer instead of recomputing the video branch. This is the key latency optimization — the expensive video backbone runs ~once, the cheap action branch runs every step.
   - **`infer_cotrain(...)`** — joint video+action path (`joint_denoising=True`). Fully samples a predicted video, saves it to `video_save_folder`, then denoises actions against it. Slower; used for visualization/debugging the world model.
   - Schedulers (`unipc`/`dpm++`/`euler`) live in `models/wan_2_2_models/scheduler/`. Text context (and the empty negative prompt) are cached in `_text_context_cache`.

4. **`models/wan_2_2_models/transformers/model.py`** — `WanModel`, the DiT. `self.blocks` is the video stream; `self.action_blocks` (gated by `action_dim`) is the action stream fed by `action_proj_in`. `forward()` is multiplexed by `return_video` / `return_action` / `store_buffer` flags and consumes `video_states_buffer` + `history_action_state`. `attention.py` centralizes the attention backend (flash-attn 2/3 vs SDPA) via the module-global `set_attention_backend()`.

### Action space (critical — wire-format contract)

Defined in README and implemented in `TauPolicy.play()` + `utils/action_space_utils.py`. The network operates internally on **relative EEF-6D** (config: `action_type: relative`, `action_space: eef6d`), but the **wire protocol uses absolute quaternion poses**:

- **State sent to server**: absolute poses of both EEFs = 14 channels `[L_xyz(3) + L_quat_xyzw(4), R_xyz(3) + R_quat_xyzw(4)]`, each in its own Arm-Base-link frame. Plus `gripper_states` (2 channels, 0=open … 120=close).
- Internally `play()` converts quaternion→6D (`quaternion_to_rotation_6d`) and interleaves grippers, producing the **20-dim** vector the model expects: per arm `xyz(3) + rot6d(6) + gripper(1)`, ×2. This 20-dim layout matches `statistics.json` (`action`/`state` mean/std, length 20) and the YAML `action_dim: 20`, `gripper_dim: 1`.
- **Action returned**: absolute EEF poses, shape `{T, 16}` = `[L_xyz(3)+L_quat(4)+L_gripper(1), R_xyz(3)+R_quat(4)+R_gripper(1)]`, gripper openness 0→1. Relative→absolute reconstruction uses `rela_eef_to_abs(action, state)`, which integrates the predicted body-frame deltas onto the current reference pose.

`action_space_utils.py` contains the full rotation toolkit (quaternion ↔ matrix ↔ 6D ↔ euler, and the forward/inverse `rela_eef_to_abs` / `abs_eef_to_rela` for both 18-dim rot6d and 12-dim rpy layouts). When touching action math, keep `rela_eef_to_abs` and `abs_eef_to_rela` as exact inverses.

### `play()` inference knobs (passed per-request in the obs dict)

`num_inference_steps` (diffusion steps, ~5), `execution_step` (how many of the predicted chunk frames to actually return, 1–100), `shift` (noise-schedule shift), `sample_solver` (`unipc`/`dpm++`/`euler`), `joint_denoising` (action-only vs joint video+action). Chunk geometry comes from the YAML: `chunk: 9` observed/predicted video frames, `action_chunk: 33` predicted action steps, `img_size: [192, 256]`.

### `obs` payload shape

Camera input (`obs["obs"]`) is either a float tensor `{V, 3, H, W}` in `[-1, 1]`, or a `uint8` numpy array `{V, H, W, 3}` in `[0, 255]` (auto-normalized in `play()`). The three views are head + two wrist cameras (`VIEW_KEYS` in `server.py`). See `web_infer_utils/simple_client.py` for a concrete example payload.

## Vendored code

- **`web_infer_utils/openpi_client/`** — copied verbatim from [openpi](https://github.com/Physical-Intelligence/openpi); provides the WebSocket client, `msgpack_numpy` (de)serialization, the action-chunk broker, and the runtime/agent scaffolding. The server's wire protocol matches this client, so any openpi WebSocket client can talk to it.
- **`models/wan_2_2_models/`** — adapted from [Wan2.2](https://github.com/Wan-Video/Wan2.2); the action branch and dual-stream `forward` are the τ₀-WM additions on top. Headers retain Alibaba Wan Team copyright. Some code is also adapted from [GE-Act / Genie-Envisioner](https://github.com/AgibotTech/Genie-Envisioner).

When editing these directories, prefer minimal, surgical changes that preserve compatibility with the upstream protocol/architecture.

## Conventions

- **Config drives class loading.** `utils.import_custom_class(class_name, source)` resolves `source` as either an installed module path or an absolute/`.py` file path. The deployment YAML names classes (`WanModel`, `WanTI2V`, `Wan2_2_VAE`, `T5EncoderModel`) and their `*_class_path` files; to swap an implementation, point the YAML at a different file rather than editing import statements.
- **Single normalization scheme.** Only `norm_type: meanstd` is implemented; anything else raises `NotImplementedError`. Action/state stats are 20-dim and must stay aligned with the model's channel layout.
- License: Apache 2.0.
