"""
pi0.5 inference backend 5-way benchmark.

Compares the inference latency of `PI0Pytorch.sample_actions` under 5 backends:
    A. eager           — un-compiled, baseline
    B. compile-default — torch.compile(mode="default"); Inductor fusion, no CUDA Graph
    C. cuda-graph      — manual torch.cuda.CUDAGraph (eager kernels)
    D. reduce-overhead — torch.compile(mode="reduce-overhead"); Inductor fusion + auto CUDA Graph
    E. max-autotune    — torch.compile(mode="max-autotune"); maximal Inductor autotune + auto CUDA Graph
                         (this is deepdive_kai0's current default in pi0_pytorch.py:113)

Inputs:
- Random weights (architecture-only speed test; weights values don't affect timing)
- Dummy observation: 3 cameras × 224×224×3 RGB (bf16), 14-dim state padded to 32, prompt 'Flatten and fold the cloth' tokenized to 64

Outputs:
- Terminal table
- Markdown report at results/pi05_inference_<hostname>_<date>.md

Usage:
    cd /home/tim/workspace/deepdive_kai0
    .venv/bin/python optimize/benchmark_pi05_inference.py
    # or
    .venv/bin/python optimize/benchmark_pi05_inference.py --backends A,B,C,D,E --n-test 100

See docs/deployment/pi05_inference_backend_benchmark_plan.md for full design notes.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


# Make sure we can import openpi from kai0/src
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent  # deepdive_kai0/
_KAI0_SRC = _REPO_ROOT / "kai0" / "src"
sys.path.insert(0, str(_KAI0_SRC))
sys.path.insert(0, str(_HERE))  # for cuda_graph_wrapper

from cuda_graph_wrapper import CUDAGraphedSampleActions  # noqa: E402
from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing  # noqa: E402

# Dynamo cannot trace `class SimpleProcessedObservation:` defined inside
# preprocess_observation_pytorch (Python's __build_class__ is a C builtin).
# Wrap preprocess in dynamo.disable so it runs eagerly during compile.
_preprocessing.preprocess_observation_pytorch = torch._dynamo.disable(
    _preprocessing.preprocess_observation_pytorch
)


def _setup_logging():
    """Force-reconfigure root logger; openpi/jax/absl imports also touch logging
    and may clobber our handlers if we only call basicConfig once at module top."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


_setup_logging()
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ACTION_HORIZON = 50
ACTION_DIM = 32          # Pi0Config default (state/action padded to 32)
NUM_STEPS = 10           # denoising steps (Pi0RTC default; deepdive_kai0 prod also 10)
NUM_CAMERAS = 3          # top + left wrist + right wrist (sim01 setup)
IMAGE_RESOLUTION = (224, 224)
PROMPT = "Flatten and fold the cloth"
MAX_TOKEN_LEN = 48       # Pi0 default, see Pi0Config.max_token_len typical value


# ─────────────────────────────────────────────────────────────────────────────
# Model / observation construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> PI0Pytorch:
    """Build a fresh PI0Pytorch with pi05=True and random weights in bf16.

    Important: PI0Pytorch is normally a "mixed dtype" model designed for
    autocast training (PaliGemma internal bf16 + outer nn.Linear fp32 +
    sample_noise/dt fp32). Direct inference without autocast crashes on
    dtype mismatches.

    We monkey-patch the 3 hard-coded fp32 spots inside the model:
      - sample_noise -> uses model.dtype
      - sample_actions dt -> uses model.dtype
      - sample_actions time -> uses model.dtype
    Then cast the entire model (including outer Linear) to bf16. This makes
    bf16 inference self-consistent.

    This patch is what deepdive_kai0 will need to apply to pi0_pytorch.py
    before running PyTorch inference in production (see analysis doc §3.4.2);
    here we apply it only in this benchmark to keep the change local.
    """
    config = Pi0Config(
        pi05=True,
        discrete_state_input=True,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        max_token_len=MAX_TOKEN_LEN,
        dtype="bfloat16",
    )
    log.info(f"Building PI0Pytorch (pi05=True, action_horizon={ACTION_HORIZON}, action_dim={ACTION_DIM}, dtype=bf16, model-patched)")
    model = PI0Pytorch(config).to(device).to(torch.bfloat16).eval()
    _restore_eager_sample_actions_real(model)  # strip __init__ torch.compile
    _patch_pi05_for_bf16(model)
    return model


def _restore_eager_sample_actions_real(model: PI0Pytorch) -> None:
    """Actually strip the __init__-installed torch.compile wrapper from sample_actions."""
    if "sample_actions" in vars(model):
        del model.sample_actions


def _patch_pi05_for_bf16(model: PI0Pytorch) -> None:
    """Patch the 3 hard-coded fp32 spots inside model so bf16 inference works."""
    import types
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    compute_dtype = next(model.parameters()).dtype

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=compute_dtype, device=device)

    model.sample_noise = types.MethodType(sample_noise, model)

    # Patch embed_prefix: original (pi0_pytorch.py:229) uses
    #   `torch.tensor(att_masks_list, dtype=torch.bool, device=pad_masks.device)`
    # which is a host list -> CUDA tensor copy and is REJECTED by torch.cuda.CUDAGraph.capture
    # (errors with "Cannot copy between CPU and CUDA tensors during CUDA graph capture
    # unless the CPU tensor is pinned").
    # Since att_masks accumulates `[0] * num_img_embs` and `[0] * num_lang_embs` (all zeros,
    # meaning full attention within prefix), we can build it directly on-device with
    # torch.zeros, no host->device copy needed.
    import math as _math

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = []
        pad_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(_img):
                return self.paligemma_with_expert.embed_image(_img)
            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

        def lang_embed_func(_lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(_lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * _math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # CUDA-Graph-friendly: build att_masks on-device (semantically all zeros = full attention within prefix)
        att_masks = torch.zeros(pad_masks.shape[1], dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])
        return embs, pad_masks, att_masks

    model.embed_prefix = types.MethodType(embed_prefix, model)

    # Patch embed_suffix: line 311 same `torch.tensor(att_masks_list, ...)` host->device problem.
    # pi05 path produces pattern [1, 0, 0, ..., 0] of length action_horizon
    # (first action token starts new attention block, rest stay inside).
    import torch.nn.functional as _F
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    def embed_suffix(self, state, noisy_actions, timestep):
        embs = []
        pad_masks = []
        # Build attention mask pattern on-device. For pi05: [1, 0, ..., 0] length action_horizon.
        # For non-pi05: prepended with [1] for state. We handle both.
        att_mask_pattern = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self._apply_checkpoint(lambda s: self.state_proj(s), state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_mask_pattern.append(1)  # state opens new block

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0, device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(lambda a: self.action_in_proj(a), noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = _F.silu(x)
                return self.action_time_mlp_out(x)
            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = _F.silu(x)
                x = self.time_mlp_out(x)
                return _F.silu(x)
            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Action attention pattern: first action token opens new block, rest stay in.
        att_mask_pattern += [1] + [0] * (self.config.action_horizon - 1)
        total_suffix_len = len(att_mask_pattern)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # CUDA-Graph-safe attention mask construction:
        # Build pattern on-device WITHOUT any host->device copy or scalar indexing.
        # We use arange + comparison: positions where pattern[i]==1 are encoded as
        # index in a small set; for the known patterns we use arithmetic.
        arange = torch.arange(total_suffix_len, device=embs.device)
        if self.pi05:
            # pattern = [1, 0, 0, ..., 0]  →  arange == 0
            att_masks_1d = (arange == 0)
        else:
            # pattern = [1] (state) + [1, 0, 0, ..., 0] (actions) = [1, 1, 0, 0, ..., 0]
            #          →  arange < 2
            att_masks_1d = (arange < 2)
        att_masks_1d = att_masks_1d.to(embs.dtype)
        att_masks = att_masks_1d[None, :].expand(bsize, total_suffix_len)
        return embs, pad_masks, att_masks, adarms_cond

    model.embed_suffix = types.MethodType(embed_suffix, model)

    # Patch _prepare_attention_masks_4d: returns fp32 mask due to torch.where
    # literal 0.0/-inf; Inductor SDPA requires bias.dtype == query.dtype (bf16).
    # NOTE: torch.tensor(scalar, device=cuda) is a host->device copy (REJECTED by CUDA Graph
    # capture). Use Tensor.new_full / .to() which are device-side ops.
    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # .new_full allocates on att_2d_masks_4d.device (no host copy)
        zero = att_2d_masks_4d.new_full((), 0.0, dtype=compute_dtype)
        # Use finfo(dtype).min so fp16 (max=65504) and bf16 (max=3.39e38) both work
        neg_inf = att_2d_masks_4d.new_full((), torch.finfo(compute_dtype).min, dtype=compute_dtype)
        return torch.where(att_2d_masks_4d, zero, neg_inf)

    model._prepare_attention_masks_4d = types.MethodType(_prepare_attention_masks_4d, model)

    # Patch denoise_step: PaliGemma's RMSNorm hardcodes fp32 output, so
    # outputs_embeds returns fp32 even when input is bf16. We cast back before
    # action_out_proj (which is bf16) to avoid dtype mismatch.
    orig_denoise = type(model).denoise_step

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        # Cast x_t to bf16 to match embed_suffix inputs
        x_t_bf = x_t.to(compute_dtype) if x_t.dtype != compute_dtype else x_t
        # Call original up to action_out_proj manually with cast
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t_bf, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon:]
        # Cast back to bf16 before action_out_proj (bf16 weight)
        suffix_out = suffix_out.to(compute_dtype)
        return self.action_out_proj(suffix_out)

    model.denoise_step = types.MethodType(denoise_step, model)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(compute_dtype)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Use new_full (device-side, CUDA-Graph-safe) instead of torch.tensor (host->device copy)
        dt = noise.new_full((), -1.0 / num_steps, dtype=compute_dtype)
        x_t = noise
        time = noise.new_full((), 1.0, dtype=compute_dtype)
        # CUDA-Graph-safe: fixed-iter for-loop instead of `while time >= -dt/2`
        # (while + GPU tensor condition requires host sync, breaks capture).
        # num_steps is a Python int known at trace time, so the loop unrolls to N denoise calls.
        for _ in range(num_steps):
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    model.sample_actions = types.MethodType(sample_actions, model)


def restore_eager_sample_actions(model: PI0Pytorch) -> None:
    """No-op now that _patch_pi05_for_bf16 always installs the bf16-aware version.

    After build_model, `model.sample_actions` is already an instance method that
    is bf16-aware and NOT torch.compile-wrapped (the patch overwrites the
    `__init__`-installed torch.compile wrapper). This function is kept for
    compatibility with the per-backend pattern below but does nothing.
    """
    pass


def make_dummy_observation(device: torch.device, dtype: torch.dtype = torch.float32, batch: int = 1):
    """Build a dummy SimpleProcessedObservation-like object.

    Shapes match what PI0Pytorch.sample_actions expects:
    - images: dict of {cam_name: (B, H, W, C) bf16, values in [-1, 1]}
    - image_masks: dict of {cam_name: (B,) bool}
    - tokenized_prompt: (B, max_token_len) int64
    - tokenized_prompt_mask: (B, max_token_len) bool
    - state: (B, action_dim) fp32 (padded; real has 14 dim)
    """
    class _Obs:
        pass

    obs = _Obs()
    cam_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    obs.images = {}
    obs.image_masks = {}
    for cam in cam_names:
        # CHW format (B, 3, H, W) — preprocessing_pytorch detects via shape[1]==3
        # Values in [-1, 1] to mimic normalized image input
        img = (torch.rand(batch, 3, *IMAGE_RESOLUTION, device=device, dtype=dtype) * 2 - 1)
        obs.images[cam] = img
        obs.image_masks[cam] = torch.ones(batch, dtype=torch.bool, device=device)

    # Random prompt tokens — actual values don't affect timing, only shape matters
    # Use values < 256000 to be safe within PaliGemma vocab
    obs.tokenized_prompt = torch.randint(0, 10000, (batch, MAX_TOKEN_LEN), dtype=torch.long, device=device)
    obs.tokenized_prompt_mask = torch.ones(batch, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    # Causal attention mask + loss mask (required by preprocess_observation_pytorch)
    obs.token_ar_mask = torch.ones(batch, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    obs.token_loss_mask = torch.ones(batch, MAX_TOKEN_LEN, dtype=torch.bool, device=device)

    # State: padded to action_dim (32). Real Piper double-arm uses 14 dim.
    obs.state = torch.zeros(batch, ACTION_DIM, dtype=torch.float32, device=device)
    obs.state[:, :14] = torch.randn(batch, 14, dtype=torch.float32, device=device) * 0.5

    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def time_call(fn, n_warmup: int, n_test: int) -> dict:
    """Time `fn()` with warm-up and per-call sync.

    Returns dict of statistics in ms.
    """
    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # First-call time (after warm-up — should be steady-state, but record anyway)
    times = []
    for _ in range(n_test):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "n": n_test,
        "mean": float(times.mean()),
        "std": float(times.std()),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "min": float(times.min()),
        "max": float(times.max()),
    }


def measure_first_call(fn) -> float:
    """Measure the first call (likely includes compile / capture overhead)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def bench_eager(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """A. PyTorch eager (no compile, no CUDA Graph)."""
    log.info("=== Backend A: eager ===")
    restore_eager_sample_actions(model)

    @torch.inference_mode()
    def run():
        return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)

    first_ms = measure_first_call(run)
    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = "A. eager"
    return stats


def bench_compile_default(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """B. torch.compile(mode='default') — Inductor fusion, no CUDA Graph."""
    log.info("=== Backend B: compile-default ===")
    restore_eager_sample_actions(model)
    # Wrap sample_actions
    model.sample_actions = torch.compile(model.sample_actions, mode="default", fullgraph=False)

    @torch.inference_mode()
    def run():
        return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)

    first_ms = measure_first_call(run)
    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = "B. compile-default"
    return stats


def bench_cuda_graph(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """C. Manual CUDA Graph (eager kernels, no Inductor fusion)."""
    log.info("=== Backend C: cuda-graph (manual capture) ===")
    restore_eager_sample_actions(model)

    runner = CUDAGraphedSampleActions.capture(
        model=model,
        sample_obs=obs,
        num_steps=NUM_STEPS,
        n_warmup=3,
    )

    def run():
        # noise stays static (random init); observation stays static
        return runner()

    first_ms = measure_first_call(run)
    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = "C. cuda-graph"
    return stats


def bench_compile_reduce_overhead(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """D. torch.compile(mode='reduce-overhead') — Inductor + auto CUDA Graph."""
    log.info("=== Backend D: compile-reduce-overhead ===")
    restore_eager_sample_actions(model)

    fullgraph_flag = True
    try:
        model.sample_actions = torch.compile(model.sample_actions, mode="reduce-overhead", fullgraph=True)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)

        first_ms = measure_first_call(run)
    except Exception as e:
        log.warning(f"fullgraph=True failed: {e}; falling back to fullgraph=False")
        fullgraph_flag = False
        restore_eager_sample_actions(model)
        model.sample_actions = torch.compile(model.sample_actions, mode="reduce-overhead", fullgraph=False)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)

        first_ms = measure_first_call(run)

    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = f"D. compile-reduce-overhead (fullgraph={fullgraph_flag})"
    return stats


def bench_compile_max_autotune(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """E. torch.compile(mode='max-autotune') — deepdive_kai0's current default."""
    log.info("=== Backend E: compile-max-autotune (deepdive_kai0 default) ===")
    restore_eager_sample_actions(model)

    fullgraph_flag = True
    try:
        model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=True)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)

        first_ms = measure_first_call(run)
    except Exception as e:
        log.warning(f"fullgraph=True failed: {e}; falling back to fullgraph=False")
        fullgraph_flag = False
        restore_eager_sample_actions(model)
        model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=False)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)

        first_ms = measure_first_call(run)

    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = f"E. compile-max-autotune (fullgraph={fullgraph_flag})"
    return stats


def bench_compile_F_coordinate_descent(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """F. max-autotune + coordinate_descent_tuning (Inductor deeper kernel search).

    Sweep around max-autotune's best kernel by coordinate descent on
    BLOCK_M/N/K/num_stages/num_warps. Compile time ~2-3x slower than E.
    """
    log.info("=== Backend F: compile-max-autotune + coordinate_descent_tuning ===")
    import torch._inductor.config as _ind_cfg

    # Save/restore
    saved_cd = _ind_cfg.coordinate_descent_tuning
    saved_cd_check = getattr(_ind_cfg, "coordinate_descent_check_all_directions", False)
    _ind_cfg.coordinate_descent_tuning = True

    try:
        restore_eager_sample_actions(model)
        fullgraph_flag = True
        try:
            model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=True)

            @torch.inference_mode()
            def run():
                return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
            first_ms = measure_first_call(run)
        except Exception as e:
            log.warning(f"F fullgraph=True failed: {e}; fallback fullgraph=False")
            fullgraph_flag = False
            restore_eager_sample_actions(model)
            _patch_pi05_for_bf16(model)
            model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=False)

            @torch.inference_mode()
            def run():
                return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
            first_ms = measure_first_call(run)

        stats = time_call(run, n_warmup, n_test)
        stats["first_call_ms"] = first_ms
        stats["name"] = f"F. max-autotune + coordinate_descent (fullgraph={fullgraph_flag})"
        return stats
    finally:
        _ind_cfg.coordinate_descent_tuning = saved_cd


def bench_compile_J_cutlass_backend(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """J. max-autotune with CUTLASS GEMM backend added to autotune candidates.

    Inductor default uses Triton templates; adding CUTLASS lets it sweep
    NVIDIA's hand-tuned GEMM kernels alongside.
    """
    log.info("=== Backend J: compile-max-autotune + CUTLASS GEMM backend ===")
    import torch._inductor.config as _ind_cfg

    saved = _ind_cfg.max_autotune_gemm_backends
    _ind_cfg.max_autotune_gemm_backends = "TRITON,CUTLASS"

    try:
        restore_eager_sample_actions(model)
        fullgraph_flag = True
        try:
            model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=True)

            @torch.inference_mode()
            def run():
                return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
            first_ms = measure_first_call(run)
        except Exception as e:
            log.warning(f"J fullgraph=True failed: {e}; fallback fullgraph=False")
            fullgraph_flag = False
            restore_eager_sample_actions(model)
            _patch_pi05_for_bf16(model)
            model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=False)

            @torch.inference_mode()
            def run():
                return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
            first_ms = measure_first_call(run)

        stats = time_call(run, n_warmup, n_test)
        stats["first_call_ms"] = first_ms
        stats["name"] = f"J. max-autotune + CUTLASS (fullgraph={fullgraph_flag})"
        return stats
    finally:
        _ind_cfg.max_autotune_gemm_backends = saved


def fuse_qkv_inplace(model: torch.nn.Module) -> int:
    """V1 §4.2.2: Fuse separate q_proj/k_proj/v_proj into one qkv_proj per attention layer.

    Inductor's max-autotune does NOT auto-fuse 3 separate matmuls (sharing input)
    into a single big matmul, because they're separate nn.Linear modules with
    independent weight tensors. Manual concat of weights + replacing forward
    gives Inductor one big matmul, reducing kernel launches + improving GEMM
    occupancy.

    This function:
      1. Finds every nn.Module that has all of q_proj/k_proj/v_proj as direct nn.Linear children
      2. Creates qkv_proj = nn.Linear(in, q_out + k_out + v_out) with concatenated weights
      3. Monkey-patches the parent's forward (Gemma and SigLIP styles) to use qkv_proj
    Returns number of layers fused.
    """
    import torch.nn as nn
    import types

    fused_count = 0

    def _find_attn_modules(root):
        for name, m in root.named_modules():
            if (hasattr(m, 'q_proj') and isinstance(getattr(m, 'q_proj'), nn.Linear)
                and hasattr(m, 'k_proj') and isinstance(getattr(m, 'k_proj'), nn.Linear)
                and hasattr(m, 'v_proj') and isinstance(getattr(m, 'v_proj'), nn.Linear)
                and not hasattr(m, 'qkv_proj')):  # don't double-fuse
                yield name, m

    # Build qkv_proj for each attention
    for name, attn in _find_attn_modules(model):
        q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
        in_features = q.in_features
        q_out, k_out, v_out = q.out_features, k.out_features, v.out_features
        has_bias = q.bias is not None

        qkv = nn.Linear(in_features, q_out + k_out + v_out, bias=has_bias,
                        device=q.weight.device, dtype=q.weight.dtype)
        with torch.no_grad():
            # nn.Linear weight shape is (out, in)
            qkv.weight.copy_(torch.cat([q.weight, k.weight, v.weight], dim=0))
            if has_bias:
                qkv.bias.copy_(torch.cat([q.bias, k.bias, v.bias], dim=0))
        attn.qkv_proj = qkv
        attn._qkv_splits = (q_out, k_out, v_out)
        fused_count += 1

    # Patch forward — different signatures for Gemma vs SigLIP
    for name, attn in _find_attn_modules(model):
        # detect by output shape: SigLIP has q_out == k_out == v_out, Gemma may have GQA
        # We always do generic split; signature differs by class name
        cls_name = type(attn).__name__
        q_out, k_out, v_out = attn._qkv_splits

        if "Siglip" in cls_name:
            # SigLIP attention.forward signature (transformers 4.53):
            #   forward(self, hidden_states, attention_mask=None, output_attentions=False, ...)
            # Internal: queries = self.q_proj(hidden_states); keys=...; values=...
            # We replace the 3 calls with one matmul + split.
            # Since we can't easily replace just the 3 lines, monkey-patch entire forward.
            # Original forward (from siglip modeling): copy + replace q_proj/k_proj/v_proj calls.
            #
            # Strategy: leave q_proj/k_proj/v_proj in place; replace forward to use qkv_proj.
            from transformers.models.siglip.modeling_siglip import SiglipAttention

            def siglip_fused_forward(self, hidden_states, attention_mask=None, **kwargs):
                bsz, q_len, _ = hidden_states.size()
                qkv = self.qkv_proj(hidden_states)
                q_dim, k_dim, v_dim = self._qkv_splits
                queries, keys, values = qkv.split([q_dim, k_dim, v_dim], dim=-1)
                queries = queries.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                keys = keys.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                values = values.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                attn_weights = torch.matmul(queries, keys.transpose(2, 3)) * self.scale
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
                attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, values)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)
                attn_output = self.out_proj(attn_output)
                return attn_output, attn_weights if kwargs.get("output_attentions") else None

            attn.forward = types.MethodType(siglip_fused_forward, attn)

        elif "Gemma" in cls_name or "Attention" in cls_name:
            # Gemma attention.forward (transformers 4.53)
            from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS

            def gemma_fused_forward(self, hidden_states, position_embeddings, attention_mask=None,
                                    past_key_value=None, cache_position=None, use_cache=False, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                qkv = self.qkv_proj(hidden_states)
                q_dim, k_dim, v_dim = self._qkv_splits
                q, k, v = qkv.split([q_dim, k_dim, v_dim], dim=-1)
                query_states = q.view(hidden_shape).transpose(1, 2)
                key_states = k.view(hidden_shape).transpose(1, 2)
                value_states = v.view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    if use_cache:
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                    else:
                        key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)

                attention_interface = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling, **kwargs,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            attn.forward = types.MethodType(gemma_fused_forward, attn)

    return fused_count


def bench_K_qkv_fusion(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """K. max-autotune + manual QKV fusion (V1 §4.2.2 paper).

    Concatenate q_proj/k_proj/v_proj weights into single qkv_proj per attention
    layer. Inductor sees one big matmul instead of 3 small ones, reducing
    kernel count and improving GEMM occupancy.
    """
    log.info("=== Backend K: compile-max-autotune + QKV fusion (V1 §4.2.2) ===")
    restore_eager_sample_actions(model)

    fused = fuse_qkv_inplace(model)
    log.info(f"Fused QKV in {fused} attention layers")

    fullgraph_flag = False  # Gemma forward has control flow, fullgraph fails
    try:
        model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=False)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
        first_ms = measure_first_call(run)
    except Exception as e:
        log.exception(f"K FAILED: {e}")
        return {
            "name": f"K. QKV fusion (FAILED: {type(e).__name__})",
            "mean": float("inf"), "std": 0, "p50": 0, "p95": 0, "p99": 0,
            "min": 0, "max": 0, "first_call_ms": 0,
        }

    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = f"K. QKV fusion + max-autotune ({fused} layers fused)"
    return stats


def bench_H_aoti(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """H. AOTInductor — export model to .pt2 package, load and run.

    Bypasses Dynamo runtime trace entirely; the compiled artifact is a
    standalone .so + .pt2 that runs without Python interpreter overhead
    beyond a single call into the loaded callable.

    Strategy: wrap PI0Pytorch.sample_actions in a tensor-input nn.Module,
    export it, AOTI-compile it.
    """
    log.info("=== Backend H: AOTInductor (export + AOT compile) ===")
    import torch.nn as nn
    from pathlib import Path

    # 1. Build a tensor-only-IO wrapper module
    num_steps = NUM_STEPS

    class _Pi05InferenceWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            # Don't register base_model as submodule (its parameters are already
            # part of state_dict). Use _modules / attribute access.
            self._base = base_model
            self.num_steps = num_steps

        def forward(
            self,
            image_base: torch.Tensor,
            image_left: torch.Tensor,
            image_right: torch.Tensor,
            tokenized_prompt: torch.Tensor,
            tokenized_prompt_mask: torch.Tensor,
            token_ar_mask: torch.Tensor,
            token_loss_mask: torch.Tensor,
            state: torch.Tensor,
            noise: torch.Tensor,
        ) -> torch.Tensor:
            class _Obs:
                pass
            o = _Obs()
            o.images = {
                "base_0_rgb": image_base,
                "left_wrist_0_rgb": image_left,
                "right_wrist_0_rgb": image_right,
            }
            o.image_masks = {
                "base_0_rgb": torch.ones(image_base.shape[0], dtype=torch.bool, device=image_base.device),
                "left_wrist_0_rgb": torch.ones(image_left.shape[0], dtype=torch.bool, device=image_left.device),
                "right_wrist_0_rgb": torch.ones(image_right.shape[0], dtype=torch.bool, device=image_right.device),
            }
            o.tokenized_prompt = tokenized_prompt
            o.tokenized_prompt_mask = tokenized_prompt_mask
            o.token_ar_mask = token_ar_mask
            o.token_loss_mask = token_loss_mask
            o.state = state
            return self._base.sample_actions(image_base.device, o, noise=noise, num_steps=self.num_steps)

    wrapper = _Pi05InferenceWrapper(model).to(device).eval()

    # 2. Extract example inputs from observation
    example_args = (
        obs.images["base_0_rgb"],
        obs.images["left_wrist_0_rgb"],
        obs.images["right_wrist_0_rgb"],
        obs.tokenized_prompt,
        obs.tokenized_prompt_mask,
        obs.token_ar_mask,
        obs.token_loss_mask,
        obs.state,
        torch.zeros(1, model.config.action_horizon, model.config.action_dim,
                    dtype=next(model.parameters()).dtype, device=device),
    )

    # 3. Export
    log.info("Exporting model via torch.export...")
    try:
        with torch.inference_mode():
            exported = torch.export.export(wrapper, example_args, strict=False)
    except Exception as e:
        log.exception(f"torch.export FAILED: {e}")
        return {
            "name": f"H. aoti (export FAILED: {type(e).__name__})",
            "mean": float("inf"), "std": 0, "p50": 0, "p95": 0, "p99": 0,
            "min": 0, "max": 0, "first_call_ms": 0,
        }

    # 4. AOTI compile
    log.info("AOTI compiling to .pt2 ...")
    package_path = str(_HERE / "results" / "pi05_aoti.pt2")
    Path(package_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
    except Exception as e:
        log.exception(f"AOTI compile FAILED: {e}")
        return {
            "name": f"H. aoti (compile FAILED: {type(e).__name__})",
            "mean": float("inf"), "std": 0, "p50": 0, "p95": 0, "p99": 0,
            "min": 0, "max": 0, "first_call_ms": 0,
        }

    # 5. Load + run
    log.info(f"Loading AOTI package {package_path}")
    compiled = torch._inductor.aoti_load_package(package_path)

    def run():
        return compiled(*example_args)

    first_ms = measure_first_call(run)
    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = "H. aoti (export + AOT compile)"
    return stats


def bench_compile_G_fp16(model, obs, device, n_warmup: int, n_test: int) -> dict:
    """G. max-autotune but model + patches re-cast to fp16 instead of bf16.

    On Blackwell sm_120, fp16 Tensor Core path is often faster than bf16
    because cuBLAS/cuDNN have more years of fp16 tuning. Mantissa precision
    is higher than bf16, range is smaller (max ~65504) — for inference of
    PaliGemma-class models this is generally safe.

    IMPORTANT: this re-builds the model in fp16. The caller passes in a model
    that was built in bf16; we replace its parameters and re-apply patches.
    """
    log.info("=== Backend G: compile-max-autotune + fp16 (instead of bf16) ===")

    # Re-cast model and re-apply patches with fp16 compute_dtype
    model.to(torch.float16)
    _restore_eager_sample_actions_real(model)  # strip torch.compile from __init__
    _patch_pi05_for_bf16(model)  # patches detect dtype from model.parameters(); now fp16

    fullgraph_flag = True
    try:
        model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=True)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
        first_ms = measure_first_call(run)
    except Exception as e:
        log.warning(f"G fullgraph=True failed: {e}; fallback fullgraph=False")
        fullgraph_flag = False
        _restore_eager_sample_actions_real(model)
        _patch_pi05_for_bf16(model)
        model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune", fullgraph=False)

        @torch.inference_mode()
        def run():
            return model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
        first_ms = measure_first_call(run)

    stats = time_call(run, n_warmup, n_test)
    stats["first_call_ms"] = first_ms
    stats["name"] = f"G. max-autotune + fp16 (fullgraph={fullgraph_flag})"
    return stats


BACKEND_DISPATCH = {
    "A": bench_eager,
    "B": bench_compile_default,
    "C": bench_cuda_graph,
    "D": bench_compile_reduce_overhead,
    "E": bench_compile_max_autotune,
    "F": bench_compile_F_coordinate_descent,
    "G": bench_compile_G_fp16,
    "H": bench_H_aoti,
    "J": bench_compile_J_cutlass_backend,
    "K": bench_K_qkv_fusion,
}


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict], baseline_mean: float, hardware: str, torch_version: str) -> str:
    """Format results as a markdown-friendly table string."""
    lines = []
    lines.append(f"### Hardware: {hardware}")
    lines.append(f"### PyTorch: {torch_version}, Date: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("| Backend | Mean | Std | P50 | P95 | P99 | Min | Speedup | First-call (incl compile/capture) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r is None:
            continue
        speedup = baseline_mean / r["mean"] if r["mean"] > 0 else 0
        lines.append(
            f"| {r['name']} | {r['mean']:.1f} | {r['std']:.1f} | {r['p50']:.1f} | "
            f"{r['p95']:.1f} | {r['p99']:.1f} | {r['min']:.1f} | {speedup:.2f}× | "
            f"{r['first_call_ms']:.0f} ms |"
        )
    lines.append("")
    lines.append("All times in ms unless noted. Speedup relative to backend A (eager).")
    return "\n".join(lines)


def save_markdown(results: list[dict], baseline_mean: float, hardware: str, torch_version: str) -> Path:
    """Save results as a markdown report under results/."""
    hostname = platform.node()
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _HERE / "results" / f"pi05_inference_{hostname}_{date}.md"

    body = print_results(results, baseline_mean, hardware, torch_version)
    full_md = [
        f"# pi0.5 inference backend benchmark — {hostname}",
        "",
        f"_Generated by `optimize/benchmark_pi05_inference.py` at {datetime.datetime.now().isoformat()}_",
        "",
        body,
        "",
        "## Notes",
        "",
        f"- num_steps (denoising): {NUM_STEPS}",
        f"- action_horizon: {ACTION_HORIZON}",
        f"- action_dim: {ACTION_DIM}",
        f"- num_cameras: {NUM_CAMERAS}",
        f"- image_resolution: {IMAGE_RESOLUTION[0]}×{IMAGE_RESOLUTION[1]}",
        f"- batch_size: 1",
        "- weights: random (architecture-only speed test)",
        "",
        "Configuration matrix (backend × optimization stack):",
        "",
        "| Backend | Inductor fusion | CUDA Graph | autotune |",
        "|:---:|:---:|:---:|---|",
        "| A. eager | ❌ | ❌ | — |",
        "| B. compile-default | ✅ | ❌ | min |",
        "| C. cuda-graph (manual) | ❌ | ✅ | — |",
        "| D. compile-reduce-overhead | ✅ | ✅ (auto) | std |",
        "| E. compile-max-autotune | ✅ | ✅ (auto) | **max** (deepdive_kai0 default) |",
    ]
    out.write_text("\n".join(full_md))
    log.info(f"Results written to: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global NUM_STEPS
    _setup_logging()  # re-apply force=True after openpi/jax/absl imports touch logging
    parser = argparse.ArgumentParser(description="pi0.5 inference backend 5-way benchmark")
    parser.add_argument("--backends", default="A,B,C,D,E,F,G,J",
                        help="Comma-separated list of backends to run. Default: A,B,C,D,E,F,G,J. "
                             "F=coord_descent_tuning, G=fp16, J=CUTLASS GEMM.")
    parser.add_argument("--n-warmup", type=int, default=10,
                        help="Warm-up iterations per backend (excluded from timing). Default: 10")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Timed iterations per backend. Default: 100")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Denoising steps per inference. Default: 10")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build model + run single inference per backend, skip benchmark loop")
    args = parser.parse_args()
    NUM_STEPS = args.num_steps

    backends = [b.strip() for b in args.backends.split(",")]
    invalid = [b for b in backends if b not in BACKEND_DISPATCH]
    if invalid:
        raise ValueError(f"Unknown backends: {invalid}. Valid: {list(BACKEND_DISPATCH.keys())}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this benchmark requires a GPU")

    device = torch.device("cuda")
    dtype = torch.float32  # weight dtype; bf16 applied via autocast inside each backend
    hardware = torch.cuda.get_device_name(device)
    torch_version = f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}"
    log.info(f"Hardware: {hardware}")
    log.info(f"{torch_version}")

    log.info("Building model + dummy observation (this allocates ~3-4 GB GPU mem)...")
    obs = make_dummy_observation(device, dtype, batch=1)
    log.info("Dummy observation built.")

    if args.dry_run:
        log.info("=== Dry run mode: testing each backend with 1 inference ===")
        for b in backends:
            log.info(f"--- Backend {b} ---")
            model = build_model(device)
            fn = BACKEND_DISPATCH[b]
            try:
                result = fn(model, obs, device, n_warmup=1, n_test=2)
                log.info(f"OK: mean={result['mean']:.1f} ms")
            except Exception as e:
                log.exception(f"FAILED: {e}")
            del model
            torch.cuda.empty_cache()
            torch._dynamo.reset()  # clear Inductor cache between backends
            torch.cuda.empty_cache()
        return

    results = []
    baseline_mean = None

    for b in backends:
        log.info(f"\n{'=' * 60}\n=== Running backend {b} ===\n{'=' * 60}")
        # Build a fresh model for each backend so prior compile/graph state doesn't leak
        model = build_model(device)
        fn = BACKEND_DISPATCH[b]
        try:
            result = fn(model, obs, device, args.n_warmup, args.n_test)
            results.append(result)
            if b == "A":
                baseline_mean = result["mean"]
            log.info(
                f"  mean={result['mean']:.1f}ms p50={result['p50']:.1f}ms "
                f"p95={result['p95']:.1f}ms p99={result['p99']:.1f}ms "
                f"first-call={result['first_call_ms']:.0f}ms"
            )
        except Exception as e:
            log.exception(f"Backend {b} FAILED: {e}")
            results.append({"name": f"{b}. FAILED ({e!s})", "mean": float("inf"),
                            "std": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0,
                            "first_call_ms": 0})
        del model
        torch.cuda.empty_cache()
        torch._dynamo.reset()  # critical: clear Inductor compile cache between backends
        torch.cuda.empty_cache()

    if baseline_mean is None and results:
        baseline_mean = results[0]["mean"]

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(print_results(results, baseline_mean, hardware, torch_version))

    save_markdown(results, baseline_mean, hardware, torch_version)


if __name__ == "__main__":
    main()
