"""V1 Triton inference WebSocket serve for deepdive_kai0 (B4 Phase 2).

Wraps optimize/v1_triton/Pi05InferenceTuned as BasePolicy that
WebsocketPolicyServer can host. Protocol identical to serve_policy.py
(:8000 JAX backend), just different inference path.

Scope:
  ✓ WebSocket server on :8002, msgpack protocol identical to JAX serve
  ✓ V1 Triton inference (Pi05InferenceTuned, P50=32 ms on 5090)
  ✓ Image preprocess (resize 224×224, bfloat16, cuda)
  ✓ Action denormalize via norm_stats.json
  ✓ Action chunk return (50, action_dim)
  ✓ **Phase 2**: per-inference state encoding via kai0 sentencepiece
                 (256-bin discretize + prefix "Task: {p}, State: {s};\n")
                 + PaliGemma embedding lookup → encoder_x buffer write
                 (绕开 V1 prebaked language_embeds)
  ✓ **B1 server-side profile**: preproc_ms / infer_ms / postproc_ms /
                 state_encode_ms / total_ms in policy_timing dict

Usage (sim01, after `convert_kai0_to_v1.py` 出 .pkl):
    .venv_5090_trt/bin/python kai0/scripts/serve_policy_v1.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \\
        --norm-stats kai0/assets/<asset_id>/<repo_id>/norm_stats.json \\
        --tokenizer openpi_cache/big_vision/paligemma_tokenizer.model \\
        --port 8002 \\
        --num-views 3 --chunk-size 50 --action-dim 14 --state-dim 14

Health check: curl http://<host>:8002/healthz → "OK"

See docs/deployment/realtime_vla_optimization_analysis.md §7.2 + §6 for context.
"""
import argparse
import json
import logging
import os
import pickle
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Repo paths (lazy imports in main() to keep --help working without all deps)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_V1_TRITON_DIR = _REPO_ROOT / "optimize" / "v1_triton"
_OPENPI_SRC = _REPO_ROOT / "kai0" / "src"
_OPENPI_CLIENT_SRC = _REPO_ROOT / "kai0" / "packages" / "openpi-client" / "src"


def _ensure_imports():
    """Lazy import V1 + openpi modules (kept out of module top so --help works).

    Required deps in serving venv (kai0/.venv_5090_trt missing some by default):
      - torch, triton (have)
      - sentencepiece (have)
      - websockets (pip install websockets)
      - msgpack-numpy (pip install msgpack-numpy)
      - openpi_client (PYTHONPATH includes kai0/packages/openpi-client/src)
      - openpi (PYTHONPATH includes kai0/src)
    """
    sys.path.insert(0, str(_V1_TRITON_DIR))
    sys.path.insert(0, str(_OPENPI_SRC))
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

    global Pi05InferenceTuned, websocket_policy_server, _base_policy
    from pi05_infer_tuned import Pi05InferenceTuned as _Pi05  # noqa
    from openpi.serving import websocket_policy_server as _wsps  # noqa
    from openpi_client import base_policy as _bp  # noqa
    Pi05InferenceTuned = _Pi05
    websocket_policy_server = _wsps
    _base_policy = _bp


logger = logging.getLogger(__name__)

# Populated by _ensure_imports() in main(). Type annotations use string forms below.
Pi05InferenceTuned = None  # type: ignore[assignment]
websocket_policy_server = None  # type: ignore[assignment]
_base_policy = None  # type: ignore[assignment]


def _resize_image_to_224(img_uint8: np.ndarray) -> np.ndarray:
    """Resize HxWx3 uint8 image to 224×224 using aspect-preserving pad.

    Matches kai0 training image transform (transforms.py:255 + image_tools.py
    resize_with_pad): keep aspect ratio, pad to 224 with zeros. The legacy
    PIL.BILINEAR-direct-resize STRETCHED 640×480→224×224, distorting aspect
    and moving the input out of training distribution; this version produces
    224×168 + 28px black pad top/bottom for 640×480, matching what the vision
    encoder saw during training. Equivalent to openpi/shared/image_tools.py
    `resize_with_pad` (the torch helper has a uint8 F.interpolate bug, so we
    do it in PIL — same math, no GPU round-trip).

    Accepts both HWC (H,W,3) and CHW (3,H,W) input. kai0 ROS2 client sends
    CHW per policy_inference_node.py:1963 (imgs.transpose(2,0,1) into obs).
    """
    from PIL import Image as _PIL_Image

    arr = np.asarray(img_uint8)
    # CHW → HWC detection: (3, H, W) with H,W >> 3
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] > 3 and arr.shape[2] > 3:
        arr = np.ascontiguousarray(arr.transpose(1, 2, 0))  # CHW → HWC

    # If already 224×224×3, return as-is
    if arr.ndim == 3 and arr.shape[:2] == (224, 224) and arr.shape[2] == 3:
        return arr

    H, W = arr.shape[:2]
    # Match jax.image.resize+pad math (image_tools.py:25-43): use max ratio.
    ratio = max(W / 224.0, H / 224.0)
    rh = int(H / ratio)
    rw = int(W / ratio)
    pil = _PIL_Image.fromarray(arr).resize((rw, rh), _PIL_Image.BILINEAR)
    resized = np.asarray(pil, dtype=np.uint8)
    # Symmetric pad to 224 (pad0=floor, pad1=floor+remainder), zeros.
    pad_h0, rem_h = divmod(224 - rh, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(224 - rw, 2)
    pad_w1 = pad_w0 + rem_w
    return np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0,
    )


def _normalize_image_uint8_to_bf16(img_uint8: np.ndarray) -> torch.Tensor:
    """uint8 [0,255] HxWx3 → bf16 [-1,1] HxWx3 cuda.

    pi05 image normalization: x / 127.5 - 1.0 (the openpi/big_vision standard).
    """
    arr = torch.from_numpy(img_uint8).cuda()  # uint8
    arr = arr.to(torch.float32) / 127.5 - 1.0
    return arr.to(torch.bfloat16)


class SentencepieceStateEncoder:
    """kai0-compatible per-inference prompt+state encoding (B4 Phase 2).

    Matches kai0/src/openpi/models/tokenizer.py:64-117 (training):
      prefix = f"Task: {cleaned_prompt}, State: {state_str};\\n"
      where state_str = " ".join(map(str, np.digitize(state, 257-edge bins) - 1))
    Then sentencepiece encode (add_bos=True) → PaliGemma embedding lookup
    → scale by sqrt(2048).

    Bypasses V1's HF AutoTokenizer path (kai0 uses sentencepiece .model).
    """

    PG_SCALE = 2048 ** 0.5  # PaliGemma embed scale (sqrt of d_model)
    _BIN_EDGES = np.linspace(-1, 1, 257)[:-1]  # 256 bins in [-1, 1]

    def __init__(
        self,
        v1_infer,  # Pi05InferenceTuned
        tokenizer_model_path: str,
        embedding_weight: torch.Tensor,
        state_norm: dict[str, np.ndarray | None],
        model_state_dim: int = 32,
    ):
        """Build a kai0 prefix encoder.

        state_norm must come from load_norm_stats()['state']: a dict with keys
        {mean, std, q01, q99}, **must be model_state_dim-long** (training norm
        stats are computed on padded state). q01/q99 may be None for legacy
        z-score-only files. When q01/q99 are present (kai0 pi05 default),
        state norm uses quantile to match training (use_quantile_norm=True for
        PI05/PI05_RTC); otherwise falls back to z-score.

        model_state_dim is the action_dim of the underlying model (pi05 = 32).
        Raw state is zero-padded to this BEFORE normalize+digitize, matching
        agilex_policy.py:76 + training/config.py:152-164 (TokenizePrompt sees
        padded state with discrete_state_input=True for pi05).
        """
        import sentencepiece

        self.v1 = v1_infer
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_model_path)
        self._model_state_dim = model_state_dim

        # PaliGemma embedding table (from V1 pkl 'embedding_weight').
        # V1 doesn't keep it in self.weights (only baked language_embeds), so we
        # accept it separately and load to CUDA bf16.
        # Shape (vocab=257152, d=2048).
        if embedding_weight is None:
            raise RuntimeError(
                "embedding_weight is None. convert_kai0_to_v1.py must have stashed it "
                "in the pkl as 'embedding_weight' (does as of v0.11). Pass it explicitly."
            )
        if embedding_weight.device.type != "cuda":
            embedding_weight = embedding_weight.cuda()
        if embedding_weight.dtype != torch.bfloat16:
            embedding_weight = embedding_weight.to(torch.bfloat16)
        self._embed_w = embedding_weight  # (vocab, 2048) bf16 cuda

        # ── State norm setup ──
        # Auto-detect quantile vs z-score by presence of q01/q99 keys.
        # kai0 pi05 trains with quantile → both q01/q99 are populated.
        # Use FULL model_state_dim (32 for pi05) — training normalizes the
        # padded state, so dims 14-31 use real (zero-ish) q01/q99 entries.
        q01 = state_norm.get("q01")
        q99 = state_norm.get("q99")
        self._use_quantile = (q01 is not None) and (q99 is not None)
        if self._use_quantile:
            if len(q01) < model_state_dim:
                # Pad norm stats (rare; norm_stats usually already model_action_dim)
                q01 = np.pad(q01, (0, model_state_dim - len(q01)), constant_values=0.0)
                q99 = np.pad(q99, (0, model_state_dim - len(q99)), constant_values=0.0)
            self._s_q01 = torch.from_numpy(q01[:model_state_dim].astype(np.float32)).cuda()
            self._s_q99 = torch.from_numpy(q99[:model_state_dim].astype(np.float32)).cuda()
            # Guard zero span (1e-6 follows openpi transforms.py normalize_quantile)
            span = self._s_q99 - self._s_q01
            self._s_q99 = torch.where(span.abs() < 1e-6, self._s_q01 + 1.0, self._s_q99)
            logger.info(f"  State norm: QUANTILE (q01/q99) — matches kai0 pi05 training; pad state→{model_state_dim}")
        else:
            mean = state_norm["mean"]
            std = state_norm["std"]
            if len(mean) < model_state_dim:
                mean = np.pad(mean, (0, model_state_dim - len(mean)), constant_values=0.0)
                std = np.pad(std, (0, model_state_dim - len(std)), constant_values=1.0)
            self._s_mean = torch.from_numpy(mean[:model_state_dim].astype(np.float32)).cuda()
            self._s_std = torch.from_numpy(std[:model_state_dim].astype(np.float32)).cuda()
            self._s_std = torch.where(self._s_std < 1e-6, torch.ones_like(self._s_std), self._s_std)
            logger.info(f"  State norm: Z-SCORE (mean/std) — fallback (no q01/q99 in norm_stats); pad state→{model_state_dim}")
        self.max_prompt_len = v1_infer.max_prompt_len

    def encode(self, task_prompt: str, state_raw: np.ndarray) -> tuple[torch.Tensor, int]:
        """Encode (prompt + state) → embeds for V1 encoder.

        Args:
            task_prompt: task instruction string
            state_raw: (state_dim,) float, raw joint state (NOT normalized)

        Returns:
            (embeds, prompt_len) — embeds shape (prompt_len, 2048) bf16 cuda
        """
        # 1. Pad raw state to model_state_dim BEFORE normalize+digitize.
        # Training (agilex_policy.py:76) pads to model.action_dim (32 for pi05)
        # BEFORE Normalize → TokenizePrompt sees a 32-dim state and produces
        # 32 state tokens. Truncating to raw 14 here was a bug: prompt was
        # shorter than training, model received OOD prefix.
        s_np = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        if s_np.shape[0] < self._model_state_dim:
            s_np = np.concatenate([
                s_np,
                np.zeros(self._model_state_dim - s_np.shape[0], dtype=np.float32),
            ])
        else:
            s_np = s_np[: self._model_state_dim]
        # agilex_policy.py:104-106 — clamp out-of-range joint values to 0 (safety)
        s_np = np.where(s_np > np.pi, 0.0, s_np)
        s_np = np.where(s_np < -np.pi, 0.0, s_np)
        s = torch.from_numpy(s_np).cuda()
        if self._use_quantile:
            # (s - q01) / (q99 - q01) * 2 - 1   (matches openpi transforms.py:210)
            s_norm = (s - self._s_q01) / (self._s_q99 - self._s_q01 + 1e-6) * 2.0 - 1.0
        else:
            s_norm = (s - self._s_mean) / self._s_std
        s_norm_np = s_norm.detach().cpu().numpy()

        # 2. Discretize to 256 bins (kai0 convention). 32 state tokens for pi05.
        discretized = np.digitize(s_norm_np, bins=self._BIN_EDGES) - 1
        discretized = np.clip(discretized, 0, 255)
        state_str = " ".join(map(str, discretized.astype(int).tolist()))

        # 3. Build prefix (kai0 training format)
        cleaned = task_prompt.lower().strip().replace("_", " ")
        prefix = f"Task: {cleaned}, State: {state_str};\n"

        # 4. Sentencepiece tokenize (add_bos=True matches kai0 training line 75)
        token_ids = self.tokenizer.encode(prefix, add_bos=True)
        # Truncate to max_prompt_len (CUDA Graph buffer ceiling)
        if len(token_ids) > self.max_prompt_len:
            logger.warning(
                f"Prompt+state tokenized to {len(token_ids)} tokens > "
                f"max_prompt_len {self.max_prompt_len}; truncating."
            )
            token_ids = token_ids[: self.max_prompt_len]
        plen = len(token_ids)
        token_ids_t = torch.tensor(token_ids, dtype=torch.long, device="cuda")

        # 5. Lookup embedding + scale (matches kai0 PaliGemma forward + V1 line 53)
        embeds = self._embed_w[token_ids_t]  # (plen, 2048) bf16
        embeds = embeds * self.PG_SCALE

        return embeds, plen

    def write_to_v1_buffer(self, embeds: torch.Tensor, plen: int) -> None:
        """Write embeds into V1 encoder_x buffer + set valid_encoder_len + RoPE.

        Replicates V1.forward()'s prompt-handling lines (816-822 in
        discrete_state_input=False branch) but with our own (prompt+state)
        embeds instead of prebaked language_embeds. Must be called RIGHT
        BEFORE the CUDA Graph replay (v1.infer_graph.replay()), but the
        replay path also re-copies prebaked language_embeds into encoder_x,
        so we use forward_with_state() below to bypass replay's overwrite.
        """
        start = self.v1.num_views * 256
        self.v1.buffers["encoder_x"][start : start + plen].copy_(embeds)
        self.v1.buffers["valid_encoder_len"].fill_(start + plen)
        self.v1.buffers["decoder_rope_weights"].copy_(self.v1.get_decoder_rope_weights(plen))


def v1_forward_with_state(
    v1,  # Pi05InferenceTuned
    image: torch.Tensor,
    noise: torch.Tensor,
    embeds: torch.Tensor,
    plen: int,
) -> torch.Tensor:
    """V1 forward with externally-supplied (prompt+state) embeds.

    Replicates Pi05Inference.forward() body but skips the prebaked-
    language_embeds copy (which would overwrite our state-conditioned
    embeds). Must be called inside torch.inference_mode().
    """
    start = v1.num_views * 256
    v1.buffers["encoder_x"][start : start + plen].copy_(embeds)
    v1.buffers["valid_encoder_len"].fill_(start + plen)
    v1.buffers["decoder_rope_weights"].copy_(v1.get_decoder_rope_weights(plen))
    v1.buffers["observation_images_normalized"].copy_(image)
    v1.buffers["diffusion_noise"].copy_(noise)
    v1.infer_graph.replay()
    return v1.buffers["diffusion_noise"]


class V1Policy:
    """Adapt V1 Triton Pi05InferenceTuned to BasePolicy protocol.

    The obs dict (msgpack from ROS2 client) contains camera images + joint
    state + prompt. Phase 1 only consumes images + builds noise; state and
    prompt are NOT re-encoded per-inference (uses prebaked language_embeds
    from convert_kai0_to_v1.py). See Phase 2 TODO at module docstring.
    """

    def __init__(
        self,
        v1_infer,  # Pi05InferenceTuned
        action_norm: dict[str, np.ndarray | None],
        action_dim: int,
        state_encoder: "SentencepieceStateEncoder | None" = None,
        default_prompt: str = "Flatten and fold the cloth",
        image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        metadata: dict[str, Any] | None = None,
        delta_action_mask: np.ndarray | None = None,
    ):
        """delta_action_mask: 1-D bool array of len action_dim. If set, model output
        is treated as delta-actions; for each masked dim, server post-applies
        `chunk[t, dim] += state[dim]` to convert back to absolute joints
        (matches openpi AbsoluteActions transform). None = absolute mode (default,
        backward-compat with task_a_new_pure_200 etc)."""
        """action_norm comes from load_norm_stats()['actions'], a dict with
        {mean, std, q01, q99}. q01/q99 trigger quantile denorm (matches
        transforms.py:240-246), else fall back to z-score (a*std+mean)."""
        self._v1 = v1_infer
        self._action_dim = action_dim
        # ── Action denorm setup ──
        # Auto-detect quantile vs z-score by presence of q01/q99 keys, matching
        # how the model was trained (kai0 pi05 → quantile).
        q01 = action_norm.get("q01")
        q99 = action_norm.get("q99")
        self._use_quantile = (q01 is not None) and (q99 is not None)
        if self._use_quantile:
            self._a_q01 = torch.from_numpy(q01[:action_dim].astype(np.float32)).cuda()
            self._a_q99 = torch.from_numpy(q99[:action_dim].astype(np.float32)).cuda()
            span = self._a_q99 - self._a_q01
            self._a_q99 = torch.where(span.abs() < 1e-6, self._a_q01 + 1.0, self._a_q99)
            logger.info(f"  Action denorm: QUANTILE (q01/q99) — matches kai0 pi05 training")
        else:
            mean = action_norm["mean"]
            std = action_norm["std"]
            self._a_mean = torch.from_numpy(mean[:action_dim].astype(np.float32)).cuda()
            self._a_std = torch.from_numpy(std[:action_dim].astype(np.float32)).cuda()
            self._a_std = torch.where(self._a_std < 1e-6, torch.ones_like(self._a_std), self._a_std)
            logger.info(f"  Action denorm: Z-SCORE (mean/std) — fallback (no q01/q99)")
        # Delta action mask (None = absolute action mode; else apply AbsoluteActions
        # transform: chunk[..., masked_dims] += state[masked_dims] before returning).
        if delta_action_mask is not None:
            mask_arr = np.asarray(delta_action_mask, dtype=bool)
            if mask_arr.shape != (action_dim,):
                raise ValueError(
                    f"delta_action_mask shape {mask_arr.shape} != action_dim ({action_dim},)")
            # Pre-multiply mask onto a (action_dim,) float tensor so we can do
            # chunk += mask_state_tensor (broadcasted) in one op.
            self._delta_mask_float = torch.from_numpy(
                mask_arr.astype(np.float32)).cuda()  # (action_dim,)
            logger.info(
                f"  Delta action mode: AbsoluteActions postprocess with mask "
                f"{mask_arr.tolist()} (chunk[..., mask] += state[..., mask])")
        else:
            self._delta_mask_float = None
        self._image_keys = image_keys
        self._metadata = metadata or {"backend": "v1_triton", "version": 2}
        self._chunk_size = v1_infer.chunk_size
        self._num_views = v1_infer.num_views
        # Optional Phase 2 state encoder. If None, falls back to prebaked
        # language_embeds (Phase 1 behavior; NOT state-conditioned).
        self._state_encoder = state_encoder
        self._default_prompt = default_prompt
        # Stable noise per-call would defeat diversity; sample fresh each infer.
        self._noise_gen = torch.Generator(device="cuda")
        self._noise_gen.manual_seed(0)

    def infer(self, obs: dict) -> dict:
        """Run one V1 inference cycle.

        Args:
            obs: dict from ROS2 client. Expected keys (subset of openpi protocol):
              - images: dict[str, HxWx3 uint8 numpy] — one entry per camera
              - state: (state_dim,) float numpy (joint state, kai0 = 14)
              - prompt: str (task instruction) — Phase 2 used

        Returns:
            dict with:
              - actions: (chunk_size, action_dim) float32 numpy (denormalized)
              - policy_timing: {preproc_ms, state_encode_ms, infer_ms,
                                postproc_ms, total_ms} — B1 server-side profile
              - server_backend: "v1_triton"
        """
        t_start = time.monotonic()

        # 1. Image preprocess: pick views in fixed order, resize, normalize, stack
        images_dict = obs.get("images") or obs.get("image") or {}
        if not isinstance(images_dict, dict):
            raise ValueError(f"obs['images'] must be dict, got {type(images_dict)}")
        view_tensors = []
        for key in self._image_keys[: self._num_views]:
            if key not in images_dict:
                # Fall back: take first num_views images in insertion order
                view_tensors = [
                    _normalize_image_uint8_to_bf16(_resize_image_to_224(np.asarray(v)))
                    for v in list(images_dict.values())[: self._num_views]
                ]
                break
            view_tensors.append(
                _normalize_image_uint8_to_bf16(_resize_image_to_224(np.asarray(images_dict[key])))
            )
        if len(view_tensors) != self._num_views:
            raise ValueError(
                f"Need {self._num_views} views, got {len(view_tensors)} (keys: {list(images_dict.keys())})"
            )
        image_input = torch.stack(view_tensors, dim=0).contiguous()  # (num_views, 224, 224, 3) bf16 cuda
        torch.cuda.synchronize()
        preproc_ms = (time.monotonic() - t_start) * 1000

        # 2. Sample fresh noise (chunk_size, 32) bf16 cuda
        noise = torch.randn(
            self._chunk_size, 32,
            dtype=torch.bfloat16, device="cuda",
            generator=self._noise_gen,
        )

        # 3. State encoding (Phase 2 if state_encoder available; else Phase 1 fallback)
        t_state = time.monotonic()
        state_embeds = None
        plen = 0
        prompt_used = obs.get("prompt", self._default_prompt) or self._default_prompt
        if self._state_encoder is not None:
            state_raw = obs.get("state")
            if state_raw is None:
                raise ValueError(
                    "obs['state'] required when state_encoder is configured (Phase 2). "
                    "Disable state encoder to use prebaked-prompt Phase 1 fallback."
                )
            state_embeds, plen = self._state_encoder.encode(prompt_used, state_raw)
        torch.cuda.synchronize()
        state_encode_ms = (time.monotonic() - t_state) * 1000

        # 4. V1 inference
        t_infer = time.monotonic()
        with torch.inference_mode():
            if self._state_encoder is not None:
                # Phase 2: bypass V1's prebaked language_embeds, write state-conditioned embeds
                action_chunk = v1_forward_with_state(
                    self._v1, image_input, noise, state_embeds, plen,
                )
            else:
                # Phase 1 fallback: V1 forward uses prebaked language_embeds
                action_chunk = self._v1.forward(image_input, noise)
        torch.cuda.synchronize()
        infer_ms = (time.monotonic() - t_infer) * 1000

        # 5. Take first action_dim, denormalize, to numpy.
        # Quantile path (kai0 pi05 default): (a+1)/2 * (q99-q01) + q01  matches
        # openpi transforms.py:246; z-score fallback: a*std + mean.
        t_post = time.monotonic()
        a = action_chunk[:, : self._action_dim].to(torch.float32)  # (chunk_size, action_dim)
        if self._use_quantile:
            a = (a + 1.0) / 2.0 * (self._a_q99[None, :] - self._a_q01[None, :] + 1e-6) + self._a_q01[None, :]
        else:
            a = a * self._a_std[None, :] + self._a_mean[None, :]
        # Delta → Absolute (config-driven; equivalent to openpi AbsoluteActions output transform).
        # When delta_action_mask is set, model output is interpreted as joint deltas
        # relative to current state. We add state[..., masked_dims] back so the client
        # sees absolute joint targets — no client change required (action_kind stays "joint").
        if self._delta_mask_float is not None:
            # state_raw was extracted in step 3 if state_encoder; otherwise need it now.
            state_for_delta = obs.get("state")
            if state_for_delta is None:
                raise ValueError(
                    "delta_action_mask is set but obs['state'] missing — delta mode requires state.")
            s_t = torch.from_numpy(
                np.asarray(state_for_delta, dtype=np.float32).reshape(-1)[: self._action_dim]
            ).cuda()  # (action_dim,)
            # a (chunk_size, action_dim) += (action_dim,) * (action_dim,)  broadcast
            a = a + (s_t * self._delta_mask_float)[None, :]
        actions_np = a.detach().cpu().numpy()  # (chunk_size, action_dim) float32
        postproc_ms = (time.monotonic() - t_post) * 1000

        total_ms = (time.monotonic() - t_start) * 1000

        return {
            "actions": actions_np,
            # B1 server-side profile (each step's wall-clock, sum ≈ total_ms)
            "policy_timing": {
                "preproc_ms": float(preproc_ms),
                "state_encode_ms": float(state_encode_ms),
                "infer_ms": float(infer_ms),  # V1 forward only
                "postproc_ms": float(postproc_ms),
                "total_ms": float(total_ms),
            },
            "server_backend": "v1_triton",
            "phase": 2 if self._state_encoder is not None else 1,
            "action_kind": "joint",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def load_v1_inference(pkl_path: str, num_views: int, chunk_size: int):
    """Load V1 pkl + build Pi05InferenceTuned.

    Returns (infer, embedding_weight). embedding_weight is extracted from
    pkl 'embedding_weight' field (full PaliGemma table 257152×2048, baked by
    convert_kai0_to_v1.py for Phase 2 re-tokenization); None if absent.
    """
    logger.info(f"Loading V1 ckpt from {pkl_path} ...")
    t0 = time.perf_counter()
    with open(pkl_path, "rb") as f:
        ckpt = pickle.load(f)
    logger.info(
        f"  loaded {sum(v.numel()*v.element_size() for v in ckpt.values())/1e9:.2f} GB tensors "
        f"in {time.perf_counter()-t0:.1f}s"
    )
    embedding_weight = ckpt.get("embedding_weight")  # (vocab, 2048) bf16 cpu; for Phase 2
    if embedding_weight is None:
        logger.warning("pkl missing 'embedding_weight'; Phase 2 state encoding unavailable")

    logger.info(
        f"Building Pi05InferenceTuned(num_views={num_views}, chunk_size={chunk_size}) "
        f"+ CUDA Graph capture ..."
    )
    t0 = time.perf_counter()
    infer = Pi05InferenceTuned(
        ckpt, num_views=num_views, chunk_size=chunk_size,
        discrete_state_input=False,  # Phase 2 overrides via v1_forward_with_state
    )
    logger.info(f"  build + capture in {time.perf_counter()-t0:.1f}s")
    return infer, embedding_weight


def load_norm_stats(norm_stats_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load state + action norm stats from openpi-format norm_stats.json.

    Loads BOTH z-score (mean/std) AND quantile (q01/q99) when present so caller
    can pick the schema matching training. kai0 pi05 training uses quantile
    (per src/openpi/training/config.py:352 — use_quantile_norm True for PI05/
    PI05_RTC), so pi05 ckpts need quantile path; pi0 ckpts need z-score.

    Format (per kai0/assets/<asset>/<repo>/norm_stats.json):
      {"norm_stats": {"state":   {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
                      "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}}}

    Returns: {"state":   {"mean": np, "std": np, "q01": np|None, "q99": np|None},
              "actions": {...}}
    Downstream auto-detects quantile via presence of q01/q99.
    """
    with open(norm_stats_path) as f:
        data = json.load(f)
    norm = data["norm_stats"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in ("state", "actions"):
        if key not in norm:
            if key == "actions":
                raise ValueError(f"norm_stats.json missing 'actions' (have: {list(norm.keys())})")
            continue  # state is optional (Phase 1 doesn't need it)
        entry = norm[key]
        if "mean" not in entry or "std" not in entry:
            raise ValueError(f"norm_stats['{key}'] needs mean+std (have: {list(entry.keys())})")
        out[key] = {
            "mean": np.asarray(entry["mean"], dtype=np.float32),
            "std": np.asarray(entry["std"], dtype=np.float32),
            "q01": np.asarray(entry["q01"], dtype=np.float32) if "q01" in entry else None,
            "q99": np.asarray(entry["q99"], dtype=np.float32) if "q99" in entry else None,
        }
    return out


def warmup(policy: V1Policy, n: int = 3, state_dim: int = 14) -> None:
    """Warm-up to ensure CUDA Graph + caches are hot before serving."""
    logger.info(f"Warming up V1 inference ({n} dummy iters) ...")
    H, W = 480, 640  # arbitrary; resize to 224 happens inside
    dummy_obs = {
        "images": {
            f"view_{i}": np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            for i in range(policy._num_views)
        },
        "state": np.zeros(state_dim, dtype=np.float32),
        "prompt": "warmup test",
    }
    for i in range(n):
        out = policy.infer(dummy_obs)
        pt = out["policy_timing"]
        logger.info(
            f"  warmup {i+1}/{n}: total={pt['total_ms']:.1f}ms "
            f"(preproc={pt['preproc_ms']:.1f} + state={pt['state_encode_ms']:.1f} "
            f"+ infer={pt['infer_ms']:.1f} + post={pt['postproc_ms']:.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 Triton WebSocket serve for deepdive_kai0")
    parser.add_argument("--pkl", required=True, help="V1 pickle from convert_kai0_to_v1.py")
    parser.add_argument("--norm-stats", required=True,
                        help="kai0 norm_stats.json (for state+action normalize)")
    parser.add_argument("--tokenizer", default=None,
                        help="sentencepiece .model path (e.g. openpi_cache/big_vision/"
                             "paligemma_tokenizer.model). REQUIRED for Phase 2 "
                             "state encoding; omit for Phase 1 prebaked-prompt mode.")
    parser.add_argument("--default-prompt", default="Flatten and fold the cloth",
                        help="Used when obs lacks 'prompt' field (matches V1 pkl bake prompt)")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14,
                        help="output action dim to take from V1's (chunk, 32) output")
    parser.add_argument("--state-dim", type=int, default=14,
                        help="state dim used for sentencepiece discretization (Phase 2)")
    parser.add_argument("--image-keys", nargs="+",
                        default=["top_head", "hand_left", "hand_right"],
                        help="Camera keys in obs['images'] dict, in stack order. "
                             "Default matches kai0 ROS2 client (policy_inference_node._get_observation). "
                             "Order positions IS the V1 model channel order — must match training "
                             "(agilex_policy: top_head→view0, hand_left→view1, hand_right→view2). "
                             "Wrong order = left/right wrist cameras swapped, model gets mirrored world. "
                             "Use ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'] only for openpi "
                             "official agilex client format.")
    parser.add_argument("--warmup-iters", type=int, default=3)
    # C.4 2026-05-23 (§7.8): SHM transport. 默认 ws (旧行为). shm 启用 POSIX shm
    # 替 msgpack+TCP loopback, P95 cycle 估 -5-7ms. both = 并行跑 (用 client 选).
    parser.add_argument("--transport", choices=("ws", "shm", "both"), default="ws",
        help="Transport: ws (default, backward compat) | shm (POSIX shm, low-latency) | both")
    parser.add_argument("--shm-req-name", default="kai0_v1_obs",
        help="POSIX shm name for client→server request region (when transport in {shm, both})")
    parser.add_argument("--shm-resp-name", default="kai0_v1_chunk",
        help="POSIX shm name for server→client response region")
    # Delta-mode 2026-05-23: if model trained with delta_joint_actions=True,
    # server applies AbsoluteActions transform (chunk[..., :14] += state for
    # masked dims) before returning. Default mask matches kai0 Task_A delta:
    # left_joint(6)+left_grip(abs)+right_joint(6)+right_grip(abs) = [T*6,F,T*6,F].
    parser.add_argument("--delta-joint-actions", action="store_true",
        help="Treat model output as delta-joint. Server post-applies AbsoluteActions "
             "(chunk[..., joint_dims] += state[joint_dims]). Required for ckpts "
             "trained with use_delta_joint_actions=True (e.g. task_a_base_delta).")
    parser.add_argument("--delta-action-mask-csv",
        default="1,1,1,1,1,1,0,1,1,1,1,1,1,0",
        help="Delta mask (length=action_dim, 1=joint delta, 0=absolute e.g. gripper). "
             "Default matches kai0 dual-Piper: 6 joints + 1 gripper × 2 arms.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    # Lazy import heavy deps now that --help has succeeded.
    _ensure_imports()

    if not torch.cuda.is_available():
        logger.error("CUDA not available — V1 requires a CUDA-capable GPU.")
        sys.exit(1)
    logger.info(f"Device: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}0)")
    logger.info(f"torch {torch.__version__}, cuda {torch.version.cuda}")

    # 1. Load V1 inference + extract embedding_weight (for Phase 2)
    v1_infer, embedding_weight = load_v1_inference(args.pkl, args.num_views, args.chunk_size)

    # 2. Load state + action norm stats (mean/std + q01/q99 if present)
    norm = load_norm_stats(args.norm_stats)
    a_stats = norm["actions"]
    a_dim_n = len(a_stats["mean"])
    logger.info(f"Action norm: dim={a_dim_n} (taking first {args.action_dim})")

    # 3. Build sentencepiece state encoder (Phase 2) if tokenizer provided
    state_encoder = None
    if args.tokenizer:
        if "state" not in norm:
            raise ValueError(
                "Phase 2 needs norm_stats['state'] for state normalization, "
                "but norm_stats.json has no 'state' entry."
            )
        if embedding_weight is None:
            raise ValueError(
                "Phase 2 needs 'embedding_weight' in V1 pkl; re-run convert_kai0_to_v1.py "
                "(or use expand_v1_pkl_for_phase2.py if pkl already has embedding_weight "
                "but small language_embeds)."
            )
        s_stats = norm["state"]
        # Pass FULL norm stats (typically already model_action_dim=32 long).
        # Encoder pads raw state to model_state_dim internally before normalize+
        # digitize. Don't slice to args.state_dim — that's only the RAW input dim.
        logger.info(f"Loading sentencepiece tokenizer: {args.tokenizer}")
        state_encoder = SentencepieceStateEncoder(
            v1_infer,
            tokenizer_model_path=args.tokenizer,
            embedding_weight=embedding_weight,
            state_norm=s_stats,
            model_state_dim=a_dim_n,  # = action_dim from norm_stats (32 for pi05)
        )
        logger.info(
            f"  Phase 2 state encoding enabled: state_dim={args.state_dim}, "
            f"max_prompt_len={state_encoder.max_prompt_len}, "
            f"default_prompt={args.default_prompt!r}"
        )
    else:
        logger.warning(
            "No --tokenizer; running Phase 1 (prebaked language_embeds, "
            "NOT state-conditioned). Inference will not react to changing state."
        )

    # 4. Build V1Policy
    phase = 2 if state_encoder is not None else 1
    # Parse delta-action-mask (default: kai0 dual-piper [T*6,F,T*6,F]; only used
    # when --delta-joint-actions is set).
    delta_mask = None
    if args.delta_joint_actions:
        try:
            delta_mask = np.asarray(
                [bool(int(x)) for x in args.delta_action_mask_csv.split(",")],
                dtype=bool,
            )
        except Exception as e:
            raise SystemExit(f"invalid --delta-action-mask-csv: {e}")
        if delta_mask.shape != (args.action_dim,):
            raise SystemExit(
                f"--delta-action-mask-csv length {len(delta_mask)} != action_dim {args.action_dim}")
    policy = V1Policy(
        v1_infer,
        action_norm=a_stats,
        action_dim=args.action_dim,
        state_encoder=state_encoder,
        default_prompt=args.default_prompt,
        image_keys=tuple(args.image_keys),
        delta_action_mask=delta_mask,
        metadata={
            "backend": "v1_triton",
            "version": 2,
            "ckpt_pkl": str(Path(args.pkl).resolve()),
            "norm_stats": str(Path(args.norm_stats).resolve()),
            "tokenizer": str(Path(args.tokenizer).resolve()) if args.tokenizer else None,
            "num_views": args.num_views,
            "chunk_size": args.chunk_size,
            "action_dim": args.action_dim,
            "state_dim": args.state_dim,
            "default_prompt": args.default_prompt,
            "phase": phase,
            "delta_joint_actions": bool(args.delta_joint_actions),
        },
    )

    # 5. Warm-up
    warmup(policy, n=args.warmup_iters, state_dim=args.state_dim)

    # 5. Start server(s). transport=ws (default) / shm / both.
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "?"

    shm_server = None
    if args.transport in ("shm", "both"):
        # POSIX shm transport (P95 cycle -5-7ms 估 vs WS+msgpack TCP loopback)
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from shm_transport import ShmServer
        shm_server = ShmServer(
            infer_callback=policy.infer,
            req_name=args.shm_req_name,
            resp_name=args.shm_resp_name,
            logger=logger,
        )
        shm_server.start()
        logger.info(f"SHM transport: /dev/shm/{args.shm_req_name} (req) + /dev/shm/{args.shm_resp_name} (resp)")

    if args.transport in ("ws", "both"):
        logger.info(f"Serving V1 Triton policy on {args.host}:{args.port} (hostname: {hostname}, ip: {local_ip})")
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if shm_server is not None:
                shm_server.stop()
    else:
        # SHM-only mode: keep main thread alive while ShmServer's daemon thread polls.
        logger.info("SHM-only mode (--transport shm). Ctrl-C to stop.")
        try:
            while True:
                import time as _time
                _time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            if shm_server is not None:
                shm_server.stop()


if __name__ == "__main__":
    main()
