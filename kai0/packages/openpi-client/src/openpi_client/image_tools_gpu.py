"""GPU-accelerated equivalent of openpi_client.image_tools.resize_with_pad.

P2 Level 1 (2026-05-23, §7.8 in realtime_vla_optimization_analysis.md).
Replaces the 3× CPU PIL resize chain in policy_inference_node._get_observation
with a single batched torch GPU call. Numerical parity within ~1 pixel value
(antialiased bilinear matches PIL.Image.BILINEAR + antialias).

Opt-in: only used when policy_inference_node param `gpu_preprocess=true`. JAX
legacy path keeps CPU resize_with_pad (bit-identical, no torch dep).

Design:
- Batch the 3 RGB images as a single tensor [3, H, W, 3] → permute → [3, 3, H, W]
- `F.interpolate(mode='bilinear', align_corners=False, antialias=True)` matches
  PIL.Image.BILINEAR + LANCZOS-ish anti-aliasing semantics (PyTorch ≥ 1.11).
- `F.pad(left, right, top, bottom)` with PIL's asymmetric pad offsets
  (`(target - resized) // 2` on each lower side, remainder on upper side).

GPU choice: pass `device='cuda:0'` (same as V1 serve). Server's CUDA Graphs sit
in their own process context; preprocess kernel runs serially with WS round-trip
so there's no SM contention. Memory cost ~30MB (3× 480x640x3 float32 + intermediates).
"""

from __future__ import annotations

import numpy as np


_TORCH_OK = False
_F = None
_torch = None
_warmup_done = False


def _lazy_import():
    """Import torch lazily to keep openpi_client thin when GPU preprocess is off."""
    global _TORCH_OK, _F, _torch
    if _TORCH_OK:
        return
    import torch
    import torch.nn.functional as F
    _torch = torch
    _F = F
    _TORCH_OK = True


def resize_with_pad_torch_batch(
    imgs: np.ndarray,
    height: int,
    width: int,
    device: str = "cuda:0",
) -> np.ndarray:
    """GPU batched resize_with_pad (PIL-equivalent).

    Args:
        imgs: np.ndarray uint8 of shape [B, H, W, 3] (already stacked).
            All images must share the same input H, W (batched op).
        height, width: target size (e.g. 224, 224).
        device: torch device string (e.g. "cuda:0").

    Returns:
        np.ndarray uint8 [B, height, width, 3] — same layout as
        openpi_client.image_tools.resize_with_pad output.

    Numerical parity vs PIL: bilinear + antialias=True matches PIL.Image.BILINEAR
    semantics within ±1 pixel value typically (no functional difference for the
    downstream vision encoder which is tolerant to ±1/255 jitter).
    """
    _lazy_import()
    torch = _torch
    F = _F
    if imgs.dtype != np.uint8:
        raise TypeError(f"imgs.dtype must be uint8, got {imgs.dtype}")
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"imgs.shape must be [B, H, W, 3], got {imgs.shape}")
    B, H, W, C = imgs.shape
    if (H, W) == (height, width):
        return imgs.copy()  # no-op (mirror CPU path semantics)

    # PIL semantics: ratio = max(W/target_w, H/target_h); rh = H/ratio, rw = W/ratio.
    ratio = max(W / width, H / height)
    rh = int(H / ratio)
    rw = int(W / ratio)

    # H2D: uint8 → float32 on GPU (avoid CPU float convert; saves bandwidth).
    t = torch.from_numpy(imgs).to(device=device, non_blocking=True)  # [B, H, W, 3] uint8
    t = t.permute(0, 3, 1, 2).contiguous().to(torch.float32)  # [B, 3, H, W]

    # Antialiased bilinear (PIL.BILINEAR equivalent for downscale).
    resized = F.interpolate(t, size=(rh, rw), mode="bilinear",
                            align_corners=False, antialias=True)  # [B, 3, rh, rw]

    # Pad to (height, width). PIL behavior: paste at (pad_w, pad_h), remaining = 0.
    # F.pad(input, (left, right, top, bottom), value=0).
    pad_left = max(0, (width - rw) // 2)
    pad_right = width - rw - pad_left
    pad_top = max(0, (height - rh) // 2)
    pad_bottom = height - rh - pad_top
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    # → [B, 3, height, width]

    # D2H + uint8 conversion. .round() to match PIL's nearest-int output.
    out = padded.clamp(0.0, 255.0).round().to(torch.uint8)  # [B, 3, h, w]
    out = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [B, h, w, 3]
    return out


def warmup(device: str, target_h: int = 224, target_w: int = 224,
           sample_h: int = 480, sample_w: int = 640):
    """Run one dummy call to compile/cache CUDA kernels + alloc workspace.

    Should be called once at node startup to avoid first-cycle latency spike.
    """
    global _warmup_done
    _lazy_import()
    dummy = np.zeros((3, sample_h, sample_w, 3), dtype=np.uint8)
    _ = resize_with_pad_torch_batch(dummy, target_h, target_w, device=device)
    _torch.cuda.synchronize()
    _warmup_done = True
