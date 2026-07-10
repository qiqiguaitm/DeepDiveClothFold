"""Encoder registry — the single source of truth for every frozen visual encoder.

Each entry is an `EncoderSpec`. Adding a new encoder = adding one row here; nothing
downstream changes. This is what makes CRAVE encoder-agnostic.

Key invariants the rest of the code relies on:
  - `res` is chosen so the patch grid is exactly `grid`×`grid` (decoder is fixed 16→128).
      dinov2 (patch14) @224 → 16×16 ;  dinov3 (patch16) @256 → 16×16.
  - `nprefix` tokens are skipped before patch pooling/reshape:
      dinov2 = 1 (CLS only) ;  dinov3 = 5 (CLS + 4 register tokens).
  - dinov3 MUST run in bf16 — fp16 overflows to NaN in ViT-H+/7B.
"""
from __future__ import annotations

from dataclasses import dataclass

from crave.config.paths import HF_HUB, REPO


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    kind: str          # "hf_dino" | "wan_vae"
    path: str
    dim: int           # pooled / per-token feature dimension
    dtype: str = "fp16"  # "fp16" | "bf16" | "int8" | "fp32"
    res: int = 224     # processor target resolution
    grid: int = 16     # patch-grid side (after skipping prefix)
    nprefix: int = 1   # tokens to skip (CLS + registers)


ENCODERS: dict[str, EncoderSpec] = {
    # ---- DINOv2 (patch14) ----
    "dinov2-small": EncoderSpec("dinov2-small", "hf_dino", str(HF_HUB / "dinov2-small"), 384, "fp16", 224, 16, 1),
    "dinov2-base":  EncoderSpec("dinov2-base", "hf_dino", str(HF_HUB / "dinov2-base"), 768, "fp16", 224, 16, 1),
    "dinov2-large": EncoderSpec("dinov2-large", "hf_dino", str(HF_HUB / "dinov2-large"), 1024, "fp16", 224, 16, 1),
    # ---- DINOv3 (patch16, 1 CLS + 4 register tokens, bf16 mandatory) ----
    # Smaller debug model (ungated HF-format re-upload, camenduru) — fetch/run faster than H+ while 7B downloads.
    # (ViT-S/B only exist as raw .pth upstream; they'd need a .pth->HF conversion before this AutoModel path can load them.)
    "dinov3-base":      EncoderSpec("dinov3-base", "hf_dino", str(REPO / "temp/dinov3_vitb16"), 768, "fp16", 256, 16, 5),
    "dinov3-l":         EncoderSpec("dinov3-l", "hf_dino", str(REPO / "temp/dinov3_vitl16"), 1024, "bf16", 256, 16, 5),
    "dinov3-h":         EncoderSpec("dinov3-h", "hf_dino", str(HF_HUB / "dinov3-vith16plus-pretrain-lvd1689m"), 1280, "bf16", 256, 16, 5),
    "dinov3-7b":        EncoderSpec("dinov3-7b", "hf_dino", str(REPO / "temp/dinov3_7b"), 4096, "bf16", 256, 16, 5),
    "dinov3-7b-int8":   EncoderSpec("dinov3-7b-int8", "hf_dino", str(REPO / "temp/dinov3_7b_int8"), 4096, "int8", 256, 16, 5),
    # ---- Wan2.2 VAE latent (appearance) ----
    "wan-vae":      EncoderSpec("wan-vae", "wan_vae", str(REPO / "kai0/checkpoints/Wan2.2-TI2V-5B-Diffusers"), 48 * 16 * 16, "fp16", 256, 16, 0),
}

# Back-compat alias used by the legacy `CRAVE_ENC` env values.
LEGACY_ALIASES = {
    "dinov2": "dinov2-large",
    "dinov3l": "dinov3-l",
    "dinov3h": "dinov3-h",
    "dinov3_7b": "dinov3-7b",
    "dinov3_7b_int8": "dinov3-7b-int8",
}


def resolve(name: str) -> EncoderSpec:
    """Look up an encoder by canonical name or legacy alias."""
    key = LEGACY_ALIASES.get(name, name)
    if key not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; known: {sorted(ENCODERS)} (aliases {sorted(LEGACY_ALIASES)})")
    return ENCODERS[key]
