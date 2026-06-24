"""Encoder factory — `load_encoder(name)` is the only entry point downstream needs.

    from crave.encoders import load_encoder
    enc = load_encoder("dinov3-h")          # or "dinov2-large", "dinov3-7b-int8", "wan-vae"
    pooled = enc.encode_pooled(frames)      # (N, dim)
    grids  = enc.encode_grid(frames)        # (N, dim, P, P)

Selection can also come from the env var CRAVE_ENC (canonical name or legacy alias).
"""
from __future__ import annotations

import os
from dataclasses import replace

from crave.config.encoders import ENCODERS, EncoderSpec, resolve
from crave.encoders.base import Encoder
from crave.encoders.hf_dino import HFDinoEncoder
from crave.encoders.wan_vae import WanVAEEncoder

_KINDS = {"hf_dino": HFDinoEncoder, "wan_vae": WanVAEEncoder}


def load_encoder(name: str | None = None, device: str = "cuda",
                 dtype: str | None = None, path: str | None = None) -> Encoder:
    """Instantiate an encoder by registry name (or $CRAVE_ENC).

    `dtype`/`path` override the registry spec — e.g. `load_encoder("dinov2-small",
    dtype="fp32", path="facebook/dinov2-small")` to reproduce legacy fp32 hub-id scoring.
    """
    if name is None:
        name = os.environ.get("CRAVE_ENC", "dinov2-large")
    spec: EncoderSpec = resolve(name)
    if dtype is not None or path is not None:
        spec = replace(spec, dtype=dtype or spec.dtype, path=path or spec.path)
    return _KINDS[spec.kind](spec, device=device)


__all__ = ["Encoder", "HFDinoEncoder", "WanVAEEncoder", "EncoderSpec", "ENCODERS", "load_encoder", "resolve"]
