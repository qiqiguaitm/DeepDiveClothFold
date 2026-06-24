"""Abstract encoder interface.

Every encoder exposes the same two operations CRAVE needs from a frozen backbone:
  - `encode_pooled(imgs)` → (N, dim)        global per-frame descriptor (for clustering)
  - `encode_grid(imgs)`   → (N, dim, P, P)  patch grid (for the centroid decoder)

`imgs` is a list/array of HxWx3 uint8 RGB frames; the encoder owns resize/normalize.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from crave.config.encoders import EncoderSpec


class Encoder(ABC):
    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def grid(self) -> int:
        return self.spec.grid

    @abstractmethod
    def encode_pooled(self, imgs, bs: int = 128) -> np.ndarray:
        ...

    @abstractmethod
    def encode_grid(self, imgs, bs: int = 64) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        s = self.spec
        return f"<Encoder {s.name} dim={s.dim} dtype={s.dtype} res={s.res} grid={s.grid} nprefix={s.nprefix}>"
