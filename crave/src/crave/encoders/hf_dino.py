"""DINOv2 / DINOv3 encoder via HuggingFace transformers.

Handles all three dtypes (fp16 for dinov2, bf16 for dinov3, int8 via bitsandbytes for
the 7B), the prefix-token skip (CLS + register tokens), and the per-encoder resolution
that pins the patch grid to grid×grid. Lazy-loads the model on first use.
"""
from __future__ import annotations

import os

import numpy as np

from crave.config.encoders import EncoderSpec
from crave.encoders.base import Encoder


class HFDinoEncoder(Encoder):
    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        super().__init__(spec, device)
        self._proc = None
        self._model = None

    # -- lazy load --------------------------------------------------------
    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._torch = torch
        self._proc = AutoImageProcessor.from_pretrained(self.spec.path)
        dt = self.spec.dtype
        if dt == "int8":
            from transformers import BitsAndBytesConfig
            self._model = AutoModel.from_pretrained(
                self.spec.path, quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()  # bnb auto-places on GPU
            self._in_dtype = torch.bfloat16
        elif dt == "bf16":
            self._model = AutoModel.from_pretrained(self.spec.path, dtype=torch.bfloat16).to(self.device).eval()
            self._in_dtype = torch.bfloat16
        elif dt == "fp32":
            self._model = AutoModel.from_pretrained(self.spec.path).to(self.device).eval()
            self._in_dtype = None
        else:  # fp16
            self._model = AutoModel.from_pretrained(self.spec.path).half().to(self.device).eval()
            self._in_dtype = torch.float16

    def _prep(self, batch):
        sz = {"height": self.spec.res, "width": self.spec.res}
        px = self._proc(images=batch, return_tensors="pt", size=sz).to(self.device)
        if self._in_dtype is not None:
            px = {k: (v.to(self._in_dtype) if self._torch.is_floating_point(v) else v) for k, v in px.items()}
        return px

    # -- public API -------------------------------------------------------
    def encode_pooled(self, imgs, bs: int = 128) -> np.ndarray:
        self._ensure()
        torch = self._torch
        out = np.zeros((len(imgs), self.spec.dim), np.float32)
        npf = self.spec.nprefix
        for b in range(0, len(imgs), bs):
            with torch.no_grad():
                px = self._prep(imgs[b:b + bs])
                hs = self._model(**px).last_hidden_state[:, npf:].mean(1)
                out[b:b + bs] = hs.float().cpu().numpy()
        return out

    def encode_grid(self, imgs, bs: int = 64) -> np.ndarray:
        self._ensure()
        torch = self._torch
        P, dim, npf = self.spec.grid, self.spec.dim, self.spec.nprefix
        grids = np.zeros((len(imgs), dim, P, P), np.float16)
        for b in range(0, len(imgs), bs):
            with torch.no_grad():
                px = self._prep(imgs[b:b + bs])
                toks = self._model(**px).last_hidden_state[:, npf:]
                side = int(round(toks.shape[1] ** 0.5))
                assert side == P, f"grid {side}!=P({P}) for {self.spec.name}; res={self.spec.res} patch mismatch"
                g = toks.reshape(toks.shape[0], P, P, dim).permute(0, 3, 1, 2).float().contiguous().cpu().numpy()
            grids[b:b + bs] = g.astype(np.float16)
        return grids

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
