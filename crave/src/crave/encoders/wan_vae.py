"""Wan2.2 VAE latent encoder (appearance features, vs DINO's semantics).

Ported from crave_wanvae_centroid.py: 256×256 uint8 → [-1,1] → add time dim →
`vae.encode(...).latent_dist.mode()` → (N, 48, 16, 16) latent grid.

`encode_grid`   → (N, 48, 16, 16)
`encode_pooled` → (N, 48*16*16) flattened latent (global appearance descriptor)
"""
from __future__ import annotations

import numpy as np

from crave.config.encoders import EncoderSpec
from crave.encoders.base import Encoder

_C, _P = 48, 16  # Wan latent channels / spatial side at 256px


class WanVAEEncoder(Encoder):
    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        super().__init__(spec, device)
        self._vae = None

    def _ensure(self):
        if self._vae is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan
        self._torch = torch
        self._vae = AutoencoderKLWan.from_pretrained(
            self.spec.path, subfolder="vae", torch_dtype=torch.float32).to(self.device).eval()

    def _latents(self, imgs, bs):
        """→ (N, 48, 16, 16) float32."""
        self._ensure()
        torch = self._torch
        outs = []
        for b in range(0, len(imgs), bs):
            batch = [self._to256(im) for im in imgs[b:b + bs]]
            x = torch.from_numpy(np.stack(batch).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)[:, :, None].to(self.device)
            with torch.no_grad():
                e = self._vae.encode(x)
                z = e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent
            outs.append(z.squeeze(2).float().cpu().numpy())  # drop time dim → (n,48,16,16)
        return np.concatenate(outs, 0)

    @staticmethod
    def _to256(im):
        import cv2
        if im.shape[:2] != (256, 256):
            im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        return im

    def encode_grid(self, imgs, bs: int = 16) -> np.ndarray:
        return self._latents(imgs, bs).astype(np.float16)

    def encode_pooled(self, imgs, bs: int = 16) -> np.ndarray:
        z = self._latents(imgs, bs)
        return z.reshape(len(z), -1).astype(np.float32)

    def encode_latents(self, imgs, bs: int = 16) -> np.ndarray:
        """Raw latents (N, 48, 16, 16) float32 — for medoid/centroid latent ops."""
        return self._latents(imgs, bs)

    def decode(self, latents) -> np.ndarray:
        """Decode latents (N,48,16,16) → uint8 RGB images (N,256,256,3)."""
        self._ensure()
        torch = self._torch
        z = torch.from_numpy(np.asarray(latents, np.float32))
        if z.ndim == 4:
            z = z[:, :, None]            # add time dim → (N,48,1,16,16)
        with torch.no_grad():
            o = self._vae.decode(z.to(self.device)).sample
        o = o.squeeze(2).float().cpu().numpy().transpose(0, 2, 3, 1)
        return np.clip((o + 1) * 127.5, 0, 255).astype(np.uint8)

    def unload(self):
        if self._vae is not None:
            del self._vae
            self._vae = None
            import torch
            torch.cuda.empty_cache()
