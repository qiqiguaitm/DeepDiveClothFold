"""Faithful π0.5 SigLIP-So400m/14 vision tower, loaded DIRECTLY from the big_vision PaliGemma
pt_224.npz (JAX). Pure-torch, no transformers. Mirrors the DINOv3 standalone pattern in this repo.

Layout (verified from npz): img/embedding (14x14x3x1152 conv) + img/pos_embedding (1,256,1152);
27 stacked encoderblocks {LayerNorm_0, MHA(q/k/v/out, 16x72), LayerNorm_1, MlpBlock(Dense_0 gelu_tanh
Dense_1, 4304)}; img/Transformer/encoder_norm. Pre-LN residual. No CLS/register -> 256 patch tokens.
Output = encoder_norm(last hidden) = the 1152-d patch grid the PaliGemma projector consumes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

D, NH, HD, NL, EPS = 1152, 16, 72, 27, 1e-6


class SiglipBigVision:
    def __init__(self, npz_path, device="cuda", dtype=torch.float32):
        z = np.load(npz_path)
        keys = {k for k in z.files}
        pre = "params/" if "params/img/embedding/kernel" in keys else ""
        def g(k):
            return torch.tensor(np.asarray(z[pre + "img/" + k]), dtype=dtype, device=device)
        T = "Transformer/encoderblock/"
        self.patch_w = g("embedding/kernel").permute(3, 2, 0, 1).contiguous()  # (1152,3,14,14)
        self.patch_b = g("embedding/bias")
        self.pos = g("pos_embedding")                                          # (1,256,1152)
        self.ln0_s, self.ln0_b = g(T + "LayerNorm_0/scale"), g(T + "LayerNorm_0/bias")
        self.ln1_s, self.ln1_b = g(T + "LayerNorm_1/scale"), g(T + "LayerNorm_1/bias")
        M = T + "MultiHeadDotProductAttention_0/"
        self.qk, self.qb = g(M + "query/kernel"), g(M + "query/bias")          # (27,1152,16,72),(27,16,72)
        self.kk, self.kb = g(M + "key/kernel"), g(M + "key/bias")
        self.vk, self.vb = g(M + "value/kernel"), g(M + "value/bias")
        self.ok, self.ob = g(M + "out/kernel"), g(M + "out/bias")              # (27,16,72,1152),(27,1152)
        self.d0k, self.d0b = g(T + "MlpBlock_0/Dense_0/kernel"), g(T + "MlpBlock_0/Dense_0/bias")
        self.d1k, self.d1b = g(T + "MlpBlock_0/Dense_1/kernel"), g(T + "MlpBlock_0/Dense_1/bias")
        self.enc_s, self.enc_b = g("Transformer/encoder_norm/scale"), g("Transformer/encoder_norm/bias")
        self.device = device

    @torch.no_grad()
    def _forward(self, px):                                                    # px (N,3,224,224) in [-1,1]
        x = F.conv2d(px, self.patch_w, self.patch_b, stride=14)                # (N,1152,16,16)
        x = x.flatten(2).transpose(1, 2) + self.pos                            # (N,256,1152)
        for l in range(NL):
            h = F.layer_norm(x, (D,), self.ln0_s[l], self.ln0_b[l], EPS)
            q = torch.einsum("nld,dhk->nlhk", h, self.qk[l]) + self.qb[l]
            k = torch.einsum("nld,dhk->nlhk", h, self.kk[l]) + self.kb[l]
            v = torch.einsum("nld,dhk->nlhk", h, self.vk[l]) + self.vb[l]
            a = torch.einsum("nlhk,nmhk->nhlm", q, k) * (HD ** -0.5)
            o = torch.einsum("nhlm,nmhk->nlhk", a.softmax(-1), v)
            x = x + torch.einsum("nlhk,hkd->nld", o, self.ok[l]) + self.ob[l]
            h = F.layer_norm(x, (D,), self.ln1_s[l], self.ln1_b[l], EPS)
            h = F.gelu(h @ self.d0k[l] + self.d0b[l], approximate="tanh")
            x = x + (h @ self.d1k[l] + self.d1b[l])
        return F.layer_norm(x, (D,), self.enc_s, self.enc_b, EPS)              # (N,256,1152)

    @torch.no_grad()
    def encode_grid(self, imgs_u8, bs=32):
        """imgs_u8 (N,224,224,3) uint8 -> (N,1152,16,16) float32."""
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        out = []
        for s in range(0, len(imgs_u8), bs):
            x = torch.from_numpy(imgs_u8[s:s + bs]).to(self.device).float().permute(0, 3, 1, 2) / 255.0
            x = (x - mean) / mean
            h = self._forward(x)                                              # (B,256,1152)
            P = int(round(h.shape[1] ** 0.5))
            out.append(h.permute(0, 2, 1).reshape(h.shape[0], D, P, P).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
