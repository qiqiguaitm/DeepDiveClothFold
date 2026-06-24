"""Standalone pure-PyTorch DINOv3-ViT — NO transformers dependency.

Faithful port of transformers `DINOv3ViTModel` (model_type=dinov3_vit) forward, with param
names matching the HF safetensors so `load_state_dict` maps 1:1. Needed because kai0 pins
transformers==4.53.2 (+ transformers_replace siglip patch), which lacks dinov3; upgrading
would break pi0_pytorch. This module loads the exact HF lvd1689m weights the LeWM compactor
was distilled on → features match the teacher (cache_dinov3L_kai0.py used the same weights).

forward(pixels [B,3,H,W], ImageNet-normalized) → last_hidden_state (B, 1+R+P, hidden).
Token order = [CLS, R register tokens, P patch tokens]; take [:, 1+R:] for patch tokens.

ref: transformers 5.5.3 models/dinov3_vit/modeling_dinov3_vit.py (config: ViT-L/16, gated_mlp=False,
gelu, 24 layers, head_dim 64, 4 reg, rope_theta 100, eval→no pos-aug).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class _Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden_size"]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, h))            # unused at inference, present in ckpt
        self.register_tokens = nn.Parameter(torch.zeros(1, c["num_register_tokens"], h))
        self.patch_embeddings = nn.Conv2d(c["num_channels"], h, c["patch_size"], c["patch_size"])

    def forward(self, px):
        pe = self.patch_embeddings(px).flatten(2).transpose(1, 2)       # (B,P,h)
        b = px.shape[0]
        return torch.cat([self.cls_token.expand(b, -1, -1),
                          self.register_tokens.expand(b, -1, -1), pe], dim=1)


def _rope_coords(nph, npw, device):
    ch = torch.arange(0.5, nph, dtype=torch.float32, device=device) / nph
    cw = torch.arange(0.5, npw, dtype=torch.float32, device=device) / npw
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)  # (P,2)
    return 2.0 * coords - 1.0


class _Rope(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch = c["patch_size"]
        head_dim = c["hidden_size"] // c["num_attention_heads"]
        inv = 1.0 / (c["rope_theta"] ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32))  # (head_dim/4,)
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, px):
        _, _, H, W = px.shape
        coords = _rope_coords(H // self.patch, W // self.patch, px.device)               # (P,2)
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]          # (P,2,head_dim/4)
        angles = angles.flatten(1, 2).tile(2)                                            # (P,head_dim)
        return torch.cos(angles).to(px.dtype), torch.sin(angles).to(px.dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    np_ = sin.shape[-2]
    npre = q.shape[-2] - np_                                  # CLS + register prefix
    qp, qP = q.split((npre, np_), dim=-2)
    kp, kP = k.split((npre, np_), dim=-2)
    qP = qP * cos + _rotate_half(qP) * sin
    kP = kP * cos + _rotate_half(kP) * sin
    return torch.cat((qp, qP), dim=-2), torch.cat((kp, kP), dim=-2)


class _Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c["num_attention_heads"]
        self.hd = c["hidden_size"] // self.nh
        d = c["hidden_size"]
        self.q_proj = nn.Linear(d, d, bias=c["query_bias"])
        self.k_proj = nn.Linear(d, d, bias=c["key_bias"])
        self.v_proj = nn.Linear(d, d, bias=c["value_bias"])
        self.o_proj = nn.Linear(d, d, bias=c["proj_bias"])

    def forward(self, x, pos):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.nh, self.hd).transpose(1, 2)
        cos, sin = pos
        q, k = _apply_rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)          # scaling = hd**-0.5 (default), no mask
        o = o.transpose(1, 2).reshape(b, n, -1)
        return self.o_proj(o)


class _LayerScale(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(c["hidden_size"]) * c["layerscale_value"])

    def forward(self, x):
        return x * self.lambda1


class _MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=c["mlp_bias"])
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=c["mlp_bias"])
        self.act = nn.GELU()                                  # hidden_act=gelu (exact erf)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class _Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        eps, d = c["layer_norm_eps"], c["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=eps)
        self.attention = _Attn(c)
        self.layer_scale1 = _LayerScale(c)
        self.norm2 = nn.LayerNorm(d, eps=eps)
        self.mlp = _MLP(c)
        self.layer_scale2 = _LayerScale(c)

    def forward(self, x, pos):
        x = x + self.layer_scale1(self.attention(self.norm1(x), pos))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DINOv3ViTStandalone(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        c = json.loads((Path(model_dir) / "config.json").read_text())
        assert not c.get("use_gated_mlp", False), "this port assumes gated_mlp=False (ViT-L/16 lvd1689m)"
        self.cfg = c
        self.hidden = c["hidden_size"]
        self.n_prefix = 1 + c["num_register_tokens"]
        self.embeddings = _Embeddings(c)
        self.rope_embeddings = _Rope(c)
        self.layer = nn.ModuleList([_Layer(c) for _ in range(c["num_hidden_layers"])])
        self.norm = nn.LayerNorm(self.hidden, eps=c["layer_norm_eps"])
        sd = load_file(str(Path(model_dir) / "model.safetensors"))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "inv_freq" not in m]   # buffer, not in ckpt
        if missing or unexpected:
            raise RuntimeError(f"DINOv3 weight load mismatch: missing={missing}, unexpected={unexpected}")
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, px):
        x = self.embeddings(px)
        pos = self.rope_embeddings(px)
        for L in self.layer:
            x = L(x, pos)
        return self.norm(x)                                    # (B, 1+R+P, hidden); patches = [:, n_prefix:]
