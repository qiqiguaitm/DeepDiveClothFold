"""LeWMVisionEncoder — pi0.5 视觉前端的【旁路变体】: DINOv3-L/16(冻结) + kai0-LeWM
OctCompactor(per-view) + 新投影 256→gemma_width。把官方 SigLIP 的 768 dense token 换成
15 个 object-centric token(3 view × (1 CLS + 4 obj))。

⚠️ 纯加法、零侵入: 本文件不 import / 不修改任何现有 openpi 模型代码; SigLIP 路径完全不动。
只有当某个 config 显式设 vision_encoder="lewm" 时, pi0_pytorch 才走到这里(见该处旁路分支)。

OctCompactor / DINOv3 加载逻辑 vendored 自 LeWM 源 (kai0_lewm.py / dino_backbone.py),
权重 strict 载入(已核对 ckpt compactor.{th,hl,hr}.* 与本定义逐键一致)。

plan: docs/.../pi05_from_paligemma_base_training_plan.md §9
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# 视角顺序 = pi0.5 三相机 [top_head, hand_left, hand_right] ↔ LeWM (th, hl, hr)
VIEWS = ("th", "hl", "hr")
# per-view 忠实分辨率 (H, W)，patch16 → th 18×24=432, wrist 12×16=192 (§9.1/§9.4-Q2)
VIEW_HW = {"th": (288, 384), "hl": (192, 256), "hr": (192, 256)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV3_VITL16_DIR_DEFAULT = (
    "/vePFS-North-E/shared_data/shock/.CACHE/hf_cache/hub/dinov3-vitl16-pretrain-lvd1689m"
)


# ---- vendored OctCompactor (与 LeWM ckpt 逐键一致) -----------------------------
class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, q, kv):
        attn_out, _ = self.cross(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out
        q = q + self.mlp(self.norm_m(q))
        return q


class OctCompactor(nn.Module):
    """Learnable 1 CLS + n_obj queries cross-attending DINOv3-L patches → (B,1+n_obj,d_model)."""
    def __init__(self, n_obj: int = 4, d_in: int = 1024, d_model: int = 256,
                 depth: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_obj = int(n_obj)
        self.in_proj = nn.Linear(d_in, d_model)
        self.queries = nn.Parameter(torch.randn(1, 1 + self.n_obj, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d_model, heads, 4.0, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat):
        # feat: (B, P, d_in) → (B, 1+n_obj, d_model)
        kv = self.in_proj(feat)
        q = self.queries.expand(feat.size(0), -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)


# ---- frozen DINOv3-L/16 (HF AutoModel, offline) -------------------------------
class _DinoV3Frozen(nn.Module):
    """冻结 DINOv3-ViT-L/16, 返回 P 个 patch token (B,P,1024)。CLS+register 丢弃。
    权重不进 state_dict(buffer-free 持有), 故本 encoder ckpt 不含 1.2G DINO 权重。"""
    _CACHE: dict = {}

    def __init__(self, local_dir: str):
        super().__init__()
        self.local_dir = local_dir
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(local_dir)
        self.patch_size = int(cfg.patch_size)
        self.hidden = int(cfg.hidden_size)  # 1024
        nreg = getattr(cfg, "num_register_tokens", None)
        self.n_prefix = 1 + (int(nreg) if nreg is not None else 4)  # CLS + registers

    def _bb(self, device):
        bb = _DinoV3Frozen._CACHE.get(self.local_dir)
        if bb is None:
            from transformers import AutoModel
            prev = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                bb = AutoModel.from_pretrained(self.local_dir)
            finally:
                os.environ["HF_HUB_OFFLINE"] = prev if prev is not None else "1"
            for p in bb.parameters():
                p.requires_grad_(False)
            bb.eval()
            _DinoV3Frozen._CACHE[self.local_dir] = bb
        if next(bb.parameters()).device != device:
            bb.to(device)
        return bb

    @torch.no_grad()
    def forward(self, pixels):  # (B,3,H,W) imagenet-normed → (B,P,1024)
        bb = self._bb(pixels.device)
        lhs = bb(pixel_values=pixels, interpolate_pos_encoding=True).last_hidden_state
        return lhs[:, self.n_prefix:, :]  # drop CLS+registers → P patch tokens


# ---- the encoder --------------------------------------------------------------
class LeWMVisionEncoder(nn.Module):
    """3 view → DINOv3-L/16(frozen) → per-view OctCompactor → concat 15×256 → proj 256→width.

    Args:
        gemma_width: PaliGemma LLM hidden width (proj 输出维度)。
        freeze_compactor: True=用 LeWM 学到的表示(冻 compactor); False=随策略微调。
        dinov3_dir: DINOv3-L/16 本地权重目录 (offline)。
    forward(images): images = [th, hl, hr] 三个 (B,3,H_v,W_v) 已按 per-view 分辨率+ImageNet 归一化。
        → (B, 15, gemma_width)
    """
    def __init__(self, gemma_width: int, freeze_compactor: bool = False,
                 dinov3_dir: str = DINOV3_VITL16_DIR_DEFAULT, n_obj: int = 4, d_model: int = 256):
        super().__init__()
        self.dino = _DinoV3Frozen(dinov3_dir)
        self.compactor = nn.ModuleDict({
            v: OctCompactor(n_obj=n_obj, d_in=self.dino.hidden, d_model=d_model)
            for v in VIEWS
        })
        self.proj = nn.Linear(d_model, gemma_width)
        self.freeze_compactor = bool(freeze_compactor)
        if self.freeze_compactor:
            for p in self.compactor.parameters():
                p.requires_grad_(False)
        self.tokens_per_view = 1 + n_obj
        self.num_tokens = self.tokens_per_view * len(VIEWS)  # 15

    def load_compactor_ckpt(self, ckpt_path: str):
        """Strict-load compactor.{th,hl,hr}.* from a LeWM Kai0LeWM ckpt."""
        sd = torch.load(ckpt_path, map_location="cpu")
        for k in ("model", "state_dict", "ema"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
        sub = {k[len("compactor."):]: v for k, v in sd.items() if k.startswith("compactor.")}
        missing, unexpected = self.compactor.load_state_dict(sub, strict=True)
        return missing, unexpected

    def forward(self, images: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(images) == len(VIEWS), f"need {len(VIEWS)} views, got {len(images)}"
        toks = []
        for v, img in zip(VIEWS, images):
            patches = self.dino(img)                       # (B,P,1024) no-grad
            comp = self.compactor[v](patches)              # (B,5,256)
            toks.append(comp)
        x = torch.cat(toks, dim=1)                         # (B,15,256)
        return self.proj(x)                                # (B,15,width)
