#!/usr/bin/env python
"""编码→解码往返一致性:帧 → DINOv3-H encode → decode 回图,与原始比一致性。

"解码回来"=自重建(retrieval 会返回另一帧,不适用),用最忠实的合成解码器 **patch-grid**
(pooled-L1 作对照)。500 帧聚合 + 12 帧画廊。一致性指标:
  - L1(重建 vs 原始, /255)
  - 再编码 cos(重建 → DINOv3-H pooled → 与原帧 latent 的 cos)= 语义一致性
  - SSIM(结构相似)
Run: REPO=/home/tim/workspace/deepdive_kai0 PYTHONPATH=crave/src:lmwm/src:lmwm/scripts \
  /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/roundtrip_consistency.py
输出: crave/docs/visualization/decoder_benchmark/roundtrip_consistency.png
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from crave.decoding.decoder import make_decoder
from crave.encoders import load_encoder
from crave.render import setup_mpl
from lmwm.retrieval_decoder import LatentRetrievalDecoder
from train_dinov3h_decoder import PooledDecoder

plt = setup_mpl()
REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FEAT = REPO / "temp/crave_full_dinov3h"
DATA = REPO / "kai0/data/Task_A/kai0_base"
CKPT = REPO / "lmwm/checkpoints"
OUT = REPO / "crave/docs/visualization/decoder_benchmark"
DEV, RES, N_AGG, N_SHOW = "cuda", 128, 500, 12
rng = np.random.RandomState(7)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def to_uint8(t):
    o = t.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip((o + 1) * 127.5, 0, 255).astype(np.uint8)


def main():
    enc = load_encoder("dinov3-h", device=DEV)
    rdec = LatentRetrievalDecoder(feature_dir=FEAT, dataset_root=DATA, device=DEV, res=None)
    mz = np.load(FEAT / "milestones_uniform_dinov3h.npz")
    proto, pord = l2(mz["C"].astype(np.float32)), mz["Pord"].astype(np.float32)

    q = rng.choice(len(rdec.feat), N_AGG, replace=False)
    q_pool = rdec.feat[q].astype(np.float32)
    real = np.stack([cv2.resize(rdec.frame(int(g)), (RES, RES)) for g in q])
    print(f"encoding {N_AGG} grids ...", flush=True)
    grids = enc.encode_grid(list(real)).astype(np.float32)

    # patch-grid decoder
    dpk = torch.load(CKPT / "patch_decoder/patch_dec.pt", map_location=DEV, weights_only=False)
    Dp = make_decoder(dpk["din"], dpk["dec"]).to(DEV).eval(); Dp.load_state_dict(dpk["model"])
    muT = torch.from_numpy(dpk["mu"]).view(1, -1, 1, 1).float().to(DEV)
    sdT = torch.from_numpy(dpk["sd"]).view(1, -1, 1, 1).float().to(DEV)
    with torch.no_grad():
        patch = to_uint8(Dp((torch.from_numpy(grids).to(DEV) - muT) / sdT))
    # pooled-L1 decoder (contrast)
    dp2 = torch.load(CKPT / "dinov3h_decoder/dec.pt", map_location=DEV, weights_only=False)
    Gp = PooledDecoder(din=dp2["din"], res=dp2["res"]).to(DEV).eval(); Gp.load_state_dict(dp2["model"])
    with torch.no_grad():
        pooled = to_uint8(Gp(torch.from_numpy(l2(q_pool)).to(DEV)))

    def stats(recon):
        l1 = np.abs(recon.astype(np.float32) - real.astype(np.float32)).mean((1, 2, 3)) / 255
        re = l2(enc.encode_pooled(list(recon)).astype(np.float32))
        cos = np.sum(re * q_pool, axis=1)
        ss = np.array([ssim(real[i], recon[i]) for i in range(len(recon))])
        return l1, cos, ss
    pl1, pcos, pss = stats(patch)
    ol1, ocos, oss = stats(pooled)
    for nm, (a, c, s) in [("patch-grid", (pl1, pcos, pss)), ("pooled-L1", (ol1, ocos, oss))]:
        print(f"{nm:11s} L1={a.mean():.3f}±{a.std():.3f} | reencode_cos={c.mean():.3f}±{c.std():.3f} | SSIM={s.mean():.3f}", flush=True)

    # ---- gallery: N_SHOW frames spanning progress ----
    prog = pord[(q_pool @ proto.T).argmax(1)]
    show = [int(np.argmin(np.abs(prog - t))) for t in np.linspace(0.03, 0.95, N_SHOW)]

    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(4, N_SHOW, height_ratios=[1, 1, 1, 1.5], hspace=0.22, wspace=0.06)
    rows = [("原始帧", real), ("patch-grid 解码回来", patch), ("pooled-L1 解码回来", pooled)]
    for r, (lab, imgs) in enumerate(rows):
        for c, i in enumerate(show):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(imgs[i]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=12, rotation=0, ha="right", va="center")
            if r == 0:
                ax.set_title(f"进度{prog[i]:.2f}", fontsize=9)
            if r == 1:
                ax.set_xlabel(f"cos {pcos[i]:.2f}", fontsize=9)
    # aggregate histograms
    axc = fig.add_subplot(gs[3, :N_SHOW // 2])
    axc.hist(pcos, bins=np.linspace(0, 1, 41), color="#1f77b4", alpha=.8, label=f"patch-grid  均值 {pcos.mean():.3f}")
    axc.hist(ocos, bins=np.linspace(0, 1, 41), color="#9467bd", alpha=.6, label=f"pooled-L1  均值 {ocos.mean():.3f}")
    axc.axvline(1.0, color="#2ca02c", ls="--", lw=1.5, label="完美一致 =1.0")
    axc.set_xlabel("再编码 cos(重建→DINOv3-H→与原帧 latent 的 cos)= 语义一致性"); axc.set_ylabel("帧数")
    axc.set_title(f"往返语义一致性分布（{N_AGG} 帧）", fontsize=11); axc.legend(fontsize=9); axc.grid(alpha=.25)
    axl = fig.add_subplot(gs[3, N_SHOW // 2:])
    axl.hist(pl1, bins=40, color="#1f77b4", alpha=.8, label=f"patch-grid  L1 {pl1.mean():.3f}")
    axl.hist(ol1, bins=40, color="#9467bd", alpha=.6, label=f"pooled-L1  L1 {ol1.mean():.3f}")
    axl.set_xlabel("像素 L1(重建 vs 原始, /255)"); axl.set_ylabel("帧数")
    axl.set_title(f"往返像素误差分布（{N_AGG} 帧）· patch-grid SSIM={pss.mean():.2f}", fontsize=11); axl.legend(fontsize=9); axl.grid(alpha=.25)

    fig.suptitle(f"DINOv3-H 编码→解码 往返一致性 · patch-grid 解码器 · {N_AGG} 帧 · kai0_base"
                 f"  —— 结构/颜色/臂位保住(L1 {pl1.mean():.1%}, 语义 cos {pcos.mean():.2f}),仅高频略软",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT / "roundtrip_consistency.png", dpi=115, bbox_inches="tight")
    print("SAVED", OUT / "roundtrip_consistency.png", flush=True)


if __name__ == "__main__":
    main()
