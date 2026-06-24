"""簇中心渲染:用 Wan2.2 VAE(高保真重建解码器)渲染 milestone 簇中心,对比现有 small 解码器。
聚类仍用 DINOv2-large(语义),渲染用 Wan VAE(只跑推理, 不下主体)。四行对比:
  (1) Wan VAE 簇内 latent 平均 → 解码(合成质心)
  (2) Wan VAE medoid latent → 解码(锐利代表)
  (3) DINOv2 small 解码器 簇中心(现方案)
  (4) 最近真实帧(medoid 原图)
Run: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_wanvae_centroid.py
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from crave.config import resolve_dataset, viz_dir
from crave.data import kai0
from crave.decoding import train_dec
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import otsu

OUTV = viz_dir("centroid_decoder")
RES = 128; P = 16; DIM = 1024; dev = "cuda"


def encode_grids_valid(enc, imgs224, valid):
    """Byte-identical to legacy encode_grids: encode only valid frames into a zero
    (N,DIM,16,16) float16 array (invalid rows stay zero-grids)."""
    grids = np.zeros((len(imgs224), enc.dim, P, P), np.float16)
    idxs = np.where(valid)[0]
    if len(idxs):
        g = enc.encode_grid([imgs224[i] for i in idxs])
        for k_, i in enumerate(idxs):
            grids[i] = g[k_]
    return grids


def main():
    t0 = time.time(); MINE = 120
    cfg = resolve_dataset("kai0_base")
    from pathlib import Path
    rawset = set(int(p.stem[2:]) for p in Path(cfg.raw_cache).glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in Path(cfg.arm_cache).glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:MINE].tolist())
    E_, FR_, T_ = [], [], []
    for e in mined:
        _, _, _, n = kai0.loadep_tcc(cfg, e)
        for i in range(n): E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_)
    print(f"挖矿 {len(mined)} ep, {len(E_)} 帧; 并行解码 ...", flush=True)
    imgs224, valid = kai0.decode_images(cfg, np.arange(len(E_)), E_, FR_)
    imgs128 = np.stack([cv2.resize(imgs224[i], (RES, RES), interpolation=cv2.INTER_AREA) for i in range(len(imgs224))]).astype(np.uint8)
    print(f"  解码完成 ({time.time()-t0:.0f}s); DINOv2-large grids ...", flush=True)
    enc = load_encoder("dinov2-large", dtype="fp32")   # 局部镜像 fp32 = 旧 LARGE 路径
    grids = encode_grids_valid(enc, imgs224, valid)
    pooled = grids.reshape(len(grids), DIM, -1).mean(2).astype(np.float32); pooled /= (np.linalg.norm(pooled, 1, keepdims=True) + 1e-9)

    from sklearn.cluster import KMeans
    km = KMeans(96, n_init=3, random_state=0).fit(pooled); lab = km.labels_; cen = km.cluster_centers_
    tpos = np.array([T_[lab == c].mean() for c in range(96)]); cov = np.array([len(set(E_[lab == c].tolist())) / len(mined) for c in range(96)])
    tau = otsu(cov); sel = sorted([c for c in range(96) if cov[c] >= tau], key=lambda c: tpos[c])
    NS = min(12, len(sel)); sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]
    print(f"自适应 milestone {len([c for c in range(96) if cov[c]>=tau])}, 展示 {NS}; 训 small 解码器 ...", flush=True)
    decf = train_dec(grids, imgs128, DIM, "small", 55)

    # Wan VAE
    print("加载 Wan2.2 VAE ...", flush=True)
    wan = load_encoder("wan-vae")

    def c256(i): return cv2.resize(imgs224[i], (256, 256), interpolation=cv2.INTER_AREA)

    rows = {"wan_avg": [], "wan_med": [], "small": [], "near": []}
    for c in sel:
        mem = np.where(lab == c)[0]
        d = np.linalg.norm(pooled[mem] - cen[c], axis=1); order = mem[np.argsort(d)]
        keep = order[:40]; md = order[0]
        zs = wan.encode_latents([c256(i) for i in keep])                 # (n,48,16,16) float32
        rows["wan_avg"].append(wan.decode(zs.mean(0, keepdims=True))[0])
        rows["wan_med"].append(wan.decode(zs[:1])[0])
        rows["small"].append(cv2.resize(decf(grids[mem].astype(np.float32).mean(0)[None])[0], (256, 256)))
        rows["near"].append(c256(md))
    print(f"  渲染完成 ({time.time()-t0:.0f}s); 出图 ...", flush=True)

    plt = setup_mpl()
    labels = ["(1) Enc: DINOv2-large(300M)\nDec: Wan2.2-VAE\nlatent-MEAN (synthetic)",
              "(2) Enc: DINOv2-large(300M)\nDec: Wan2.2-VAE\nmedoid (1 real frame)",
              "(3) Enc: DINOv2-large(300M)\nDec: trained CNN 0.92M\ngrid-MEAN (current std)",
              "(4) Enc/Dec: none\nnearest REAL frame"]
    keys = ["wan_avg", "wan_med", "small", "near"]
    fig, axes = plt.subplots(4, NS, figsize=(1.7 * NS, 6.8))
    for r, k in enumerate(keys):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[k][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle("Cluster-center rendering — CLUSTERING encoder = DINOv2-large for all rows; rows differ by DECODER.  (120ep@3Hz, columns=milestones by progress P)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTV / "crave_wanvae_centroid.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED crave_wanvae_centroid.png  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
