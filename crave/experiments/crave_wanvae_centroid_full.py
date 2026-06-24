"""全量 kai0_base @3Hz 聚类 → Wan2.2 VAE 渲染簇中心(看更大数据是否让 milestone/质心更好)。
内存友好:分块 decode→DINOv2-large pooled(只留 pooled, 不留 32GB grids);只对入选 milestone 的成员跑 Wan。
三行:① Wan latent 平均(合成质心) ② Wan medoid 解码(锐利) ③ 最近真实帧。
Run: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_wanvae_centroid_full.py [--mine-n 550]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from crave.config import resolve_dataset, viz_dir
from crave.data import kai0
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import otsu

OUTV = viz_dir("centroid_decoder")
RES = 128; DIM = 1024; dev = "cuda"; P = 16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mine-n", type=int, default=550); ap.add_argument("--chunk", type=int, default=6000)
    a = ap.parse_args(); t0 = time.time()
    cfg = resolve_dataset("kai0_base")
    rawset = set(int(p.stem[2:]) for p in Path(cfg.raw_cache).glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in Path(cfg.arm_cache).glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    E_, FR_, T_ = [], [], []
    for e in mined:
        _, _, _, n = kai0.loadep_tcc(cfg, e)
        for i in range(n): E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_, np.float32); N = len(E_)
    print(f"全量 {len(mined)} ep, {N} 帧 @3Hz; 分块 DINOv2-large pooled ...", flush=True)

    enc = load_encoder("dinov2-large", dtype="fp32")   # 局部镜像 fp32 = 旧 LARGE 路径
    pooled = np.zeros((N, DIM), np.float32)
    for s in range(0, N, a.chunk):
        idx = np.arange(s, min(s + a.chunk, N))
        imgs224, valid = kai0.decode_images(cfg, idx, E_, FR_)
        vi = np.where(valid)[0]
        if len(vi):
            pf = enc.encode_pooled([imgs224[i] for i in vi])   # patch-token 均值池化 (encode_pooled skips CLS)
            for k_, i in enumerate(vi): pooled[s + i] = pf[k_]
        del imgs224; print(f"  pooled {min(s+a.chunk,N)}/{N} ({time.time()-t0:.0f}s)", flush=True)
    enc.unload()
    pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)

    from sklearn.cluster import KMeans
    print("KMeans-96 (全量) ...", flush=True)
    km = KMeans(96, n_init=3, random_state=0).fit(pooled); lab = km.labels_; cen = km.cluster_centers_
    tpos = np.array([T_[lab == c].mean() for c in range(96)])
    cov = np.array([len(set(E_[lab == c].tolist())) / len(mined) for c in range(96)])
    tau = otsu(cov); selall = [c for c in range(96) if cov[c] >= tau]
    sel = sorted(selall, key=lambda c: tpos[c]); NS = min(12, len(sel))
    sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]
    print(f"自适应 milestone {len(selall)} (tau={tau:.3f}), 展示 {NS}; 加载 Wan VAE ...", flush=True)

    wan = load_encoder("wan-vae")

    rows = {"avg": [], "med": [], "near": []}
    for c in sel:
        mem = np.where(lab == c)[0]; d = np.linalg.norm(pooled[mem] - cen[c], axis=1); order = mem[np.argsort(d)][:40]
        need = {}
        for i in order: need.setdefault(int(E_[i]), []).append((int(FR_[i]), int(i)))
        imgs256, ids = [], []
        for e, lst in need.items():
            fm = kai0.grab_ep(cfg, e, [f for f, _ in lst])
            for f, gi in lst:
                if f in fm: imgs256.append(cv2.resize(fm[f], (256, 256), interpolation=cv2.INTER_AREA)); ids.append(gi)
        if not imgs256:
            for k in rows: rows[k].append(np.zeros((256, 256, 3), np.uint8))
            continue
        zs = wan.encode_latents(imgs256)                            # (n,48,16,16) float32
        rows["avg"].append(wan.decode(zs.mean(0, keepdims=True))[0])
        mpos = int(np.argmin([np.linalg.norm(pooled[g] - cen[c]) for g in ids]))   # medoid = 离 center 最近且成功解码的
        rows["med"].append(wan.decode(zs[mpos:mpos + 1])[0]); rows["near"].append(imgs256[mpos])
    print(f"  渲染完成 ({time.time()-t0:.0f}s); 出图 ...", flush=True)

    plt = setup_mpl()
    labels = ["(1) Wan-VAE latent-AVG\n(synthetic centroid)", "(2) Wan-VAE medoid\n(sharp)", "(3) nearest real frame"]
    keys = ["avg", "med", "near"]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 4.9))
    for r, k in enumerate(keys):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[k][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle(f"FULL kai0_base @3Hz ({len(mined)}ep/{N}fr) cluster → Wan2.2-VAE centroid: latent-avg / medoid / nearest-real  (milestones={len(selall)})", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTV / "crave_wanvae_centroid_full.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED crave_wanvae_centroid_full.png  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
