"""Cross-encoder centroid-decode comparison.

For each encoder: encode frames -> KMeans cluster -> per-cluster mean patch-grid ->
train a small 16->128 decoder -> decode the centroids back to images. One row per
encoder, K columns ordered by task phase (mean normalized time), so rows are
visually comparable. Shows how centroid *decoding* quality varies with the backbone.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python \
    crave/experiments/encoder_centroid_decode_compare.py [--ds vis] [--n-eps 45] [--k 10] [--epochs 40]
Out: crave/docs/visualization/encoders/enc_centroid_decode_compare.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.clustering.kmeans import cpu_kmeans          # noqa: E402
from crave.config import resolve_dataset, viz_dir        # noqa: E402
from crave.data.loaders import list_eps, load_ep         # noqa: E402
from crave.decoding.decoder import train_dec             # noqa: E402
from crave.encoders import load_encoder                  # noqa: E402
from crave.render.mpl import setup_mpl                   # noqa: E402

ENCODERS = ["dinov2-large", "dinov3-l", "dinov3-h", "dinov3-7b-int8"]


def L2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="vis")
    ap.add_argument("--n-eps", type=int, default=45)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=5000)
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    a = ap.parse_args()

    cfg = resolve_dataset(a.ds)
    eps = list_eps(cfg)[: a.n_eps]
    print(f"[load] {a.ds}: {len(eps)} eps (stride {cfg.stride})...", flush=True)

    f224, th128, tpos = [], [], []
    for e in eps:
        fr, _st, thumb, _ni = load_ep(cfg, e, cfg.stride)
        n = len(fr)
        if n == 0:
            continue
        f224.extend(fr)
        th128.extend(thumb)
        tpos.extend((np.arange(n) / max(n - 1, 1)).tolist())   # normalized time in-episode
    f224 = np.asarray(f224)
    th128 = np.asarray(th128)
    tpos = np.asarray(tpos, np.float32)
    if len(f224) > a.max_frames:                                # uniform subsample to cap cost
        idx = np.linspace(0, len(f224) - 1, a.max_frames).astype(int)
        f224, th128, tpos = f224[idx], th128[idx], tpos[idx]
    print(f"[load] {len(f224)} frames; thumb {th128.shape}", flush=True)

    setup_mpl()
    import matplotlib.pyplot as plt
    K = a.k
    fig, axes = plt.subplots(len(a.encoders), K, figsize=(K * 1.25, len(a.encoders) * 1.45))
    if len(a.encoders) == 1:
        axes = axes[None, :]

    for r, name in enumerate(a.encoders):
        t0 = time.time()
        enc = load_encoder(name)
        din = enc.spec.dim
        grids = enc.encode_grid(f224)                           # (N, din, 16, 16) fp16
        pooled = L2(grids.reshape(len(grids), din, -1).mean(2).astype(np.float32))
        _cen, lab = cpu_kmeans(pooled, K)
        # per-cluster mean grid + phase position, order columns early->late
        order = sorted(range(K), key=lambda c: tpos[lab == c].mean() if (lab == c).any() else 9)
        cen_grid = np.stack([grids[lab == c].mean(0) if (lab == c).any()
                             else np.zeros((din, 16, 16), np.float16) for c in order])
        decode = train_dec(grids, th128, din, dec="small", epochs=a.epochs)
        dec_imgs = decode(cen_grid.astype(np.float32))          # (K, 128,128,3) uint8
        del grids
        import torch; torch.cuda.empty_cache()
        enc.unload()
        for c in range(K):
            ax = axes[r, c]; ax.imshow(dec_imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{name}\n(d={din})", fontsize=8)
            if r == 0:
                ax.set_title(f"phase {c+1}", fontsize=7)
        print(f"[{name}] din={din} done {time.time()-t0:.0f}s", flush=True)

    fig.suptitle(f"各编码器 簇中心解码对比 — {a.ds}, K={K}, {len(f224)}帧 (列按任务相位排序)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = viz_dir("encoders") / "enc_centroid_decode_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
