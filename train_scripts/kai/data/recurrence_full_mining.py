#!/usr/bin/env python
"""全量重复度挖掘(CPU): 消费集群提取的 feat_cache(per-ep npz)→ MiniBatchKMeans →
first-visit 覆盖率 → milestone(覆盖率局部峰值, betweenness 先例)→ 与 50-ep V0 对比稳定性。
用法: python recurrence_full_mining.py --cache temp/tcc_kai0/feat_cache --out temp/full_mining_kai0 [--k 64]
"""
import argparse, re
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=64)
ap.add_argument("--top-m", type=int, default=14)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cache, out = Path(args.cache), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

files = sorted(cache.glob("ep*.npz"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
print(f"[mine] {len(files)} episodes from {cache}")
F, E, T = [], [], []
for p in files:
    f = np.load(p)["f"]
    ep = int(re.findall(r"\d+", p.stem)[0])
    n = len(f)
    if n < 5:
        continue
    F.append(f); E.append(np.full(n, ep)); T.append(np.arange(n) / max(1, n - 1))
F = np.concatenate(F); E = np.concatenate(E); T = np.concatenate(T)
print(f"[mine] frames {F.shape}")

from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=4096, n_init=5).fit(F)
lab = km.labels_
n_ep = len(set(E.tolist()))
cov = np.array([len(set(E[lab == c].tolist())) / n_ep for c in range(args.k)])
tpos = np.array([T[lab == c].mean() for c in range(args.k)])
dom = np.array([np.bincount(E[lab == c]).max() / (lab == c).sum() for c in range(args.k)])

# milestone = 覆盖率沿时间的局部峰值 (排序后相对邻居)
order = np.argsort(tpos)
cov_o, ids_o = cov[order], order
peaks = []
for i in range(len(ids_o)):
    lo, hi = max(0, i - 2), min(len(ids_o), i + 3)
    if cov_o[i] == cov_o[lo:hi].max() and cov_o[i] >= np.median(cov):
        peaks.append(ids_o[i])
peaks = sorted(set(peaks), key=lambda c: tpos[c])[: args.top_m]
print(f"[mine] {n_ep} eps; coverage med={np.median(cov):.0%} max={cov.max():.0%}")
print("[mine] milestones(local peaks):", [(int(c), f"{cov[c]:.0%}", f"t={tpos[c]:.2f}", f"dom={dom[c]:.0%}") for c in peaks])

np.savez_compressed(out / "mining.npz", centroids=km.cluster_centers_, cov=cov, tpos=tpos,
                    dom=dom, milestones=np.array(peaks))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(tpos[order], cov[order], "o-", ms=4, alpha=.8)
for c in peaks:
    ax.plot(tpos[c], cov[c], "r*", ms=14)
    ax.annotate(str(c), (tpos[c], cov[c]), fontsize=8, color="r")
ax.set_xlabel("cluster mean time"); ax.set_ylabel("episode coverage (first-visit)")
ax.set_title(f"FULL-data mining: {n_ep} eps, k={args.k} — red★ = milestone (local peak)")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "full_coverage_curve.png", dpi=120)
print("plot ->", out / "full_coverage_curve.png")
