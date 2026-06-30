"""Inspect one milestone's intra-cluster value(=normalized progress T) distribution.

Reproduces the FULL-kai0 K=24 cov>=mean clustering (deterministic) and, for milestone --m,
shows the distribution of frame progress T of its members + sample real frames at T quantiles
+ the decoded centroid. Tells whether a milestone's value looks "too small" because it's
genuinely early, bimodal/contaminated, or a recurring (cyclic) state.

Run: CUDA_VISIBLE_DEVICES=0 PY crave/experiments/inspect_milestone.py --encoder dinov3-h --m 5
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO, resolve_dataset, viz_dir
from crave.data import kai0
from crave.utils import L2

SRC = {"dinov2-large": (REPO / "temp/crave_full/dino", 1024, REPO / "temp/crave_full/index_dino.npz"),
       "dinov3-h": (REPO / "temp/crave_full_dinov3h", 1280, REPO / "temp/crave_full_dinov3h/index.npz"),
       "dinov3-7b-int8": (REPO / "temp/crave_full_dinov37bint8", 4096, REPO / "temp/crave_full_dinov37bint8/index.npz")}
cfg = resolve_dataset("kai0_base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov3-h")
    ap.add_argument("--k0", type=int, default=24)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--selection", choices=["full", "covmean"], default="full")
    a = ap.parse_args()
    sd, dim, idxp = SRC[a.encoder]
    idx = np.load(idxp); E, FR, T, N = idx["E"], idx["FR"], idx["T"], int(idx["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(sd / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; F = L2(feat[vi].astype(np.float32)); Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))

    from sklearn.cluster import MiniBatchKMeans
    K0 = a.k0
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx])
    cen = km.cluster_centers_; lab = km.predict(F)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    from crave.utils import otsu
    if a.selection == "covmean":
        sel = sorted([c for c in range(K0) if cov[c] >= cov.mean()], key=lambda c: tpos[c])
    else:  # full: coverage-Otsu + temporal-purity + progress dedup (the validated CRAVE selection)
        tau_cov = otsu(cov); vt = tstd[tstd < 9]; tau_pur = float(np.percentile(vt, 60)) if len(vt) else 9.0
        cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
        gap = max(0.006, 0.5 / max(len(cand), 1)); sel = []
        for c in cand:
            if not sel or tpos[c] - tpos[sel[-1]] >= gap: sel.append(c)
            elif cov[c] > cov[sel[-1]]: sel[-1] = c
    print(f"[{a.encoder}] K0={K0} sel={a.selection} -> {len(sel)} milestones", flush=True)
    for i, c in enumerate(sel):
        loc = lab == c; t = Tv[loc]
        print(f"  m{i}: cluster#{c} cov={cov[c]:.2f} tpos={tpos[c]:.3f} n={loc.sum()} "
              f"T[min/q25/med/q75/max]={t.min():.2f}/{np.percentile(t,25):.2f}/{np.median(t):.2f}/{np.percentile(t,75):.2f}/{t.max():.2f}", flush=True)

    c = sel[a.m]; loc = np.where(lab == c)[0]; t = Tv[loc]
    print(f"\n=== m{a.m} = cluster#{c}: cov={cov[c]:.2f} tpos={tpos[c]:.3f} n={len(loc)} eps={len(set(Ev[loc].tolist()))} ===", flush=True)

    # sample real frames at T quantiles within the cluster
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    picks = [loc[np.argmin(np.abs(t - np.quantile(t, q)))] for q in qs]
    gidx = vi[picks]
    imgs, ok = kai0.decode_images(cfg, gidx, E, FR)

    from crave.render import setup_mpl
    plt = setup_mpl()
    fig = plt.figure(figsize=(13, 4.2)); gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1])
    axh = fig.add_subplot(gs[0, :]); axh.hist(t, bins=40, color="#1a7f37")
    axh.axvline(tpos[c], color="r", lw=2, label=f"tpos(mean)={tpos[c]:.3f}")
    axh.axvline(np.median(t), color="orange", lw=1.5, ls="--", label=f"median={np.median(t):.3f}")
    axh.set_title(f"{a.encoder} m{a.m} (cluster#{c}, cov={cov[c]:.2f}, n={len(loc)}) — 簇内帧进度 T 分布"); axh.set_xlabel("T = normalized progress in episode"); axh.set_xlim(0, 1); axh.legend(fontsize=8)
    for j, q in enumerate(qs):
        ax = fig.add_subplot(gs[1, j]); ax.axis("off")
        if ok[j]: ax.imshow(imgs[j]); ax.set_title(f"T≈{np.quantile(t,q):.2f}", fontsize=8)
    fig.tight_layout()
    out = viz_dir("encoders") / f"inspect_{a.encoder.replace('-','')}_m{a.m}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
