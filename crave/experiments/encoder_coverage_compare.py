"""3-way cluster-coverage distribution comparison: DINOv2-large vs DINOv3-H+ vs DINOv3-7B.

Coverage(cluster) = fraction of episodes the cluster appears in — the recurrence signal CRAVE
selects milestones from. Lower-coverage mass = more fragmented clustering (e.g. by appearance).
Uses the frame-aligned FULL kai0_base @3Hz pooled features already on disk (no re-encode).

Run: CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python \
       crave/experiments/encoder_coverage_compare.py
Out: crave/docs/visualization/encoders/enc_coverage_compare_3way.png (+ .json)
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.clustering.kmeans import gpu_kmeans                    # noqa: E402
from crave.config import REPO, viz_dir                            # noqa: E402
from crave.utils import L2, otsu                                  # noqa: E402

# tag -> (shard_dir, dim, index_path, color)
SRC = {
    "dinov2-large":   (REPO / "temp/crave_full/dino", 1024, REPO / "temp/crave_full/index_dino.npz", "#1f77b4"),
    "dinov3-h":       (REPO / "temp/crave_full_dinov3h", 1280, REPO / "temp/crave_full_dinov3h/index.npz", "#2ca02c"),
    "dinov3-7b-int8": (REPO / "temp/crave_full_dinov37bint8", 4096, REPO / "temp/crave_full_dinov37bint8/index.npz", "#d62728"),
}


def coverage(tag):
    shard_dir, dim, idxp, _ = SRC[tag]
    idx = np.load(idxp); E, N = idx["E"], int(idx["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; Fv = L2(feat[vi].astype(np.float32)); Ev = E[vi]
    ne = len(set(Ev.tolist()))
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 64, 320))
    _cen, lab = gpu_kmeans(Fv, K0)
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    return cov, K0, ne, len(vi)


def main():
    t0 = time.time()
    res = {}
    for tag in SRC:
        cov, K0, ne, nf = coverage(tag)
        res[tag] = {"cov": cov, "K0": K0, "ne": ne, "nf": nf, "tau": float(otsu(cov))}
        print(f"[{tag}] K0={K0} ne={ne} median-cov={np.median(cov):.3f} "
              f"τ_cov={res[tag]['tau']:.3f} frac<τ={np.mean(cov < res[tag]['tau']):.3f} ({time.time()-t0:.0f}s)", flush=True)

    from crave.render import setup_mpl
    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0, max(np.percentile(res[t]["cov"], 99) for t in res), 36)
    for tag in SRC:
        c = SRC[tag][3]; cov = res[tag]["cov"]
        ax[0].hist(cov, bins=bins, alpha=0.45, color=c,
                   label=f"{tag}  med={np.median(cov):.3f} frac<τ={np.mean(cov<res[tag]['tau']):.0%}")
        ax[0].axvline(res[tag]["tau"], color=c, ls="--", lw=1.2)
        sc = np.sort(cov); ax[1].plot(sc, np.linspace(0, 1, len(sc)), color=c, lw=2, label=tag)
    ax[0].set_title("簇覆盖率分布 (虚线=各自 Otsu τ_cov)\n左移/更高峰=更多低覆盖簇=聚类更碎")
    ax[0].set_xlabel("coverage = 该簇出现的 episode 占比"); ax[0].set_ylabel("#clusters"); ax[0].legend(fontsize=8)
    ax[1].set_title("覆盖率 CDF\n左上更靠=低覆盖簇更多"); ax[1].set_xlabel("coverage"); ax[1].set_ylabel("CDF"); ax[1].legend(fontsize=9)
    fig.suptitle(f"簇覆盖率分布对比 — FULL kai0_base @3Hz ({res['dinov2-large']['ne']}ep) — DINOv2-large vs DINOv3-H+ vs DINOv3-7B", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = viz_dir("encoders") / "enc_coverage_compare_3way.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    summary = {tag: {"K0": res[tag]["K0"], "median_cov": float(np.median(res[tag]["cov"])),
                     "tau_cov": res[tag]["tau"], "frac_below_tau": float(np.mean(res[tag]["cov"] < res[tag]["tau"])),
                     "mean_cov": float(np.mean(res[tag]["cov"]))} for tag in res}
    json.dump(summary, open(viz_dir("encoders") / "enc_coverage_compare_3way.json", "w"), indent=2)
    print(f"SAVED {out}\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
