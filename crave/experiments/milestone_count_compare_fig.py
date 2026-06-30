"""Combined comparison: milestone-count ↔ coverage tradeoff curve + the centroid-decode
figure at several milestone counts. Answers "how does the number of milestones affect
coverage and the centroid-decode picture" on FULL kai0_base @3Hz (DINOv3-H+).

Run after milestone_count_sweep.py and the K=24/64/160/318 full-selection decodes exist.
Out: crave/docs/visualization/encoders/enc_milestone_count_vs_coverage_decode.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO, viz_dir

SWEEP = REPO / "temp/crave_full_dinov3h/milestone_count_sweep.json"
VDIR = REPO / "crave/docs/visualization/encoders"
# (K used, decode png) for the strips to show, low→high milestone count
DECODES = [(24, "enc_full_dinov3h_kai0_centroid_decode_k24.png"),
           (64, "enc_full_dinov3h_kai0_centroid_decode_k64.png"),
           (160, "enc_full_dinov3h_kai0_centroid_decode_k160.png"),
           (318, "enc_full_dinov3h_kai0_centroid_decode.png")]


def main():
    rows = json.load(open(SWEEP)); byK = {r["K"]: r for r in rows}
    M = np.array([r["M"] for r in rows]); med = np.array([r["cov_med"] for r in rows])
    lo = np.array([r["cov_min"] for r in rows]); hi = np.array([r["cov_max"] for r in rows])
    Ks = np.array([r["K"] for r in rows]); o = np.argsort(M)

    import matplotlib.image as mpimg
    from crave.render import setup_mpl
    plt = setup_mpl()
    strips = [(k, mpimg.imread(VDIR / p), byK.get(k)) for k, p in DECODES if (VDIR / p).exists()]
    nS = len(strips)
    fig = plt.figure(figsize=(13, 3.2 + 1.7 * nS))
    gs = fig.add_gridspec(1 + nS, 1, height_ratios=[2.6] + [1.6] * nS)

    ax = fig.add_subplot(gs[0])
    ax.fill_between(M[o], lo[o], hi[o], color="#1a7f37", alpha=0.15, label="milestone cov [min,max]")
    ax.plot(M[o], med[o], "-o", color="#1a7f37", lw=2, label="milestone cov (median)")
    for m, c, k in zip(M, med, Ks):
        ax.annotate(f"K={k}", (m, c), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    for k, _img, r in strips:
        if r: ax.axvline(r["M"], color="#d62728", ls=":", lw=1)
    ax.set_xlabel("milestone 数目 M"); ax.set_ylabel("milestone 覆盖率"); ax.set_ylim(0, 1)
    ax.set_title("① milestone 数目 ↑ → 每个 milestone 覆盖率 ↓ (红虚线=下方解码图所用 M)")
    ax.legend(fontsize=8)

    for i, (k, img, r) in enumerate(strips):
        axi = fig.add_subplot(gs[1 + i]); axi.imshow(img); axi.axis("off")
        lab = f"M={r['M']}  (K={k}, cov {r['cov_min']:.2f}–{r['cov_max']:.2f})" if r else f"K={k}"
        axi.set_title(f"② 簇中心解码 @ {lab}", fontsize=9, loc="left")

    fig.suptitle("milestone 数目对 覆盖率 与 簇中心解码图 的影响 — FULL kai0_base @3Hz, DINOv3-H+", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = viz_dir("encoders") / "enc_milestone_count_vs_coverage_decode.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"SAVED {out}  (strips: {[(k, r['M'] if r else '?') for k, _i, r in strips]})", flush=True)


if __name__ == "__main__":
    main()
