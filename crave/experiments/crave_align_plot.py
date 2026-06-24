"""Plot baseline (production) vs best (Viterbi-over-milestones) value/assignment curves for 1-2 long eps.
Run: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_align_plot.py <ds>
"""
import sys, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

from crave.clustering import build_clusters
from crave.config import REPO
from crave.render import setup_mpl
from crave.value import readout_production, readout_viterbi_ms

plt = setup_mpl()
OUT = REPO / "temp/crave_align"


def run(ds, wproprio=1.0, tempctx=False):
    """wproprio scales proprio block; tempctx appends short-window mean+delta of pooled img feat.
    Replica of crave_align_analyze.run (cache read + production proprio-norm + F build)."""
    z = np.load(OUT / f"{ds}_cache.npz")
    img, Pm, E, Tv, thumb = z["img"], z["state"], z["ep"], z["tpos"], z["thumb"]
    ne = len(np.unique(E))
    # normalize proprio as production
    PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
    Pn = (Pm - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    blocks = [img, Pn * wproprio]
    if tempctx:
        # temporal context: per-episode short-window mean and delta of pooled img feature
        ctx_mean = np.zeros_like(img); ctx_d = np.zeros_like(img)
        for e in np.unique(E):
            fi = np.where(E == e)[0]; x = img[fi]
            k = 3; cm = np.zeros_like(x)
            for j in range(len(x)):
                cm[j] = x[max(0, j - k):j + 1].mean(0)
            d = np.zeros_like(x); d[1:] = x[1:] - x[:-1]
            ctx_mean[fi] = cm; ctx_d[fi] = d
        # small weight so it adds temporal smoothness without dominating
        blocks += [ctx_mean * 0.3, ctx_d * 0.3]
    F = np.concatenate(blocks, 1).astype(np.float32)
    cl = build_clusters(F, E, Tv, ne)
    return F, E, Tv, thumb, cl


def main(ds):
    F, E, Tv, thumb, cl = run(ds, wproprio=1.0)
    eps = sorted(set(E.tolist()))
    # two longest eps
    lens = {e: int((E == e).sum()) for e in eps}
    long2 = sorted(lens, key=lambda e: -lens[e])[:2]
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    for col, e in enumerate(long2):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        Fq = F[fi]; Tt = Tv[fi]; nn = len(Fq)
        vb, msb = readout_production(Fq, cl)
        vv, msv = readout_viterbi_ms(Fq, cl, lam=8.0)
        ax = axes[0, col]
        ax.plot(Tt, vb, color="#d62728", lw=1.6, label="baseline (bins->argmin)")
        ax.plot(Tt, vv, color="#1a7f37", lw=1.6, label="viterbi-over-ms")
        ax.plot(Tt, Tt, color="0.6", lw=0.8, ls="--", label="norm time (ref)")
        ax.set_ylim(-.02, 1.02); ax.set_title(f"[{ds}] ep{e} (len {nn}) value"); ax.legend(fontsize=7)
        ax.set_ylabel("forward value")
        ax2 = axes[1, col]
        ax2.step(range(nn), msb, where="post", color="#d62728", lw=1.2, alpha=.8, label="baseline ms")
        ax2.step(range(nn), msv, where="post", color="#1a7f37", lw=1.2, label="viterbi ms")
        ax2.set_ylabel("milestone"); ax2.set_xlabel("frame(3Hz)"); ax2.legend(fontsize=7)
    fig.suptitle(f"[{ds}] alignment: production bins-indirection (red) vs Viterbi-over-milestones (green)")
    fig.tight_layout(); fig.savefig(OUT / f"{ds}_align_compare.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[{ds}] saved {OUT/f'{ds}_align_compare.png'}  long2={long2}")


if __name__ == "__main__":
    main(sys.argv[1])
