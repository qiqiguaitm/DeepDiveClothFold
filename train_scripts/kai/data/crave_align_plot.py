"""Plot baseline (production) vs best (Viterbi-over-milestones) value/assignment curves for 1-2 long eps.
Run: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_align_plot.py <ds>
"""
import sys, os
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import crave_align_analyze as A

OUT = Path("/vePFS/tim/workspace/deepdive_kai0/temp/crave_align")


def main(ds):
    F, E, Tv, thumb, cl = A.run(ds, wproprio=1.0)
    eps = sorted(set(E.tolist()))
    # two longest eps
    lens = {e: int((E == e).sum()) for e in eps}
    long2 = sorted(lens, key=lambda e: -lens[e])[:2]
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    for col, e in enumerate(long2):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        Fq = F[fi]; Tt = Tv[fi]; nn = len(Fq)
        vb, msb = A.readout_production(Fq, cl)
        vv, msv = A.readout_viterbi_ms(Fq, cl, lam=8.0)
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
