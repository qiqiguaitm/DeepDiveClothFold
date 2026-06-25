"""Compare coffee milestone clustering at different rates (50Hz native vs 30Hz vs 25Hz).
Short task → over-dense 50Hz frames mis-calibrate build_milestones' rate-dependent heuristics.
Resamples the cached 50Hz features (encoding is per-frame → subsampling cache == encoding subsampled).
Run: PY crave/experiments/coffee_cluster_rate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import med, smooth_monotone
from generalize import build_milestones
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence

NATIVE = 50.0


def resample(F, E, Tv, target_fps):
    """keep target_fps/NATIVE of each episode's frames, evenly."""
    if target_fps >= NATIVE: return F, E, Tv
    keepF, keepE, keepT = [], [], []
    for e in sorted(set(E.tolist())):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        m = max(5, int(round(len(fi) * target_fps / NATIVE)))
        sel = np.round(np.linspace(0, len(fi) - 1, m)).astype(int)
        keepF.append(F[fi[sel]]); keepE.append(E[fi[sel]]); keepT.append(Tv[fi[sel]])
    return np.concatenate(keepF), np.concatenate(keepE), np.concatenate(keepT)


def eval_rate(F, E, Tv, fps, gamma=0.8, margin=0.15):
    from sklearn.cluster import KMeans
    eps = sorted(set(E.tolist())); ne = len(eps)
    cen, lab, order, Pord, M = build_milestones(F, E, Tv, ne); C = cen[order]
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
    for e in eps:
        seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
    pen = build_pen(counts, Pord, 1.0, 0.4, 1.0)
    mw = max(5, int(round(5 * fps / 3))) | 1
    corrs, ndrop, curves = [], 0, {}
    for e in eps:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        if len(fi) < 5: continue
        Fq = F[fi]; em = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq); tx = np.arange(nn) / nn
        dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1)
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        em = em + gamma * np.maximum(0.0, tx[:, None] - Pord[None, :] - margin)
        ms = viterbi_pen(em, pen); v = smooth_monotone(med(Pord[ms], mw), fps=fps); tq = Tv[fi]
        c = float(np.corrcoef(v, tq)[0, 1]) if v.std() > 1e-6 else np.nan
        corrs.append(c); ndrop += int(np.any((tq > 0.4) & (v < 0.05))); curves[e] = (tq, v)
    c = np.array(corrs); ok = np.isfinite(c)
    # value-step granularity: median value gap between adjacent milestones (smaller = finer)
    step = float(np.median(np.diff(np.sort(Pord)))) if M > 1 else 1.0
    return {"M": M, "corr_mean": float(c[ok].mean()), "corr_median": float(np.median(c[ok])),
            "frac_ge_0.7": float(np.mean(c[ok] >= 0.7)), "n_drop": int(ndrop), "value_step": step}, curves, Pord


def main():
    z = np.load(REPO / "temp/xreb_cache_coffee.npz", allow_pickle=True)
    F0, E0, Tv0 = z["F"].astype(np.float32), z["E"], z["Tv"]
    plt = setup_mpl(); fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    rates = [50, 30, 25]
    for i, r in enumerate(rates):
        Fr, Er, Tvr = resample(F0, E0, Tv0, r)
        st, curves, Pord = eval_rate(Fr, Er, Tvr, float(r))
        print(f"[coffee @{r}Hz] M={st['M']} corr_mean={st['corr_mean']:.3f} median={st['corr_median']:.3f} "
              f"%>=0.7={st['frac_ge_0.7']:.0%} drops={st['n_drop']} value_step={st['value_step']:.3f} N={len(Fr)}", flush=True)
        # plot 3 example curves + milestone value levels
        elong = sorted(curves, key=lambda e: -len(curves[e][0]))[:3]
        for e in elong: ax[i].plot(curves[e][0], curves[e][1], lw=1.6, alpha=.85, label=f"ep{e}")
        for p in Pord: ax[i].axhline(p, color="gray", lw=0.4, alpha=.3)
        ax[i].plot([0, 1], [0, 1], "k--", lw=1, alpha=.4); ax[i].set_xlim(0, 1); ax[i].set_ylim(-.02, 1.02)
        ax[i].set_title(f"coffee @{r}Hz 聚簇 · M={st['M']} · corr={st['corr_mean']:.3f} · %≥0.7={st['frac_ge_0.7']:.0%}\nvalue步长={st['value_step']:.3f}(越小越精细) 跌零={st['n_drop']}")
        ax[i].set_xlabel("归一化进度"); ax[i].set_ylabel("CRAVE value"); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
    fig.suptitle("coffee 短程任务:聚簇帧率 vs milestone 精度(灰线=milestone value 档位)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPO / "crave/docs/visualization/cross_dataset/coffee_cluster_rate.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
