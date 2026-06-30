"""Ablate the two transition-cost terms in build_pen, with the FULL current readout
(emission + progress prior + far-ahead cap + confidence hold) held fixed:
  A full      : lam_geo=1.0 beta=0.4   (geo + empirical)
  B no-geo    : lam_geo=0.0 beta=0.4   (empirical only)  <- the question: is geo still needed?
  C no-emp    : lam_geo=1.0 beta=0.0   (geo only)
  D neither   : lam_geo=0.0 beta=0.0   (just back_barrier + emission)
Cache-based → fast. Run: PY crave/experiments/transition_term_ablation.py --ds xvla --native-fps 30
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.utils import med, smooth_monotone
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
from milestone_select import build_milestones_uniform


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="xvla"); ap.add_argument("--native-fps", type=float, default=30.0)
    ap.add_argument("--stats-n", type=int, default=60)
    a = ap.parse_args()
    z = np.load(REPO / f"temp/xreb_cache_{a.ds}.npz", allow_pickle=True)
    F = z["F"].astype(np.float32); E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
    cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps), max_gap=0.12); C = cen[order]
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
    for e in eps:
        seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
    # fraction of forward milestone pairs NEVER co-observed (where empirical prior = Laplace floor only)
    fwd = Pord[None, :] > Pord[:, None]; unobs = (counts == 0) & fwd
    print(f"[{a.ds}] M={M}  forward pairs never co-observed: {unobs.sum()}/{fwd.sum()} ({100*unobs.sum()/max(fwd.sum(),1):.0f}%) → these rely on geo fallback", flush=True)
    dM = np.linalg.norm(F[:, None] - C[None], axis=2).min(1); tau_lo, tau_hi = np.percentile(dM, 50), np.percentile(dM, 82)
    mw = max(5, int(round(5 * a.native_fps / 3))) | 1; sub = eps[:a.stats_n]

    def readout(Fq, pen):
        base = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq); tx = np.arange(nn) / nn
        em = base.copy(); dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1)
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        em = em + 0.8 * np.maximum(0.0, tx[:, None] - Pord[None, :] - 0.15)
        em = em + 1.3 * np.maximum(0.0, Pord[None, :] - tx[:, None] - 0.25)
        ms = viterbi_pen(em, pen); dmin = base.min(1); conf = np.clip((tau_hi - dmin) / (tau_hi - tau_lo + 1e-6), 0, 1)
        for t in range(1, nn):
            if conf[t] < 0.5: ms[t] = ms[t - 1]
        return ms

    configs = {"A full (geo+emp)": (1.0, 0.4), "B no-geo (emp only)": (0.0, 0.4),
               "C no-emp (geo only)": (1.0, 0.0), "D neither": (0.0, 0.0)}
    for name, (lg, bt) in configs.items():
        pen = build_pen(counts, Pord, lg, bt, 1.0); J, Cr = [], []
        for e in sub:
            fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
            if len(fi) < 5: continue
            ms = readout(F[fi], pen); v = smooth_monotone(med(Pord[ms], mw), fps=a.native_fps); tq = Tv[fi]
            J.append(int(np.sum(np.diff(ms) != 0))); Cr.append(np.corrcoef(v, tq)[0, 1] if v.std() > 1e-6 else np.nan)
        Cr = np.array(Cr); ok = np.isfinite(Cr)
        print(f"  {name:22s}: corr={Cr[ok].mean():.4f} median={np.median(Cr[ok]):.4f} %>=0.7={np.mean(Cr[ok]>=0.7):.0%} jumps/ep={np.mean(J):.0f}", flush=True)


if __name__ == "__main__":
    main()
