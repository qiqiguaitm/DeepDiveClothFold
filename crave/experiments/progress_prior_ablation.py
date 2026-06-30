"""Ablate the DUAL progress prior WITH the transition prior (簇间转移概率, β=0.4) held ON +
confidence hold ON. Question: given the inter-cluster transition prior, is the progress prior still needed?
  A full        : γ=0.8 (低) + γ_fwd=1.3 (高)
  B no-progress : γ=0     + γ_fwd=0       (只剩 emission + 转移先验 + 置信门控)
  C low-only    : γ=0.8   + γ_fwd=0       (只修循环假跌)
  D far-only    : γ=0     + γ_fwd=1.3     (只修前向误跳)
Cache-based. Run: PY crave/experiments/progress_prior_ablation.py --ds xvla --native-fps 30
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
    pen = build_pen(counts, Pord, 1.0, 0.4, 1.0)   # transition prior ON (簇间转移概率 β=0.4)
    dM = np.linalg.norm(F[:, None] - C[None], axis=2).min(1); tau_lo, tau_hi = np.percentile(dM, 50), np.percentile(dM, 82)
    mw = max(5, int(round(5 * a.native_fps / 3))) | 1; sub = eps[:a.stats_n]

    def readout(Fq, g, gf):
        base = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq); tx = np.arange(nn) / nn
        em = base.copy(); dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1)
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        if g > 0: em = em + g * np.maximum(0.0, tx[:, None] - Pord[None, :] - 0.15)
        if gf > 0: em = em + gf * np.maximum(0.0, Pord[None, :] - tx[:, None] - 0.25)
        ms = viterbi_pen(em, pen); dmin = base.min(1); conf = np.clip((tau_hi - dmin) / (tau_hi - tau_lo + 1e-6), 0, 1)
        for t in range(1, nn):
            if conf[t] < 0.5: ms[t] = ms[t - 1]
        return ms

    cfgs = {"A full (γ0.8 + fwd1.3)": (0.8, 1.3), "B no-progress (0+0)": (0.0, 0.0),
            "C low-only (γ0.8)": (0.8, 0.0), "D far-only (fwd1.3)": (0.0, 1.3)}
    print(f"[{a.ds}] M={M} | 转移先验(簇间转移概率 β0.4)+置信门控 全程 ON,消融进度先验:", flush=True)
    for name, (g, gf) in cfgs.items():
        J, Cr, nd = [], [], 0
        for e in sub:
            fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
            if len(fi) < 5: continue
            ms = readout(F[fi], g, gf); v = smooth_monotone(med(Pord[ms], mw), fps=a.native_fps); tq = Tv[fi]
            J.append(int(np.sum(np.diff(ms) != 0))); nd += int(np.any((tq > 0.4) & (v < 0.05)))
            Cr.append(np.corrcoef(v, tq)[0, 1] if v.std() > 1e-6 else np.nan)
        Cr = np.array(Cr); ok = np.isfinite(Cr)
        print(f"  {name:24s}: corr={Cr[ok].mean():.4f} median={np.median(Cr[ok]):.4f} %>=0.7={np.mean(Cr[ok]>=0.7):.0%} jumps/ep={np.mean(J):.0f} 跌零ep={nd}", flush=True)


if __name__ == "__main__":
    main()
