"""Find a good proprio weight w for F = [vis(1280,L2) ⊕ w·prop(28,L2)].
For each w: rebuild F_w, RE-CLUSTER, run full readout, measure value quality (corr vs time) and
visual-match quality (median visual dist of each frame to its assigned milestone's visual centroid,
+ vision-agreement = how often assignment = vision-only nearest). Cache-based.
Run: PY crave/experiments/proprio_weight_sweep.py --ds xvla --native-fps 30
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

ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="xvla"); ap.add_argument("--native-fps", type=float, default=30.0)
ap.add_argument("--stats-n", type=int, default=60); ap.add_argument("--ws", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2, 0.1, 0.0])
a = ap.parse_args()
VD = 1280
z = np.load(REPO / f"temp/xreb_cache_{a.ds}.npz", allow_pickle=True)
F0 = z["F"].astype(np.float32); E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist())); ne = len(eps)
sub = eps[:a.stats_n]
# fixed visual-quality threshold: p75 of each frame's distance to its vision-only nearest milestone (w=0 clustering)
from sklearn.cluster import KMeans
mw = max(5, int(round(5 * a.native_fps / 3))) | 1
print(f"[{a.ds}] M·corr·%>=0.7·jumps/ep · 视觉中位距离(↓好) · 视觉同档比例(↑好)  (proprio 权重扫描)", flush=True)
for w in a.ws:
    F = F0.copy(); F[:, VD:] = F0[:, VD:] * w
    cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, ne); C = cen[order]
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
    for e in eps:
        seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
    pen = build_pen(counts, Pord, 1.0, 0.4, 1.0)
    dM = np.linalg.norm(F[:, None] - C[None], axis=2).min(1); tlo, thi = np.percentile(dM, 50), np.percentile(dM, 82)
    Cv = C[:, :VD]
    J, Cr, vdist, vagree = [], [], [], []
    for e in sub:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        if len(fi) < 5: continue
        Fq = F[fi]; base = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq); tx = np.arange(nn) / nn
        em = base.copy(); dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1)
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        em = em + 0.8 * np.maximum(0.0, tx[:, None] - Pord[None, :] - 0.15) + 1.3 * np.maximum(0.0, Pord[None, :] - tx[:, None] - 0.25)
        ms = viterbi_pen(em, pen); conf = np.clip((thi - base.min(1)) / (thi - tlo + 1e-6), 0, 1)
        for t in range(1, nn):
            if conf[t] < 0.5: ms[t] = ms[t - 1]
        v = smooth_monotone(med(Pord[ms], mw), fps=a.native_fps); tq = Tv[fi]
        J.append(int(np.sum(np.diff(ms) != 0))); Cr.append(np.corrcoef(v, tq)[0, 1] if v.std() > 1e-6 else np.nan)
        visq = Fq[:, :VD]; dvis_all = np.linalg.norm(visq[:, None] - Cv[None], axis=2)
        vdist.append(dvis_all[np.arange(nn), ms])                 # visual dist to ASSIGNED milestone centroid
        vagree.append((dvis_all.argmin(1) == ms).mean())          # assignment == vision-only nearest?
    Cr = np.array(Cr); ok = np.isfinite(Cr); vd = np.concatenate(vdist)
    print(f"  w={w:>4}: M={M:2d} corr={Cr[ok].mean():.3f} %>=0.7={np.mean(Cr[ok]>=0.7):.0%} jumps={np.mean(J):.0f} | 视觉中位={np.median(vd):.3f} 视觉同档={np.mean(vagree):.0%}", flush=True)
