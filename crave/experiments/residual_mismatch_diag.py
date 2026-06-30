"""Deep-dive the RESIDUAL XVLA mis-match/value-jumps after proprio w=0.2.
Hypothesis: DINOv3-H has low visual discriminability on XVLA top-down cloth → emission is ambiguous
(small margin between nearest clusters) → adjacent frames flip → mis-id + value jumps.
Measures emission margin, raw flip rate, cluster separation; compares XVLA vs coffee. w=0.2 features.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
from milestone_select import build_milestones_uniform

ap = argparse.ArgumentParser(); ap.add_argument("--w", type=float, default=0.2); ap.add_argument("--vep", type=int, default=7)
a = ap.parse_args()
VD = 1280
for ds in ["xvla", "coffee"]:
    z = np.load(REPO / f"temp/xreb_cache_{ds}.npz", allow_pickle=True)
    F = z["F"].astype(np.float32).copy(); F[:, VD:] *= a.w; E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
    cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps)); C = cen[order]
    # emission ambiguity: gap between nearest and 2nd-nearest milestone (full-F and vision-only)
    D = np.linalg.norm(F[:, None] - C[None], axis=2); Ds = np.sort(D, 1)
    margin = Ds[:, 1] - Ds[:, 0]; nmargin = margin / (Ds[:, 0] + 1e-6)
    Dv = np.linalg.norm(F[:, None, :VD] - C[None, :, :VD], axis=2); Dvs = np.sort(Dv, 1)
    vmargin = Dvs[:, 1] - Dvs[:, 0]
    # cluster separation (silhouette-ish): intra dist vs nearest-other-centroid dist
    am = D.argmin(1); intra = np.array([np.linalg.norm(F[am == m] - C[m], axis=1).mean() if (am == m).any() else np.nan for m in range(M)])
    cc = np.linalg.norm(C[:, None] - C[None], axis=2); np.fill_diagonal(cc, np.inf); inter = cc.min(1)
    sep = np.nanmean(intra / inter)   # >1 means clusters overlap (poorly separated)
    # raw emission flip rate (per-frame argmin changes, before any Viterbi)
    flips = []
    for e in eps:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        if len(fi) < 5: continue
        a_ = D[fi].argmin(1); flips.append(np.mean(np.diff(a_) != 0))
    print(f"[{ds} w={a.w}] M={M} | emission margin(中位)={np.median(margin):.3f} 归一margin={np.median(nmargin):.2f} | 视觉margin={np.median(vmargin):.3f} | 簇内/簇间分离比={sep:.2f}(>1差) | 原始逐帧翻转率={np.mean(flips):.0%}", flush=True)

# value-jump attribution on the xvla video episode
z = np.load(REPO / "temp/xreb_cache_xvla.npz", allow_pickle=True)
F = z["F"].astype(np.float32).copy(); F[:, VD:] *= a.w; E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
if a.vep in set(E.tolist()):
    cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps)); C = cen[order]
    fi = np.where(E == a.vep)[0]; fi = fi[np.argsort(Tv[fi])]; Fq = F[fi]
    D = np.linalg.norm(Fq[:, None] - C[None], axis=2); raw = D.argmin(1)
    jumps = np.where(np.abs(np.diff(Pord[raw])) > 0.1)[0]
    print(f"\n[xvla ep{a.vep}] 原始 emission 读出: {len(fi)}帧, |Δvalue|>0.1 的跳变 {len(jumps)} 处 ({len(jumps)/len(fi):.0%}/帧)", flush=True)
    print(f"  这些跳变里,前后两帧 emission 距离差(margin)中位={np.median([D[j].min() for j in jumps]):.3f} → margin 小=模糊导致翻", flush=True)
