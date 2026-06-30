"""Deep mechanism of XVLA value 突变: at jump frames, is it because several clusters are near-TIED
in distance AND those tied clusters have DISPARATE progress values (visual aliasing — different task
stages that look similar to DINOv3)? Quantify tie-set size & value-spread at jumps vs non-jumps;
also dump the worst jumps with the competing clusters' values + visual similarity.
Run: PY crave/experiments/jump_mechanism_diag.py --ep 7
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from milestone_select import build_milestones_uniform

ap = argparse.ArgumentParser(); ap.add_argument("--ep", type=int, default=7); ap.add_argument("--w", type=float, default=0.2)
ap.add_argument("--tie", type=float, default=1.15); ap.add_argument("--jth", type=float, default=0.15)
a = ap.parse_args()
VD = 1280
z = np.load(REPO / "temp/xreb_cache_xvla.npz", allow_pickle=True)
F = z["F"].astype(np.float32).copy(); F[:, VD:] *= a.w; E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps)); C = cen[order]
fi = np.where(E == a.ep)[0]; fi = fi[np.argsort(Tv[fi])]; Fq = F[fi]; n = len(fi)
D = np.linalg.norm(Fq[:, None] - C[None], axis=2); am = D.argmin(1); val = Pord[am]

# tie-set per frame: clusters within tie× nearest distance; their Pord spread
dmin = D.min(1)
tie_mask = D < (a.tie * dmin[:, None])
tie_n = tie_mask.sum(1)
tie_spread = np.array([Pord[tie_mask[t]].max() - Pord[tie_mask[t]].min() for t in range(n)])

jumps = np.where(np.abs(np.diff(val)) > a.jth)[0]
nonj = np.setdiff1d(np.arange(n - 1), jumps)
print(f"[xvla ep{a.ep} w={a.w}] {n}帧 M={M} | 原始emission value 突变(|Δ|>{a.jth}) {len(jumps)}处 ({len(jumps)/n:.0%})", flush=True)
print(f"  跳变帧: 平均tie-set大小={tie_n[jumps].mean():.1f}  tie内Pord跨度={tie_spread[jumps].mean():.2f}", flush=True)
print(f"  平稳帧: 平均tie-set大小={tie_n[nonj].mean():.1f}  tie内Pord跨度={tie_spread[nonj].mean():.2f}", flush=True)
# at jumps: the before/after clusters — visual distance vs Pord gap
vc = C[:, :VD]
vis_gap, p_gap = [], []
for j in jumps:
    a0, a1 = am[j], am[j + 1]
    vis_gap.append(np.linalg.norm(vc[a0] - vc[a1]))   # visual distance between the two competing milestone centroids
    p_gap.append(abs(Pord[a0] - Pord[a1]))
vis_gap, p_gap = np.array(vis_gap), np.array(p_gap)
# reference: typical visual distance between RANDOM milestone pairs
allpair = np.linalg.norm(vc[:, None] - vc[None], axis=2); iu = np.triu_indices(M, 1)
print(f"  跳变前后两簇: 视觉距离中位={np.median(vis_gap):.3f}(全体簇对中位={np.median(allpair[iu]):.3f}) | Pord跳幅中位={np.median(p_gap):.2f}", flush=True)
frac_close = np.mean(vis_gap < np.percentile(allpair[iu], 25))
print(f"  → 跳变前后两簇'视觉很近(< 全体簇对p25)'占比 {frac_close:.0%}  = 视觉相似但 value 差很多 → 翻一下就突变", flush=True)
# worst 5 jumps
ix = np.argsort(-p_gap)[:5]
print("  最大几处跳变(帧·进度 | from m→to m | Pord跳 | 两簇视觉距离):", flush=True)
for k in ix:
    j = jumps[k]; a0, a1 = am[j], am[j + 1]
    print(f"    t={j} prog={j/n:.2f} | m{a0}(P{Pord[a0]:.2f})→m{a1}(P{Pord[a1]:.2f}) | Δ{p_gap[k]:.2f} | 视觉dist {vis_gap[k]:.3f}", flush=True)
