"""Diagnose XVLA mis-matching: is the milestone assignment driven by proprio (arm pose) rather than
vision? F = [vis(1280, L2) ‖ prop(28, L2)] → in ‖F−C‖² the proprio half weighs EQUAL to the whole image.
Compare full-F vs vision-only nearest-milestone on the video episode."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.utils import L2
from milestone_select import build_milestones_uniform

ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="xvla"); ap.add_argument("--ep", type=int, default=10)
a = ap.parse_args()
z = np.load(REPO / f"temp/xreb_cache_{a.ds}.npz", allow_pickle=True)
F = z["F"].astype(np.float32); E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps), max_gap=0.12); C = cen[order]
VD = 1280  # visual dims
vis, prop = F[:, :VD], F[:, VD:]; Cv, Cp = C[:, :VD], C[:, VD:]

# global: average distance contribution of vis vs prop (each is unit-norm → both ~comparable magnitude)
print(f"[{a.ds}] feature split: vis={VD}d prop={F.shape[1]-VD}d ; per-vector norms vis~{np.linalg.norm(vis,axis=1).mean():.2f} prop~{np.linalg.norm(prop,axis=1).mean():.2f}", flush=True)

fi = np.where(E == a.ep)[0]; fi = fi[np.argsort(Tv[fi])]; Fq = F[fi]
dfull = np.linalg.norm(Fq[:, None] - C[None], axis=2)            # full-F distance
dvis = np.linalg.norm(Fq[:, None, :VD] - Cv[None], axis=2)       # vision-only distance
dprop = np.linalg.norm(Fq[:, None, VD:] - Cp[None], axis=2)      # proprio-only distance
am_full = dfull.argmin(1); am_vis = dvis.argmin(1); am_prop = dprop.argmin(1)

# how often does full-F assignment differ from vision-only? and how far apart in value?
diff = am_full != am_vis
print(f"[ep{a.ep}] frames={len(fi)}", flush=True)
print(f"  full-F vs vision-only 不同档比例: {diff.mean():.0%}  | 当不同时 |ΔPord| 中位={np.median(np.abs(Pord[am_full[diff]]-Pord[am_vis[diff]])) if diff.any() else 0:.2f}", flush=True)
print(f"  full-F vs proprio-only 相同档比例: {(am_full==am_prop).mean():.0%}  (越高=assignment 越被 proprio 主导)", flush=True)
# for the FULL-assigned milestone: how far is the query in vision vs proprio (squared, comparable)
i = np.arange(len(fi))
dv2 = (dvis[i, am_full]) ** 2; dp2 = (dprop[i, am_full]) ** 2
print(f"  被分到的档:平均 视觉距离² ={dv2.mean():.3f}  proprio距离² ={dp2.mean():.3f}  → proprio 占比 {dp2.mean()/(dv2.mean()+dp2.mean()):.0%}", flush=True)
# fraction of frames where vision says the assigned cluster is a BAD visual match (visual dist in top-25% globally) yet got assigned
gv = np.linalg.norm(F[:, None, :VD] - Cv[None], axis=2).min(1)   # global vision nearest-dist
thr = np.percentile(gv, 75)
badvis = dv2[i] ** 0.5 > thr
print(f"  被分到的档里「视觉其实很远(>全局p75={thr:.2f})」的帧占比: {badvis.mean():.0%}  (这些就是'视觉不像却被归同档')", flush=True)
