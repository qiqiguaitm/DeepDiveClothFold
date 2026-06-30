"""Visual proof of XVLA aliasing: decode the real medoid frames of the value-disparate clusters that
alias (e.g. m0 start P~0 vs m40 end P~1), + a query frame where it flips — show they look similar to
DINOv3 despite very different task value. CPU decode, no GPU."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO, resolve_dataset
from crave.data import load_ep
from milestone_select import build_milestones_uniform

VD = 1280
z = np.load(REPO / "temp/xreb_cache_xvla.npz", allow_pickle=True)
F = z["F"].astype(np.float32).copy(); F[:, VD:] *= 0.2; E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
eplen = dict(zip(z["ek"].tolist(), z["ev"].tolist()))
cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps)); C = cen[order]
cfg = resolve_dataset("xvla")
_cache = {}
def grab(e, frac):
    if e not in _cache:
        f224, _, _, _ = load_ep(cfg, e, strd=1); _cache[e] = f224
    fr = _cache[e]; return fr[min(int(round(frac * (len(fr) - 1))), len(fr) - 1)]

def medoid_frame(milestone_idx):
    c = order[milestone_idx]; loc = np.where(lab == c)[0]
    gi = loc[int(np.argmin(np.linalg.norm(F[loc] - cen[c], axis=1)))]
    e = int(E[gi]); frac = float(Tv[gi]); return grab(e, frac), e, frac

# the start/end aliasing pair + two mid milestones for reference
picks = [0, M // 3, 2 * M // 3, M - 1]
imgs, labels = [], []
for mi in picks:
    img, e, frac = medoid_frame(mi); imgs.append(cv2.resize(img, (200, 200)))
    labels.append(f"m{mi} value={Pord[mi]:.2f}\n(ep{e} @{frac:.2f})")
# query frame at the worst flip in ep7 (t~3413, prog~0.99) — the folded-end frame that aliases to start
fi = np.where(E == 7)[0]; fi = fi[np.argsort(Tv[fi])]
q = grab(7, 0.99); imgs.insert(0, cv2.resize(q, (200, 200))); labels.insert(0, "ep7 query @进度0.99\n(末态折叠帧)")

# visual distances of the query to m0(start) and m40(end) in DINOv3 space
vq = F[fi][int(0.99 * (len(fi) - 1)), :VD]
print(f"M={M}; 末态 query 到 m0(start,P{Pord[0]:.2f}) 视觉dist={np.linalg.norm(vq-C[0,:VD]):.3f} | 到 m{M-1}(end,P{Pord[M-1]:.2f}) 视觉dist={np.linalg.norm(vq-C[M-1,:VD]):.3f}", flush=True)

from crave.render import setup_mpl
plt = setup_mpl(); fig, ax = plt.subplots(1, len(imgs), figsize=(2.4 * len(imgs), 3.0))
for k, (im, lb) in enumerate(zip(imgs, labels)):
    ax[k].imshow(im); ax[k].axis("off"); ax[k].set_title(lb, fontsize=9)
fig.suptitle("XVLA 视觉混叠证据:末态折叠帧 与 start(m0,value≈0)/end(value≈1) milestone 原型 —— 对 DINOv3 都'白布绿台'极相似,故 value 易翻", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = REPO / "crave/docs/visualization/cross_dataset/xvla_alias_proof.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
