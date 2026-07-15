import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, "crave/src")
sys.path.insert(0, "crave/scripts")
from crave.config import resolve_dataset
from crave.data import list_eps, load_ep
from crave.encoders import load_encoder
from crave.utils import L2, mkp_gap
from generalize import build_milestones  # crave/scripts/generalize.py

REPO = Path("/home/tim/workspace/deepdive_kai0")
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = resolve_dataset("vis")
enc = load_encoder("dinov3-h")
eps = list_eps(cfg)
t0 = time.time()
print(f"[vis] {len(eps)} eps stride={STRIDE} cam={cfg.cam}", flush=True)

POOL, STATE, EPID, TPOS, FRIDX = [], [], [], [], []
eplen = {}
for k, e in enumerate(eps):
    try:
        f224, state, th, nidx = load_ep(cfg, e, strd=STRIDE)
    except Exception as ex:
        print(f"  ep{e} skip ({ex})", flush=True); continue
    if len(f224) < 5:
        continue
    pooled = L2(enc.encode_pooled(f224)); n = len(f224)
    POOL.append(pooled); STATE.append(mkp_gap(state, STRIDE))
    EPID.append(np.full(n, e, np.int64)); TPOS.append(np.arange(n) / max(1, n - 1))
    FRIDX.append(nidx[:n].astype(np.int64)); eplen[e] = n
    if (k + 1) % 25 == 0:
        print(f"  {k+1}/{len(eps)} ({time.time()-t0:.0f}s) N={sum(len(p) for p in POOL)}", flush=True)

img = np.concatenate(POOL)          # (N,1280) L2-normalized DINOv3-H pooled
Pm = np.concatenate(STATE)
E = np.concatenate(EPID)
Tv = np.concatenate(TPOS)           # normalized time within ep
FR = np.concatenate(FRIDX)          # native video frame index
PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
Pn = L2((Pm - PMU) / PSD)
F = np.concatenate([img, Pn], 1)    # clustering space [img | proprio]
ne = len(eps)
N = len(F)
print(f"[vis] N={N} img={img.shape} F={F.shape} clustering...", flush=True)

cen, lab, order, Pord, M = build_milestones(F, E, Tv, ne)
print(f"[vis] M={M} milestones", flush=True)

proto = cen[order][:, :1280].astype(np.float32)
pord = np.asarray(Pord, np.float32)

# --- write bank ---
out = REPO / "temp/vis_dinov3h"; out.mkdir(parents=True, exist_ok=True)
T = (FR / 30.0).astype(np.float32)   # 30 fps video (load_ep_native uses fps=30)
np.savez(out / "index.npz", E=E.astype(np.int64), FR=FR.astype(np.int64), T=T, n=np.int64(N))
np.savez(out / "shard_0.npz", gidx=np.arange(N, dtype=np.int64),
         feat=img.astype(np.float16), valid=np.ones(N, bool))
rg = REPO / "lmwm/data/recurrence_graphs/vis_dinov3h"; rg.mkdir(parents=True, exist_ok=True)
np.savez(rg / "recurrence_graph.npz", prototype_table=proto, pord=pord)
print(f"[vis] wrote: N={N} feat={img.shape} proto={proto.shape} "
      f"pord[{pord.min():.3f},{pord.max():.3f}] M={M} ({time.time()-t0:.0f}s)", flush=True)
