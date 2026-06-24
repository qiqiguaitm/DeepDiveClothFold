"""CRAVE alignment study — fast-iteration cache builder.

Subsamples ~3Hz, caches DINOv2-large pooled image (1024) + proprio(28, mkp_gap) + state +
ep ids + normalized time + thumbnails to temp/crave_align/<ds>_cache.npz.

Reuses production loaders/encoder from crave_generalize.py (does NOT edit it).

Run:  HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_align_cache.py <vis|xvla|coffee>
"""
import sys, os, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from pathlib import Path
import crave_generalize as G

OUT = Path("/vePFS/tim/workspace/deepdive_kai0/temp/crave_align")
OUT.mkdir(parents=True, exist_ok=True)

# cap eps per ds for fast iteration (3Hz subsample → stride = cfg["stride"])
MAXEP = {"vis": 150, "xvla": 150, "coffee": 50}


def build(ds):
    t0 = time.time()
    cfg = G.CFG[ds]
    eps = G.list_eps(ds, cfg)[: MAXEP[ds]]
    print(f"[{ds}] building cache from {len(eps)} eps @ 3Hz (stride={cfg['stride']})", flush=True)
    proc, enc = G.make_enc()
    POOL, STATE, EPID, TPOS, THUMB, NIDX, eplen = [], [], [], [], [], [], {}
    for k, e in enumerate(eps):
        try:
            f224, state, th, nidx = G.load_ep(ds, cfg, e)  # default stride → 3Hz
        except Exception as ex:
            print(f"  ep{e} skip ({ex})", flush=True); continue
        if len(f224) < 5:
            continue
        pooled = G.enc_pooled(f224, proc, enc)
        pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)
        n = len(f224)
        POOL.append(pooled.astype(np.float32))
        STATE.append(G.mkp_gap(state, cfg["stride"]).astype(np.float32))  # (n,28)
        EPID.append(np.full(n, e))
        TPOS.append((np.arange(n) / max(1, n - 1)).astype(np.float32))
        THUMB += [t.astype(np.uint8) for t in th]
        NIDX.append(nidx.astype(np.int32))
        eplen[e] = n
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(eps)} ({time.time()-t0:.0f}s)", flush=True)
    img = np.concatenate(POOL)
    Pm = np.concatenate(STATE)
    E = np.concatenate(EPID).astype(np.int32)
    Tv = np.concatenate(TPOS)
    NIDX = np.concatenate(NIDX)
    THUMB = np.stack(THUMB).astype(np.uint8)
    np.savez_compressed(
        OUT / f"{ds}_cache.npz",
        img=img, state=Pm, ep=E, tpos=Tv, nidx=NIDX, thumb=THUMB,
        eps=np.array(sorted(eplen)), stride=cfg["stride"],
    )
    del enc
    import torch; torch.cuda.empty_cache()
    print(f"[{ds}] cache: N={len(img)} img{img.shape} state{Pm.shape} thumb{THUMB.shape} "
          f"({time.time()-t0:.0f}s) → {OUT/f'{ds}_cache.npz'}", flush=True)


if __name__ == "__main__":
    build(sys.argv[1])
