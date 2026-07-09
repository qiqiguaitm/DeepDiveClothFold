#!/usr/bin/env python
"""GPU0: extract DINOv3-H grid features for 200 eps → temp/grid_feats_dinov3h/"""
import sys, time, numpy as np, cv2, av
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crave.encoders import load_encoder
from crave.config import resolve_dataset
from crave.data import kai0

rng = np.random.RandomState(42)
REPO = Path("/home/tim/workspace/deepdive_kai0")

def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
idx = np.load(REPO / "temp/crave_full_dinov3h/index.npz")
E, FR = idx["E"], idx["FR"]
ep_list = sorted(set(E.tolist()))
samp_eps = sorted(rng.choice(ep_list, 200, replace=False))

enc = load_encoder("dinov3-h")
OUT = REPO / "temp/grid_feats_dinov3h"; OUT.mkdir(exist_ok=True)
t0 = time.time()
for ei, e in enumerate(samp_eps):
    fp = OUT / f"ep{e}.npy"
    if fp.exists(): continue
    loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); all_frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [all_frames[i] for i in fr if i < len(all_frames)]
    if imgs:
        grids = enc.encode_grid(imgs)
        np.save(fp, grids.reshape(len(grids), -1).astype(np.float16))
    if (ei + 1) % 20 == 0:
        print(f"grid {ei+1}/{len(samp_eps)} ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE grid {len(samp_eps)} eps in {(time.time()-t0)/60:.1f}min", flush=True)
