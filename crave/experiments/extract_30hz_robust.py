#!/usr/bin/env python
"""Robust 30Hz native DINOv3-H feature extraction for ALL kai0_base eps (crop224,shard-cos=1.0).
GPU1: --rank 0 --world 1 → resumable, skip existing.
Save: temp/crave_30hz_feat_v2/ep{e}.npy (n30,1280) fp16.
"""
import argparse, time, av, cv2, numpy as np
from pathlib import Path
from crave.encoders import load_encoder
from crave.config import resolve_dataset
from crave.data import kai0

REPO = Path("/home/tim/workspace/deepdive_kai0"); OUT = REPO / "temp/crave_30hz_feat_v2"; OUT.mkdir(exist_ok=True, parents=True)

def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
    all_eps = sorted(int(p.stem.split("_")[1]) for p in (DS / "data").glob("chunk-*/episode_*.parquet"))
    mine = [e for i, e in enumerate(all_eps) if i % a.world == a.rank and not (OUT / f"ep{e}.npy").exists()]
    print(f"[rank{a.rank}/{a.world}] {len(mine)}/{len(all_eps)} eps to do", flush=True)
    enc = load_encoder("dinov3-h")
    t0 = time.time(); done = 0; errs = 0
    for e in mine:
        try:
            vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
            cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
            cap.close()
            if len(frames) < 3:
                print(f"  WARN ep{e}: only {len(frames)} frames, skip", flush=True); continue
            feat = enc.encode_pooled(frames).astype(np.float16); np.save(OUT / f"ep{e}.npy", feat); done += 1
            if done % 50 == 0:
                el = time.time() - t0
                eta = el / done * (len(mine) - done) / 60
                print(f"  [{done}/{len(mine)}] ep{e} n={len(frames)}  ({el/60:.0f}min elapsed, ~{eta:.0f}min left)", flush=True)
        except Exception as ex:
            errs += 1
            if errs <= 5: print(f"  ERR ep{e}: {ex}", flush=True)
    print(f"[rank{a.rank}] DONE {done} eps, {errs} errors in {(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
