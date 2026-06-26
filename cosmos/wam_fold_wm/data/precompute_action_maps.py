"""Precompute EVAC spatial action maps for visrobot01 episodes (CPU, background) and cache them,
so the training dataloader just loads + aligns to its (possibly strided) latent frames instead of
running FK+projection+render per step.

Cache layout: <cache>/<rig>/episode_<id>.npy  with array [T, Hc, Wc, 3] uint8 (downsampled).
The packer (t1e mode) loads this, selects the episode's frame indices, resizes to latent (H,W),
and uses as cond_tokens channels (3 RGB).
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_map_render import MultiViewActionMapRenderer  # noqa: E402  (3-cam stacked, matches WM video layout)

_REPO = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="visrobot01_v3_train")
    ap.add_argument("--hc", type=int, default=90)  # cached stacked-map size (720->90, resized to latent later)
    ap.add_argument("--wc", type=int, default=80)  # 640->80; stacked layout preserved
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    cache = a.cache or f"/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/actmap_cache_3cam/{a.rig}"
    os.makedirs(cache, exist_ok=True)
    root = f"{_REPO}/kai0/data/wam_fold_v3/{a.rig}"
    eps = sorted(glob.glob(f"{root}/data/chunk-000/episode_*.parquet"))
    if a.limit:
        eps = eps[: a.limit]
    if a.nshards > 1:
        eps = eps[a.shard :: a.nshards]  # shard for parallel CPU workers (idempotent skip handles overlap)
    r = MultiViewActionMapRenderer()
    print(f"[precompute] {a.rig} 3cam-stacked: {len(eps)} episodes -> {cache} ({a.hc}x{a.wc})", flush=True)
    done = 0
    for ep in eps:
        eid = ep.split("episode_")[-1].split(".")[0]
        out = f"{cache}/episode_{eid}.npy"
        if os.path.exists(out):
            done += 1
            continue
        try:
            state = np.stack(pd.read_parquet(ep)["observation.state"].values)  # [T,14]
            amap = r.render(state)  # [T,480,640,3] uint8
            small = np.stack([cv2.resize(f, (a.wc, a.hc), interpolation=cv2.INTER_AREA) for f in amap])
            np.save(out, small.astype(np.uint8))
            done += 1
            if done % 20 == 0:
                print(f"[precompute] {done}/{len(eps)} (last {eid}: {small.shape})", flush=True)
        except Exception as e:
            print(f"[precompute] FAIL {eid}: {repr(e)[:150]}", flush=True)
    print(f"[precompute] DONE {done}/{len(eps)} -> {cache}", flush=True)


if __name__ == "__main__":
    main()
