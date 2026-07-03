#!/usr/bin/env python
"""Precompute DINOv3-H patch grids for the augin pairs, sharded to disk (fp16).

Full set is ~131GB (200k x 256 x 1280 x fp16); run as two row-ranges across two GPUs
in parallel. Each shard file stores the pair row indices + grid [rows,256,1280] fp16,
so training (local or on the 8-GPU server via shared FS) can mmap/load without re-encoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lever_patch_token import read_enc  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=-1)
    ap.add_argument("--shard", type=int, default=10000)
    ap.add_argument("--out_dir", default="lmwm/data/grid_cache", type=Path)
    ap.add_argument("--dino_path", default=None, help="override DINOv3-H weights dir (for gf3)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    hi = n if args.hi < 0 else min(args.hi, n)
    eps = z["episode_id"].astype(np.int64); ts = z["t"].astype(np.int64)
    enc = load_encoder("dinov3-h", device=args.device, **({"path": args.dino_path} if args.dino_path else {}))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # contiguous [s:e] shards saved as .npy (memmap-able; row r lives in the shard whose
    # [lo,hi) range contains r at local index r-lo -> no separate index needed).
    for s in range(args.lo, hi, args.shard):
        e = min(s + args.shard, hi)
        out = args.out_dir / f"grid_{s:07d}_{e:07d}.npy"
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        rows = np.arange(s, e)
        imgs = read_enc(args.dataset_root, args.camera, eps[rows], ts[rows], 256)
        grid = enc.encode_grid(imgs).astype(np.float16).reshape(len(rows), 1280, 256).transpose(0, 2, 1)  # [r,256,1280]
        np.save(out, grid)
        print(f"saved {out.name} [{s}:{e}] {grid.shape}", flush=True)
    print(f"DONE range [{args.lo}:{hi}] on {args.device}", flush=True)


if __name__ == "__main__":
    main()
