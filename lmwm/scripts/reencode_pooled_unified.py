#!/usr/bin/env python
"""Re-encode all bank frames via the UNIFIED entry: cv2 read + crave encode_pooled
(DINOv3-H). This is the deploy/viz path -> training==deploy same latent space, fixing
the old bank's ~0.86 mismatch. Sharded to a new bank dir; run as two row-ranges on 2 GPUs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lever_patch_token import read_enc  # noqa: E402  (cv2 reader + resize 256)
from crave.encoders import load_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_bank", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--out_dir", default="temp/crave_full_dinov3h_v2", type=Path)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=-1)
    ap.add_argument("--shard", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    idx = np.load(args.old_bank / "index.npz")
    E, FR = idx["E"].astype(np.int64), idx["FR"].astype(np.int64)
    n = int(idx["n"]); hi = n if args.hi < 0 else min(args.hi, n)
    enc = load_encoder("dinov3-h", device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in range(args.lo, hi, args.shard):
        e = min(s + args.shard, hi)
        out = args.out_dir / f"feat_{s:07d}_{e:07d}.npz"
        if out.exists():
            print(f"skip {out.name}", flush=True); continue
        rows = np.arange(s, e)
        imgs = read_enc(args.dataset_root, args.camera, E[rows], FR[rows], 256)
        feat = enc.encode_pooled(imgs).astype(np.float16)
        np.savez(out, gidx=rows.astype(np.int64), feat=feat)
        print(f"saved {out.name} [{s}:{e}] {feat.shape}", flush=True)
    print(f"DONE [{args.lo}:{hi}] on {args.device}", flush=True)


if __name__ == "__main__":
    main()
