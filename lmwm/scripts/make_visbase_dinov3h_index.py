#!/usr/bin/env python
"""Build a DINOv3-H feature index (index.npz + shard_0.npz, load_index-compatible) for ONE vis_base
episode, so render_twomodel_video.py can run cross-domain milestone prediction on it (assign to the
existing kai0 37-prototype milestones). Mirrors reencode_pooled_unified's encoder path.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
import cv2  # noqa: E402
from lever_patch_token import read_enc  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def ep_frame_count(root: Path, camera: str, ep: int) -> int:
    pats = [str(root / f"videos/chunk-*/{camera}/episode_{ep:06d}.mp4"),
            str(root / f"*/videos/chunk-*/{camera}/episode_{ep:06d}.mp4")]
    vids = [v for p in pats for v in glob.glob(p)]
    if not vids:
        raise SystemExit(f"no video for ep{ep} camera={camera} under {root}")
    cap = cv2.VideoCapture(vids[0]); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    print(f"ep{ep}: {vids[0]} -> {n} frames", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    T = ep_frame_count(args.dataset_root, args.camera, args.episode)
    E = np.full(T, args.episode, np.int64); FR = np.arange(T, dtype=np.int64)

    enc = load_encoder("dinov3-h", device=args.device)
    imgs = read_enc(args.dataset_root, args.camera, E, FR, 256)                 # cv2 read + resize 256
    feat = enc.encode_pooled(imgs).astype(np.float16)                          # (T, 1280) DINOv3-H pooled

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "index.npz", E=E, FR=FR, T=(FR / args.fps).astype(np.float32), n=np.int64(T))
    np.savez(args.out_dir / "shard_0.npz", gidx=np.arange(T, dtype=np.int64), feat=feat, valid=np.ones(T, bool))
    print(f"wrote {args.out_dir}: T={T} feat={feat.shape} dim={feat.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
