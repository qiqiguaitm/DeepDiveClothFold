#!/usr/bin/env python
"""Decode a full episode's DINOv3-base pooled-feature sequence into a coherent VIDEO.

Loads a temporal decoder (train_dinov3base_video_decoder.py) and runs it over an
episode's entire feature sequence with temporal context (Conv1d over the whole
sequence -> every frame sees its neighbours), then writes a side-by-side
[real | decoded] mp4 so the temporal coherence is directly visible.

Usage:
    python lmwm/scripts/make_dinov3base_decode_video.py \
        --ckpt lmwm/checkpoints/dinov3base_decoder/kai_video_dec.pt \
        --feature_dir lmvla/crave/data/kai_dinov3base \
        --dataset_root kai0/data/Task_A/kai0_base \
        --episode 100 --out lmwm/docs/assets/dinov3base_decode_ep100.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3base_video_decoder import TemporalPooledDecoder, load_features, l2  # noqa: E402


def write_h264(tmp: Path, out: Path, fps: int) -> None:
    """Transcode a cv2-written mp4v temp to browser/VS Code-playable H.264(avc1); fallback to mp4v."""
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-crf", "20", "-r", str(fps), str(out)], capture_output=True, text=True)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True)
            return
        print(f"  ffmpeg failed ({r.stderr.strip()[:120]}), keeping mp4v", flush=True)
    shutil.move(str(tmp), str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--episode", type=int, default=100)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    dec = TemporalPooledDecoder(din=ck["din"], res=ck["res"], ctx=ck["ctx"], tk=ck["tk"]).to(device)
    dec.load_state_dict(ck["model"]); dec.eval()
    R = ck["res"]

    E, FR, F, _ = load_features(args.feature_dir)
    m = E == args.episode
    if not m.any():
        raise SystemExit(f"episode {args.episode} not in features")
    fr = FR[m]; o = np.argsort(fr); fr = fr[o]; feats = l2(F[m][o].astype(np.float32))
    seq = torch.from_numpy(feats).unsqueeze(0).to(device)                        # (1,L,din)

    with torch.no_grad():
        h = dec.tenc(seq.transpose(1, 2)).transpose(1, 2)[0]                     # (L,ctx) full-sequence context
        dec_imgs = []
        for s in range(0, len(h), 64):
            img = dec.net(dec.fc(h[s:s + 64]).view(-1, 512, 4, 4))               # (b,3,R,R)
            dec_imgs.append(((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy())
    dec_imgs = np.concatenate(dec_imgs)                                          # (L,R,R,3) RGB

    chunks_size = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    cam = "observation.images.top_head"
    mp4 = args.dataset_root / f"videos/chunk-{args.episode // chunks_size:03d}/{cam}/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".mp4v.tmp.mp4")                                   # cv2 lacks H.264 -> write mp4v then transcode
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (R * 2, R))
    for t in range(len(fr)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[t]))
        ok, real = cap.read()
        real = cv2.resize(real, (R, R)) if ok else np.zeros((R, R, 3), np.uint8)
        deci = cv2.cvtColor(dec_imgs[t], cv2.COLOR_RGB2BGR)
        vw.write(np.concatenate([real, deci], axis=1))
    cap.release(); vw.release()
    write_h264(tmp, args.out, args.fps)
    print(f"wrote {args.out}  ({len(fr)} frames, [real | decoded], H.264/avc1)")


if __name__ == "__main__":
    main()
