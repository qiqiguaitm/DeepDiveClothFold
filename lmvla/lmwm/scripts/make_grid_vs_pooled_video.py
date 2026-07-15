#!/usr/bin/env python
"""Side-by-side [real | pooled-decode | grid-decode] video for one episode.

Shows the sharpness gap: pooled 768D (no spatial layout -> soft prototype) vs
DINOv3-base GRID 768x16x16 (spatial tokens -> sharp). Both temporally regularized.
Run in the CRAVE env (needs DINOv3-base for grid encode):
    /home/tim/miniconda3/envs/srpo/bin/python
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path("/home/tim/workspace/deepdive_kai0")
sys.path.insert(0, str(REPO / "lmvla/lmwm/scripts"))
sys.path.insert(0, str(REPO / "lmvla/crave/src"))
from train_dinov3base_video_decoder import TemporalPooledDecoder, load_features, l2  # noqa: E402
from crave.decoding import make_decoder  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def to_bgr(x):
    return cv2.cvtColor(((np.clip(x, -1, 1).transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def write_h264(tmp: Path, out: Path, fps: int):
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True); return
    shutil.move(str(tmp), str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled_ckpt", required=True, type=Path)
    ap.add_argument("--grid_ckpt", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--episode", type=int, default=100)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    dev = args.device; R = 128

    E, FR, F, _ = load_features(args.feature_dir)
    m = E == args.episode
    fr = FR[m]; o = np.argsort(fr); fr = fr[o]; feats = l2(F[m][o].astype(np.float32))

    # pooled decode (temporal full-sequence)
    pck = torch.load(args.pooled_ckpt, map_location=dev, weights_only=False)
    pdec = TemporalPooledDecoder(din=pck["din"], res=pck["res"], ctx=pck["ctx"], tk=pck["tk"]).to(dev)
    pdec.load_state_dict(pck["model"]); pdec.eval()
    with torch.no_grad():
        seq = torch.from_numpy(feats).unsqueeze(0).to(dev)
        h = pdec.tenc(seq.transpose(1, 2)).transpose(1, 2)[0]
        Dp = torch.cat([pdec.net(pdec.fc(h[s:s + 64]).view(-1, 512, 4, 4)) for s in range(0, len(h), 64)]).cpu().numpy()

    # read frames + encode grid + grid decode
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    mp4 = args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/observation.images.top_head/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    reals, enc224 = [], []
    for t in range(len(fr)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[t])); ok, im = cap.read()
        rgb = im[:, :, ::-1] if ok else np.zeros((R, R, 3), np.uint8)
        reals.append(cv2.resize(rgb, (R, R))); enc224.append(cv2.resize(rgb, (224, 224)))
    cap.release()
    enc = load_encoder("dinov3-base", device=dev)
    enc224 = np.stack(enc224)
    G = np.zeros((len(enc224), 768, 16, 16), np.float32)
    for b in range(0, len(enc224), 256):
        G[b:b + 256] = enc.encode_grid(enc224[b:b + 256], bs=64).astype(np.float32)
    enc.unload()
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    gck = torch.load(args.grid_ckpt, map_location=dev, weights_only=False)
    gdec = make_decoder(768, gck["dec"]).to(dev); gdec.load_state_dict(gck["model"]); gdec.eval()
    with torch.no_grad():
        Dg = torch.cat([gdec(torch.from_numpy(G[s:s + 64]).to(dev)) for s in range(0, len(G), 64)]).cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (R * 3, R + 16))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(len(fr)):
        row = np.concatenate([reals[t][:, :, ::-1], to_bgr(Dp[t]), to_bgr(Dg[t])], axis=1)
        bar = np.zeros((16, R * 3, 3), np.uint8)
        for i, txt in enumerate(["real", "pooled", "grid"]):
            cv2.putText(bar, txt, (i * R + 4, 12), font, 0.4, (255, 255, 255), 1)
        vw.write(np.concatenate([bar, row], axis=0))
    vw.release()
    write_h264(tmp, args.out, args.fps)
    print(f"wrote {args.out}  ({len(fr)} frames, [real | pooled | grid], H.264)")


if __name__ == "__main__":
    main()
