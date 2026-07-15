#!/usr/bin/env python
"""Quantify temporal consistency: temporal video decoder vs per-frame baseline, on one episode.

Metrics over the decoded episode (128x128 in [-1,1]):
  real_motion   = mean_t |I_t - I_{t-1}|                 (ground-truth inter-frame change)
  dec_motion    = mean_t |D_t - D_{t-1}|                 (decoded inter-frame change)
  tc_error      = mean_t |(D_t-D_{t-1}) - (I_t-I_{t-1})| (decoded motion vs real motion; LOWER=better)
  static_flicker= mean dec_motion over frames whose real_motion is in the bottom 30%
                  (LOWER=better; high => decoder flickers where the scene is still)
  motion_corr   = corr_t(dec_motion_t, real_motion_t)    (HIGHER=better; decoded moves in sync with real)

Temporal decoder decodes the whole sequence at once (temporal context);
per-frame decoder decodes each frame in isolation (as it was trained, clip=1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3base_video_decoder import TemporalPooledDecoder, load_features, l2  # noqa: E402


def load_dec(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    dec = TemporalPooledDecoder(din=ck["din"], res=ck["res"], ctx=ck["ctx"], tk=ck["tk"]).to(device)
    dec.load_state_dict(ck["model"]); dec.eval()
    return dec, ck["res"]


def decode_seq(dec, feats, device, per_frame=False):
    seq = torch.from_numpy(feats).unsqueeze(0).to(device)                        # (1,L,din)
    with torch.no_grad():
        if per_frame:                                                            # isolate each frame (no temporal mix)
            imgs = []
            for t in range(seq.shape[1]):
                h = dec.tenc(seq[:, t:t + 1].transpose(1, 2)).transpose(1, 2)[0]  # (1,ctx)
                imgs.append(dec.net(dec.fc(h).view(-1, 512, 4, 4)))
            out = torch.cat(imgs)
        else:
            h = dec.tenc(seq.transpose(1, 2)).transpose(1, 2)[0]                 # (L,ctx) full context
            out = torch.cat([dec.net(dec.fc(h[s:s + 64]).view(-1, 512, 4, 4)) for s in range(0, len(h), 64)])
    return out.clamp(-1, 1).cpu().numpy()                                        # (L,3,R,R)


def metrics(D, I):
    dm = np.abs(np.diff(D, axis=0)).mean((1, 2, 3))                              # (L-1,) decoded motion
    rm = np.abs(np.diff(I, axis=0)).mean((1, 2, 3))                              # real motion
    tc = np.abs(np.diff(D, axis=0) - np.diff(I, axis=0)).mean()
    lo = rm <= np.quantile(rm, 0.30)                                            # static frames
    return dict(real_motion=float(rm.mean()), dec_motion=float(dm.mean()),
                tc_error=float(tc), static_flicker=float(dm[lo].mean()),
                motion_corr=float(np.corrcoef(dm, rm)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal_ckpt", required=True, type=Path)
    ap.add_argument("--perframe_ckpt", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--episode", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    E, FR, F, _ = load_features(args.feature_dir)
    m = E == args.episode
    fr = FR[m]; o = np.argsort(fr); fr = fr[o]; feats = l2(F[m][o].astype(np.float32))

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    mp4 = args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/observation.images.top_head/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(mp4)); R = 128
    I = np.zeros((len(fr), 3, R, R), np.float32)
    for t in range(len(fr)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[t])); ok, im = cap.read()
        if ok:
            I[t] = (cv2.resize(im[:, :, ::-1], (R, R)).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
    cap.release()

    tdec, _ = load_dec(args.temporal_ckpt, device)
    pdec, _ = load_dec(args.perframe_ckpt, device)
    Dt = decode_seq(tdec, feats, device, per_frame=False)
    Dp = decode_seq(pdec, feats, device, per_frame=True)
    mt, mp = metrics(Dt, I), metrics(Dp, I)
    print(f"episode {args.episode}, {len(fr)} frames\n")
    print(f"{'metric':<16}{'per-frame':>12}{'temporal':>12}   (better)")
    for k, better in [("tc_error", "lower"), ("static_flicker", "lower"), ("motion_corr", "higher"),
                      ("dec_motion", "~real"), ("real_motion", "ref")]:
        print(f"{k:<16}{mp[k]:>12.4f}{mt[k]:>12.4f}   {better}")
    print(json.dumps({"episode": args.episode, "per_frame": mp, "temporal": mt}, indent=2))


if __name__ == "__main__":
    main()
