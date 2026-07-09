#!/usr/bin/env python
"""Corrected prod milestone+1 video: BANK-space latents (what the models were trained on)
+ dec_gan decoder (sharp) + native-speed display.

BUG fixed: the prior native video re-encoded frames fresh (encode_pooled), which is a
DIFFERENT latent space than the cached feature bank the prod model + decoder were trained
on (cos only ~0.86) -> OOD -> bad decode. Here we use the pairs npz's bank-space `current`
+ `next_medoid` directly, so predictions & decodes are in-distribution.

Native speed: read every real frame for the left panel; the right panels (pred/true decode)
are computed per stride-10 pair-frame and HELD between them.
Layout: left real | right-top PRED milestone+1 -> dec_gan | right-bottom TRUE milestone+1 -> dec_gan.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_prod_milestone import ProdNet, build_feat  # noqa: E402
from train_dinov3h_decoder import PooledDecoder, l2  # noqa: E402
from make_episode_native_video import label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1819)
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone/member_*.pt")
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec_gan.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--out", default="lmwm/docs/assets/prod_milestone_gan.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.decoder, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval()
    def decode(lat):
        with torch.no_grad():
            o = D(torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    z = np.load(args.pairs); proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    feat = build_feat(z, proto); din = feat.shape[1]
    idx = np.where(z["episode_id"] == args.episode)[0]
    idx = idx[np.argsort(z["t"][idx])]
    ts = z["t"][idx].astype(np.int64)
    med = z["next_medoid"][idx].astype(np.float32); med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8

    paths = sorted(glob.glob(args.members)); protos = None
    X = torch.from_numpy(feat[idx].astype(np.float32)).to(dev)
    for p in paths:
        c = torch.load(p, map_location="cpu"); m = ProdNet(din, len(proto)).to(dev); m.load_state_dict(c["model"]); m.eval()
        with torch.no_grad():
            _, pr = m(X)
        g = F.normalize(pr.float(), -1).cpu().numpy(); protos = g if protos is None else protos + g
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8

    pred_dec = decode(protos)            # [P,128,128,3] one per pair-frame
    true_dec = decode(med)

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    vid = args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/{args.camera}/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS); out_fps = round(fps) if fps and fps > 1 else 30

    BIG = R * 2; RW = 230; W = BIG + 8 + RW; H = 26 + BIG + 4
    def rlabel(img, t): return label(cv2.resize(img, (RW, RW)), t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    f = 0
    while f < args.max_frames:
        okf, im = cap.read()
        if not okf:
            break
        k = int(np.searchsorted(ts, f, side="right") - 1); k = max(0, min(k, len(idx) - 1))
        left = label(cv2.resize(im[:, :, ::-1], (BIG, BIG)), f"ep{args.episode} frame {f} (real, native)")
        rt = rlabel(pred_dec[k], "PRED milestone+1 -> decode")
        rb = rlabel(true_dec[k], "TRUE milestone+1 -> decode")
        right = cv2.resize(np.vstack([rt, rb]), (RW, left.shape[0]))
        canvas = np.hstack([left, np.full((left.shape[0], 8, 3), 20, np.uint8), right])
        vw.write(cv2.cvtColor(canvas[:H, :W], cv2.COLOR_RGB2BGR)); f += 1
    cap.release(); vw.release()
    print(f"saved {args.out} | {f} frames @ {out_fps}fps | {len(idx)} pair-frames (bank-space, dec_gan)")


if __name__ == "__main__":
    main()
