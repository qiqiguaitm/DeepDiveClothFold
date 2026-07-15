#!/usr/bin/env python
"""Render milestone+1 PREDICTION decode vs milestone+1 REPRESENTATIVE-FRAME decode.

For each frame of an episode:
  current milestone  m_t = argmax_m  feat_t · proto[m]          (DINOv3-base nearest centroid)
  predicted next     m1  = greedy_next[m_t]                      (recurrence-graph milestone+1)
Two renders of that predicted milestone+1 (both via the sharp GRID decoder):
  · PROTOTYPE decode  = grid-decode( mean grid over m1's top-N nearest frames )   -> averaged / smooth
  · REPRESENTATIVE    = grid-decode( grid of m1's medoid frame )                  -> a real exemplar, sharp
Panels: [ real(current) | milestone+1 预测原型 | milestone+1 代表帧 ]. Episodes concatenated into one mp4.

Run in the CRAVE env (needs DINOv3-base grid encode):
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
from train_dinov3base_video_decoder import load_features, l2  # noqa: E402
from crave.decoding import make_decoder  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def read_frames(root, cam, cs, ef_list, res):
    """ef_list: list of (E,FR). returns uint8 (N,res,res,3) RGB, grouped-by-episode reads."""
    out = np.zeros((len(ef_list), res, res, 3), np.uint8)
    by_ep: dict[int, list[tuple[int, int]]] = {}
    for i, (e, fr) in enumerate(ef_list):
        by_ep.setdefault(int(e), []).append((i, int(fr)))
    for e, items in by_ep.items():
        cap = cv2.VideoCapture(str(root / f"videos/chunk-{e // cs:03d}/{cam}/episode_{e:06d}.mp4"))
        for i, fr in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read()
            if ok:
                out[i] = cv2.resize(im[:, :, ::-1], (res, res))
        cap.release()
    return out


def to_bgr(x):
    return cv2.cvtColor(((np.clip(x, -1, 1).transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def write_h264(tmp, out, fps):
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True); return
    shutil.move(str(tmp), str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--grid_ckpt", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--episodes", default="100,300,800", help="comma list")
    ap.add_argument("--topn", type=int, default=24, help="frames per milestone for prototype/medoid")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    dev = args.device; R = 128
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])

    g = np.load(args.graph)
    proto = l2(g["prototype_table"].astype(np.float32)); greedy = g["greedy_next"].astype(np.int64); M = len(proto)
    E, FR, F, _ = load_features(args.feature_dir)
    Fl = l2(F.astype(np.float32))

    # per-milestone top-N nearest frames (medoid = rank0), collect their (E,FR)
    print("finding per-milestone representative frames ...", flush=True)
    sims = Fl @ proto.T                                                          # (N,M)
    topn_ef = {}
    for m in range(M):
        idx = np.argpartition(-sims[:, m], args.topn)[:args.topn]
        idx = idx[np.argsort(-sims[idx, m])]
        topn_ef[m] = [(int(E[i]), int(FR[i])) for i in idx]
    flat_ef = [ef for m in range(M) for ef in topn_ef[m]]

    # grid-encode representative frames -> per-milestone grid centroid + medoid grid
    print(f"grid-encoding {len(flat_ef)} representative frames ...", flush=True)
    enc = load_encoder("dinov3-base", device=dev)
    imgs = read_frames(args.dataset_root, args.camera, cs, flat_ef, 224)
    Gr = np.zeros((len(imgs), 768, 16, 16), np.float32)
    for b in range(0, len(imgs), 256):
        Gr[b:b + 256] = enc.encode_grid(imgs[b:b + 256], bs=64).astype(np.float32)
    enc.unload()
    Gr = Gr / (np.linalg.norm(Gr, axis=1, keepdims=True) + 1e-8)
    Gr = Gr.reshape(M, args.topn, 768, 16, 16)
    grid_centroid = l2(Gr.mean(1).transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)  # (M,768,16,16) mean then l2 per patch
    grid_medoid = Gr[:, 0]                                                        # (M,768,16,16)

    gck = torch.load(args.grid_ckpt, map_location=dev, weights_only=False)
    dec = make_decoder(768, gck["dec"]).to(dev); dec.load_state_dict(gck["model"]); dec.eval()

    def decode(grids):                                                           # (K,768,16,16)->(K,3,R,R)
        with torch.no_grad():
            return dec(torch.from_numpy(grids.astype(np.float32)).to(dev)).cpu().numpy()

    proto_dec = decode(grid_centroid)                                            # (M,3,R,R) per-milestone renders
    medoid_dec = decode(grid_medoid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (R * 3, R + 16))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ep in [int(x) for x in args.episodes.split(",")]:
        m = E == ep
        if not m.any():
            print(f"  ep{ep} absent, skip"); continue
        fr = FR[m]; o = np.argsort(fr); fr = fr[o]
        feats = Fl[m][o]; ms = (feats @ proto.T).argmax(1)                       # current milestone per frame
        reals = read_frames(args.dataset_root, args.camera, cs, [(ep, int(x)) for x in fr], R)
        for t in range(len(fr)):
            m1 = int(greedy[ms[t]])                                              # milestone+1 prediction
            row = np.concatenate([reals[t][:, :, ::-1], to_bgr(proto_dec[m1]), to_bgr(medoid_dec[m1])], axis=1)
            bar = np.zeros((16, R * 3, 3), np.uint8)
            for i, txt in enumerate(["real (t)", "m+1 pred proto", "m+1 repr frame"]):
                cv2.putText(bar, txt, (i * R + 2, 12), font, 0.33, (255, 255, 255), 1)
            vw.write(np.concatenate([bar, row], axis=0))
        print(f"  ep{ep}: {len(fr)} frames", flush=True)
    vw.release()
    write_h264(tmp, args.out, args.fps)
    print(f"wrote {args.out}  ([real | m+1 pred prototype | m+1 representative], H.264)")


if __name__ == "__main__":
    main()
