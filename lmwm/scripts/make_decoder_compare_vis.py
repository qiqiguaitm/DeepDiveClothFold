#!/usr/bin/env python
"""Compare the TWO decoders that satisfy both requirements (R1 determinism + R2 continuity) on the
SAME milestone hint (forward-from-current predicted milestone+1):

  real (native) | PRED-fwd -> dec_v2 (L1, blurry) | PRED-fwd -> flow FIXED-noise (sharp)

Outputs a consecutive-frame filmstrip PNG (see stability + sharpness at a glance) and a 3-panel video.
Both decoders get the identical predicted latents; only the decoder differs.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from make_prod_video_vis_fwd import train_forward, load_any_decoder  # noqa: E402
from make_prod_video_vis import viterbi_assign  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from make_episode_native_video import label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kai0/data/Task_A/vis_base/v4/2026-04-23-v4")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_v2.npz")
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin_v2.npz")
    ap.add_argument("--l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--fwd_ckpt", default="lmwm/checkpoints/fwd_from_current/fwd_predm_v2.pt", type=Path)
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--fwd_steps", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--stay", type=float, default=0.9)
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--out_vid", default="lmwm/docs/assets/decoder_compare_l1_vs_flowfixed.mp4", type=Path)
    ap.add_argument("--out_png", default="lmwm/docs/assets/decoder_compare_l1_vs_flowfixed.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    g = np.load(args.graph_npz)
    proto_n = g["prototype_table"].astype(np.float32)
    proto_n = proto_n / (np.linalg.norm(proto_n, axis=1, keepdims=True) + 1e-8)
    trans = g["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)

    fwd, predm = train_forward(args.pairs, dev, args.code_dim, args.fwd_steps, args.fwd_ckpt)
    dec_l1, R1, _ = load_any_decoder(args.l1, dev)
    dec_fx, R2, tag = load_any_decoder(args.flow, dev)   # flow branch defaults to seed=0 (fixed-noise)
    print(f"decoders: dec_v2 (L1) vs {Path(args.flow).stem} [{tag}]", flush=True)

    vids = (sorted(glob.glob(str(Path(args.root) / f"videos/chunk-*/{args.camera}/episode_*.mp4")))
            or sorted(glob.glob(str(Path(args.root) / f"*/videos/chunk-*/{args.camera}/episode_*.mp4"))))
    vid = vids[0]; print(f"episode: {vid}", flush=True)
    cap = cv2.VideoCapture(vid); frames = []
    while len(frames) < args.max_frames:
        okf, im = cap.read()
        if not okf:
            break
        frames.append(im[:, :, ::-1])
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    out_fps = round(fps) if fps and fps > 1 else 30
    N = len(frames); kt = np.arange(0, N, args.stride)
    print(f"{N} frames, {len(kt)} keyframes", flush=True)

    enc = load_encoder("dinov3-h", device=str(dev))
    lat = enc.encode_pooled(np.stack([cv2.resize(frames[t], (256, 256)) for t in kt])).astype(np.float32)
    latn = lat / (np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8)
    ms = viterbi_assign(latn, proto_n, trans, args.beta, args.stay)
    ch = np.where(np.diff(ms) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
    stage_m = [int(ms[s]) for s in st]
    stage_of = np.zeros(len(kt), int)
    for si, (s, e) in enumerate(zip(st, en)):
        stage_of[s:e] = si

    # forward-from-current predicted milestone+1 (varies smoothly across consecutive keyframes)
    Xf = torch.from_numpy(latn).to(dev)
    feat_fwd = torch.from_numpy(np.concatenate([latn, np.zeros((len(kt), 14), np.float32)], 1)).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_fwd = fwd(torch.cat([Xf, predm(feat_fwd)], -1)).float().cpu().numpy()
    pred_fwd /= (np.linalg.norm(pred_fwd, axis=1, keepdims=True) + 1e-8)

    d_l1 = dec_l1(pred_fwd); d_fx = dec_fx(pred_fwd)
    print("decoded PRED-fwd with both decoders", flush=True)

    # frame-to-frame jump (R2) on the two decode streams, over the whole sequence
    def jump(a): b = a.astype(np.float32) / 255; return float(np.abs(b[1:] - b[:-1]).mean())
    def sharp(a): return float(np.mean([cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for x in a]))
    print(f"dec_v2   : frame-jump {jump(d_l1):.4f}  sharp {sharp(d_l1):.0f}", flush=True)
    print(f"flow-fix : frame-jump {jump(d_fx):.4f}  sharp {sharp(d_fx):.0f}", flush=True)

    # --- filmstrip PNG: pick the longest stage, show up to 10 consecutive keyframes ---
    lens = [e - s for s, e in zip(st, en)]; bi = int(np.argmax(lens)); s0 = st[bi]
    cols = list(range(s0, min(s0 + 10, en[bi])))
    C = 150
    def strip(getter): return np.hstack([cv2.resize(getter(k), (C, C)) for k in cols])
    rows = [("vis_base real (native)", strip(lambda k: cv2.resize(frames[kt[k]], (C, C)))),
            ("PRED-fwd -> dec_v2 (L1): stable but BLURRY", strip(lambda k: d_l1[k])),
            ("PRED-fwd -> flow FIXED-noise: stable AND sharp", strip(lambda k: d_fx[k]))]

    def lab(img, t):
        o = img.copy(); cv2.putText(o, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); return o
    grid = np.vstack([lab(r[:, :, ::-1].copy(), name) for name, r in rows])
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_png), grid)
    print(f"saved {args.out_png} {grid.shape} ({len(cols)} consecutive keyframes, stage {stage_m[bi]})", flush=True)

    # --- 3-panel video ---
    P = 256
    def L(img, t): return label(cv2.resize(img, (P, P)), t)
    def compose(f):
        k = int(np.searchsorted(kt, f, side="right") - 1); k = max(0, min(k, len(kt) - 1))
        p0 = L(frames[f], f"vis_base real | milestone {stage_m[stage_of[k]]}")
        p1 = L(d_l1[k], "PRED-fwd -> dec_v2 (L1, blurry)")
        p2 = L(d_fx[k], "PRED-fwd -> flow FIXED-noise (sharp)")
        gap = np.full((p0.shape[0], 8, 3), 20, np.uint8)
        return np.ascontiguousarray(np.hstack([p0, gap, p1, gap, p2]))
    c0 = compose(0); Hc, Wc = c0.shape[:2]; Hc -= Hc % 2; Wc -= Wc % 2
    vw = cv2.VideoWriter(str(args.out_vid), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (Wc, Hc))
    assert vw.isOpened(), f"VideoWriter failed ({Wc}x{Hc})"
    for f in range(N):
        vw.write(cv2.cvtColor(compose(f)[:Hc, :Wc], cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out_vid} | {N} frames @ {out_fps}fps | 3-panel real|dec_v2|flow-fixed", flush=True)


if __name__ == "__main__":
    main()
