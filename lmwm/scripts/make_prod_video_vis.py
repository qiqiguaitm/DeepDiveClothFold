#!/usr/bin/env python
"""Cross-dataset milestone+1 prediction video on vis_base @ 30Hz (zero re-training).

CRAVE (frozen recipe) assigns each vis frame to a kai0 milestone via prototype cosine +
Viterbi transition prior -> stages -> per-stage medoid -> TRUE milestone+1.
The prod LMWM (trained on kai0, generalizes) predicts milestone+1 from the current obs.
Both next-milestone latents are decoded by the flow decoder (dec_best).

Layout: left real (native) | right-top PRED milestone+1 -> flow | right-bottom TRUE -> flow.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_prod_milestone import ProdNet  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from decode_best import load_best_decoder  # noqa: E402
from make_episode_native_video import label  # noqa: E402


def viterbi_assign(latn, proto_n, trans, beta, stay):
    """Frame-level Viterbi. emission = beta*(1-cos); transition = self-loop `stay` + advance
    dist from the (self-loop-free) recurrence graph. transition_probs has ~0 diagonal (it is a
    next-UNIQUE-milestone graph), so a self-loop prior is required or every frame switches."""
    K, M = len(latn), len(proto_n)
    adv = trans.copy(); np.fill_diagonal(adv, 0.0)
    rs = adv.sum(1, keepdims=True); adv = np.divide(adv, rs, out=np.zeros_like(adv), where=rs > 0)
    T = (1.0 - stay) * adv; T[np.diag_indices(M)] += stay          # frame-level transition matrix
    emit = beta * (1.0 - latn @ proto_n.T)                          # (K,M) scaled cosine distance
    pen = -np.log(T + 1e-12)                                        # (M,M) transition penalty
    cost = emit[0].copy(); bp = np.zeros((K, M), int)
    for j in range(1, K):
        tr = cost[:, None] + pen                                    # from m (row) -> m' (col)
        k = tr.argmin(0); cost = emit[j] + tr[k, np.arange(M)]; bp[j] = k
    ms = np.zeros(K, int); ms[-1] = int(cost.argmin())
    for j in range(K - 2, -1, -1):
        ms[j] = bp[j + 1, ms[j + 1]]
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="explicit episode mp4; else first under --root")
    ap.add_argument("--root", default="kai0/data/Task_A/vis_base/v4/2026-04-23-v4")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_v2.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone_v2/member_*.pt")
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--beta", type=float, default=30.0, help="emission scale (higher -> more stages)")
    ap.add_argument("--stay", type=float, default=0.9, help="self-loop prior (higher -> fewer stages)")
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--out", default="lmwm/docs/assets/visbase_milestone_pred_flow.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    g = np.load(args.graph_npz)
    proto_raw = g["prototype_table"].astype(np.float32)                       # for prod build_feat
    proto_n = proto_raw / (np.linalg.norm(proto_raw, axis=1, keepdims=True) + 1e-8)
    trans = g["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)

    vids = ([args.video] if args.video else
            sorted(glob.glob(str(Path(args.root) / f"videos/chunk-*/{args.camera}/episode_*.mp4")))
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
    N = len(frames); kt = np.arange(0, N, args.stride)                        # keyframe times
    print(f"{N} frames, {len(kt)} keyframes @ stride {args.stride}", flush=True)

    enc = load_encoder("dinov3-h", device=str(dev))
    key_small = np.stack([cv2.resize(frames[t], (256, 256)) for t in kt])
    lat = enc.encode_pooled(key_small).astype(np.float32)
    latn = lat / (np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8)

    # CRAVE assignment (Viterbi) -> stages -> per-stage medoid -> TRUE next-milestone
    ms = viterbi_assign(latn, proto_n, trans, args.beta, args.stay)
    ch = np.where(np.diff(ms) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
    stage_m = [int(ms[s]) for s in st]
    medoid = []                                                              # medoid latent per stage
    for s, e in zip(st, en):
        rel = int((latn[s:e] @ proto_n[ms[s]]).argmax()); medoid.append(latn[s + rel])
    stage_of = np.zeros(len(kt), int)
    for si, (s, e) in enumerate(zip(st, en)):
        stage_of[s:e] = si
    true_next = np.stack([medoid[min(stage_of[k] + 1, len(medoid) - 1)] for k in range(len(kt))])

    # prod LMWM predicted milestone+1 (feat = [pooled | prev-proto | cur-proto | state14])
    prev_lat = np.zeros((len(kt), 1280), np.float32); cur_lat = np.zeros((len(kt), 1280), np.float32)
    for k in range(len(kt)):
        si = stage_of[k]
        cur_lat[k] = proto_raw[stage_m[si]]
        if si > 0:
            prev_lat[k] = proto_raw[stage_m[si - 1]]
    feat = np.concatenate([latn, prev_lat, cur_lat, np.zeros((len(kt), 14), np.float32)], 1).astype(np.float32)
    Xf = torch.from_numpy(feat).to(dev)
    pred = None
    for p in sorted(glob.glob(args.members)):
        c = torch.load(p, map_location="cpu"); m = ProdNet(c["din"], c["num_m"]).to(dev)
        m.load_state_dict(c["model"]); m.eval()
        with torch.no_grad():
            _, pr = m(Xf)
        gg = F.normalize(pr.float(), -1).cpu().numpy(); pred = gg if pred is None else pred + gg
    pred /= (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)

    # flow-decode PRED + TRUE at keyframes, hold across native frames
    flow = load_best_decoder(args.flow, str(dev))
    pred_dec = np.concatenate([flow(pred[s:s + 64]) for s in range(0, len(kt), 64)])
    true_dec = np.concatenate([flow(true_next[s:s + 64]) for s in range(0, len(kt), 64)])
    print("decoded PRED + TRUE (flow)", flush=True)

    R = flow.res; BIG = R * 2; RW = 230
    def rlabel(img, t): return label(cv2.resize(img, (RW, RW)), t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def compose(f):
        k = int(np.searchsorted(kt, f, side="right") - 1); k = max(0, min(k, len(kt) - 1))
        left = label(cv2.resize(frames[f], (BIG, BIG)), f"vis_base real (native) frame {f} | milestone {stage_m[stage_of[k]]}")
        rt = rlabel(pred_dec[k], "PRED milestone+1 -> flow decode")
        rb = rlabel(true_dec[k], "TRUE milestone+1 -> flow decode")
        right = cv2.resize(np.vstack([rt, rb]), (RW, left.shape[0]))
        return np.ascontiguousarray(np.hstack([left, np.full((left.shape[0], 8, 3), 20, np.uint8), right]))
    c0 = compose(0); Hc, Wc = c0.shape[:2]
    Hc -= Hc % 2; Wc -= Wc % 2                                     # mp4v needs even dims
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (Wc, Hc))
    assert vw.isOpened(), f"VideoWriter failed to open ({Wc}x{Hc} @ {out_fps})"
    for f in range(N):
        vw.write(cv2.cvtColor(compose(f)[:Hc, :Wc], cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out} | {N} frames @ {out_fps}fps | {len(st)} stages | vis_base cross-dataset milestone+1 (CRAVE true + LMWM pred, flow decode)", flush=True)


if __name__ == "__main__":
    main()
