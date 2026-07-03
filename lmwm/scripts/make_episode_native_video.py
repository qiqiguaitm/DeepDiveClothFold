#!/usr/bin/env python
"""Native-speed (stride-1, 30Hz) episode video: prediction computed for EVERY real
frame, so playback is normal speed and both panels are smooth.

Rebuilds the augin input per real frame = [pooled DINOv3-H 1280; prev-milestone
one-hot 38; z-scored proprio 14], runs the on-manifold grid predictor -> patch
decode. Layout: left real frame | right-top PRED grid->decode | right-bottom TRUE
next-medoid enc->decode (ceiling).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402
from track_b2_grid_predict import GridGen  # noqa: E402


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def state_stats(pairs_npz, dataset_root, cs, n_sample=40):
    z = np.load(pairs_npz)
    eps = np.unique(z["episode_id"])
    rng = np.random.default_rng(0)
    pick = rng.choice(eps, min(n_sample, len(eps)), replace=False)
    vals = []
    for ep in pick:
        pq = dataset_root / f"data/chunk-{int(ep) // cs:03d}/episode_{int(ep):06d}.parquet"
        if pq.exists():
            arr = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
            vals.append(arr)
    v = np.concatenate(vals)
    return v.mean(0), v.std(0) + 1e-6


def label(img, text):
    out = cv2.copyMakeBorder(img, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1819)
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--patch_dec", default="lmwm/checkpoints/patch_decoder/patch_dec.pt", type=Path)
    ap.add_argument("--grid_pred", default="lmwm/checkpoints/grid_predictor/G_lawm.pt", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--enc_res", type=int, default=256)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--out", default="lmwm/docs/assets/ep_native_video.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    R = args.res

    ck = torch.load(args.patch_dec, map_location="cpu", weights_only=False)
    din_grid = int(ck["din"])
    D = make_decoder(din_grid, ck["dec"]).to(dev); D.load_state_dict(ck["model"]); D.eval()
    for p in D.parameters():
        p.requires_grad_(False)
    muT = torch.from_numpy(ck["mu"]).view(1, din_grid, 1, 1).to(dev)
    sdT = torch.from_numpy(ck["sd"]).view(1, din_grid, 1, 1).to(dev)
    def decode(grid): return D((grid - muT) / sdT)
    def d2img(t): return np.clip((t.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32); num_m = len(proto); START = num_m
    enc = load_encoder("dinov3-h", device=str(dev))
    Gd = torch.load(args.grid_pred, map_location="cpu", weights_only=False)
    din_in = 1280 + (num_m + 1) + 14
    G = GridGen(din_in, din_grid).to(dev); G.load_state_dict(Gd["model"]); G.eval()

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    vid = args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/{args.camera}/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS); out_fps = round(fps) if fps and fps > 1 else 30
    enc_frames, disp_frames = [], []
    while len(enc_frames) < args.max_frames:
        okf, im = cap.read()
        if not okf:
            break
        rgb = im[:, :, ::-1]
        enc_frames.append(cv2.resize(rgb, (args.enc_res, args.enc_res)))
        disp_frames.append(cv2.resize(rgb, (R * 2, R * 2)))
    cap.release()
    N = len(enc_frames)
    print(f"ep{args.episode}: {N} frames @ native {out_fps}fps", flush=True)
    enc_frames = np.stack(enc_frames)

    print("encoding pooled + grid per frame ...", flush=True)
    pooled = enc.encode_pooled(enc_frames).astype(np.float32)          # (N,1280)
    seq = (l2(pooled) @ proto.T).argmax(1)                              # milestone per frame
    ch = np.where(np.diff(seq) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [N]])
    fr2s = np.zeros(N, np.int64); stage_m = []
    for si, (s, e) in enumerate(zip(st, en)):
        fr2s[s:e] = si; stage_m.append(int(seq[s]))
    prev_oh = np.zeros((N, num_m + 1), np.float32)
    for f in range(N):
        si = fr2s[f]
        prev_oh[f, stage_m[si - 1] if si >= 1 else START] = 1.0

    pq = args.dataset_root / f"data/chunk-{args.episode // cs:03d}/episode_{args.episode:06d}.parquet"
    arr = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    mu_s, sd_s = state_stats(args.pairs, args.dataset_root, cs)
    state = (arr[np.clip(np.arange(N), 0, len(arr) - 1)] - mu_s) / sd_s

    augin = np.concatenate([pooled, prev_oh, state.astype(np.float32)], axis=1)  # (N,1332)

    # PRED per frame
    pred_imgs = []
    with torch.no_grad():
        for s in range(0, N, 256):
            pred_imgs.extend(d2img(p) for p in decode(G(torch.from_numpy(augin[s:s + 256]).to(dev))))

    # CEILING: per stage, medoid frame = frame in stage most similar to its milestone proto;
    # ceiling for stage si = decode(encode_grid(medoid of NEXT stage)). Last stage holds itself.
    Pn = l2(pooled)
    stage_medoid = []
    for si, (s, e) in enumerate(zip(st, en)):
        rel = (Pn[s:e] @ proto[stage_m[si]]).argmax()
        stage_medoid.append(s + int(rel))
    ceil_src = [stage_medoid[min(fr2s[f] + 1, len(st) - 1)] for f in range(N)]  # frame idx of target medoid
    uniq = sorted(set(ceil_src)); u2i = {u: j for j, u in enumerate(uniq)}
    med_grids = enc.encode_grid(enc_frames[uniq]).astype(np.float32)
    with torch.no_grad():
        ceil_dec = decode(torch.from_numpy(med_grids).to(dev))
    ceil_imgs = [d2img(ceil_dec[j]) for j in range(len(uniq))]

    BIG = R * 2; RW = 230; W = BIG + 8 + RW; H = 26 + BIG + 4
    def rlabel(img, text): return label(cv2.resize(img, (RW, RW)), text)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), fourcc, out_fps, (W, H))
    for f in range(N):
        left = label(disp_frames[f], f"ep{args.episode} frame {f} (real, native)")
        right = np.vstack([rlabel(pred_imgs[f], "PRED grid -> decode"),
                           rlabel(ceil_imgs[u2i[ceil_src[f]]], "TRUE medoid enc->decode (ceiling)")])
        right = cv2.resize(right, (RW, left.shape[0]))
        canvas = np.hstack([left, np.full((left.shape[0], 8, 3), 20, np.uint8), right])
        vw.write(cv2.cvtColor(canvas[:H, :W], cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out} | {N} frames @ {out_fps}fps (native speed, stride-1)")


if __name__ == "__main__":
    main()
