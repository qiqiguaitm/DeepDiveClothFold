#!/usr/bin/env python
"""Decode LMWM's predicted milestones into images with the DINOv3-H decoder.

Produces two figures for the LMWM technical report:

1. prototype_gallery.png -- decode all 37 DINOv3-H milestone prototypes (sorted by
   progress) and show the medoid (nearest real frame) beside each. Validates the
   decoder and shows the milestone vocabulary.
2. prediction_filmstrip.png -- for a held-out episode, sampled timesteps as rows;
   columns = [current frame | LMWM subgoal-latent decoded | LMWM pred-milestone
   prototype decoded | true next-milestone prototype decoded | actual future frame].
   Green/red border = predicted next milestone correct/wrong vs the real future.

Usage:
    python lmwm/scripts/visualize_lmwm_decode.py \
        --decoder lmwm/checkpoints/dinov3h_decoder/dec.pt \
        --vla_config lmwm/configs/inference/kai0base_dinov3h_vla_realfuture.yaml \
        --pairs lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz \
        --feature_dir temp/crave_full_dinov3h \
        --dataset_root kai0/data/Task_A/kai0_base \
        --out_dir lmwm/docs/assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from lmwm.data import split_indices  # noqa: E402
from lmwm.vla_interface import VLALMWMPredictor  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, type=Path)
    ap.add_argument("--vla_config", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.decoder, map_location="cpu")
    R = int(ck["res"])
    dec = PooledDecoder(din=int(ck["din"]), res=R).to(device)
    dec.load_state_dict(ck["model"])
    dec.eval()

    def decode(latents: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(l2(np.atleast_2d(latents).astype(np.float32))).to(device)
        with torch.no_grad():
            o = dec(x).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    g = np.load(args.graph_npz)
    proto = g["prototype_table"].astype(np.float32)  # (37, 1280), l2
    pord = g["pord"].astype(np.float32)
    num_m = len(proto)

    E, FR, F = load_features(args.feature_dir)
    Fn = l2(F.astype(np.float32))
    # per-prototype medoid: nearest frame by cosine
    medoid_g = np.array([(Fn @ proto[k]).argmax() for k in range(num_m)], dtype=np.int64)

    chunks_size = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    _caps: dict[int, cv2.VideoCapture] = {}

    def frame(ep: int, fr: int) -> np.ndarray:
        if ep not in _caps:
            mp4 = args.dataset_root / f"videos/chunk-{ep // chunks_size:03d}/{args.camera}/episode_{ep:06d}.mp4"
            _caps[ep] = cv2.VideoCapture(str(mp4))
        cap = _caps[ep]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, im = cap.read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if ok else np.zeros((R, R, 3), np.uint8)

    # ---- Figure 1: prototype gallery (decoded + medoid), sorted by progress ----
    order = np.argsort(pord)
    proto_dec = decode(proto[order])
    cols = 10
    rows = int(np.ceil(num_m / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.4, rows * 3.0))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            axd = axes[r * 2, c]
            axm = axes[r * 2 + 1, c]
            for a in (axd, axm):
                a.axis("off")
            if i >= num_m:
                continue
            k = int(order[i])
            axd.imshow(proto_dec[i]); axd.set_title(f"m{k}  P={pord[k]:.2f}", fontsize=6)
            mg = medoid_g[k]
            axm.imshow(frame(int(E[mg]), int(FR[mg])))
            if c == 0:
                axd.set_ylabel("decoded", fontsize=7); axm.set_ylabel("medoid", fontsize=7)
    fig.suptitle("DINOv3-H milestone prototypes: decoded (top) vs medoid nearest frame (bottom), sorted by progress", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out_dir / "prototype_gallery.png", dpi=110); plt.close(fig)
    print(f"saved {args.out_dir / 'prototype_gallery.png'}", flush=True)

    # ---- Figure 2: prediction filmstrip on a held-out episode ----
    predictor = VLALMWMPredictor.from_yaml(str(args.vla_config))
    z = np.load(args.pairs)
    cfg = torch.load(predictor.config.checkpoint, map_location="cpu")["config"]
    n = len(z["current_milestone"])
    _, val_idx = split_indices(z, n, float(cfg["training"].get("val_ratio", 0.2)),
                               int(cfg.get("seed", 2026)), torch.device("cpu"), str(cfg.get("split_mode", "random")))
    vi = val_idx.numpy()
    eps = z["episode_id"][vi]
    # pick a held-out episode with many pairs
    uep, cnt = np.unique(eps, return_counts=True)
    chosen_ep = int(uep[cnt.argmax()])
    rows_idx = vi[eps == chosen_ep]
    # sort by current progress and sample ~6 across the episode
    cur_m_all = z["current_milestone"][rows_idx]
    ordv = np.argsort(pord[cur_m_all])
    rows_idx = rows_idx[ordv]
    pick = rows_idx[np.linspace(0, len(rows_idx) - 1, min(6, len(rows_idx))).astype(int)]

    feats = z["current"][pick].astype(np.float32)
    out = predictor.predict(feats)
    ncol = 5
    fig, axes = plt.subplots(len(pick), ncol, figsize=(ncol * 2.0, len(pick) * 2.1))
    if len(pick) == 1:
        axes = axes[None, :]
    col_titles = ["current frame", "pred subgoal (decoded)", "pred milestone proto", "true milestone proto", "actual future frame"]
    for ri, row in enumerate(pick):
        ep = int(z["episode_id"][row]); t = int(z["t"][row]); ft = int(z["future_t"][row])
        pred_m = int(out["next_milestone"][ri]); true_m = int(z["future_milestone"][row])
        cur_m = int(out["current_milestone"][ri]); conf = float(out["confidence"][ri])
        imgs = [
            frame(ep, t),
            decode(out["subgoal_latent"][ri])[0],
            decode(proto[pred_m])[0],
            decode(proto[true_m])[0],
            frame(ep, ft),
        ]
        correct = pred_m == true_m
        for ci in range(ncol):
            a = axes[ri, ci]; a.imshow(imgs[ci]); a.set_xticks([]); a.set_yticks([])
            if ri == 0:
                a.set_title(col_titles[ci], fontsize=8)
        # color pred columns
        for ci in (1, 2):
            for s in axes[ri, ci].spines.values():
                s.set_color("#2ca02c" if correct else "#d62728"); s.set_linewidth(2.5)
        axes[ri, 0].set_ylabel(f"m{cur_m}->\npred m{pred_m} {'OK' if correct else 'x m'+str(true_m)}\nconf {conf:.2f}", fontsize=7)
    fig.suptitle(f"LMWM next-milestone prediction decoded vs reality  (held-out ep{chosen_ep})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out_dir / "prediction_filmstrip.png", dpi=115); plt.close(fig)
    print(f"saved {args.out_dir / 'prediction_filmstrip.png'}  (ep{chosen_ep}, {len(pick)} rows)", flush=True)

    for c in _caps.values():
        c.release()


if __name__ == "__main__":
    main()
