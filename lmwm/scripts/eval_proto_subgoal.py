#!/usr/bin/env python
"""Evaluate the proto/subgoal head against the episode-medoid target.

On the held-out episode split, for the greedy_proto head prediction:
  - cos to the true next-stage episode medoid  (the continuous target that matters)
  - cos to the global centroid of the next milestone (reference)
  - retrieval top1: nearest of the 37 centroids to the predicted latent == future milestone

Run on both the centroid-trained and medoid-trained checkpoints (same medoid pairs
file) for a clean A/B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path, help="medoid-augmented pairs (has next_medoid)")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck["config"]; meta = ck.get("meta", {})
    g = np.load(args.graph_npz)
    proto = g["prototype_table"].astype(np.float32)
    proton = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    num_m, latent_dim = proto.shape

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, val_idx = split_indices(z, n, float(cfg["training"].get("val_ratio", 0.2)),
                               int(cfg.get("seed", 2026)), torch.device("cpu"), str(cfg.get("split_mode", "random")))
    vi = val_idx.numpy()
    feats = z["current"][vi].astype(np.float32)
    fut_m = z["future_milestone"][vi].astype(np.int64)
    med = z["next_medoid"][vi].astype(np.float32)
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    # drop rows with no next stage (all-zero medoid)
    ok = np.linalg.norm(z["next_medoid"][vi].astype(np.float32), axis=1) > 1e-6
    feats, fut_m, med = feats[ok], fut_m[ok], med[ok]

    in_dim = int(meta.get("input_dim", feats.shape[1]))
    mc = cfg.get("model", {})
    model = UnifiedLMWM(in_dim, latent_dim, num_m, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(feats), 8192):
            preds.append(model(torch.from_numpy(feats[s:s + 8192]).to(device))["greedy_proto"].cpu().numpy())
    pred = np.concatenate(preds)
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)

    cos_medoid = float((pred * med).sum(1).mean())
    cos_centroid = float((pred * proton[fut_m]).sum(1).mean())
    retr = pred @ proton.T  # (n,37)
    retr_top1 = float((retr.argmax(1) == fut_m).mean())

    summary = {
        "checkpoint": str(args.checkpoint),
        "proto_target_source_trained": meta.get("proto_target_source", "centroid"),
        "held_out_rows": int(len(feats)),
        "greedy_proto_cos_to_true_next_medoid": cos_medoid,
        "greedy_proto_cos_to_next_centroid": cos_centroid,
        "retrieval_top1_nearest_centroid_eq_future_m": retr_top1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
