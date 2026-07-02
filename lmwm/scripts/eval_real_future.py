#!/usr/bin/env python
"""Evaluate an LMWM checkpoint against the REAL observed next milestone.

This is the honest, graph-independent criterion for Phase A. Existing LMWM
metrics compare the neural greedy head to ``greedy_next[current_m]`` -- the same
graph table that generated the training labels -- which is circular. Here we
score predictions against ``future_milestone`` (the actually observed next-unique
milestone) on the held-out episode split, and report non-neural baselines for
calibration.

Usage:
    python lmwm/scripts/eval_real_future.py \
        --checkpoint lmwm/checkpoints/stage3_unified/<run>/best.pt \
        --output_dir lmwm/outputs/real_future_eval/<name> [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402


def topk_acc(probs: np.ndarray, target: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=min(k, probs.shape[1] - 1), axis=1)[:, :k]
    return float((topk == target[:, None]).any(axis=1).mean())


def nll(probs: np.ndarray, target: np.ndarray) -> float:
    p = probs[np.arange(len(target)), target]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def dist_metrics(probs: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "top1": topk_acc(probs, target, 1),
        "top3": topk_acc(probs, target, 3),
        "top5": topk_acc(probs, target, 5),
        "nll": nll(probs, target),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--device", default=None, help="override device (e.g. cpu, cuda:0)")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck["config"]
    meta = ck.get("meta", {})
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    z = np.load(cfg["dataset_npz"])
    g = np.load(cfg["graph_npz"])
    n = len(z["current_milestone"])
    seed = int(cfg.get("seed", 2026))
    val_ratio = float(cfg["training"].get("val_ratio", 0.2))
    split_mode = str(cfg.get("split_mode", "random"))
    _, val_idx = split_indices(z, n, val_ratio, seed, device, split_mode)
    vi = val_idx.cpu().numpy()

    current_m = z["current_milestone"].astype(np.int64)[vi]
    future_m = z["future_milestone"].astype(np.int64)[vi]
    feats = z["current"].astype(np.float32)[vi]
    transition_probs = g["transition_probs"].astype(np.float32)
    greedy_next = g["greedy_next"].astype(np.int64)
    max_product_next = g["max_product_next"].astype(np.int64)
    num_m = int(transition_probs.shape[0])

    # ---- neural predictions ----
    latent_dim = int(meta.get("latent_dim", g["prototype_table"].shape[1]))
    in_dim = int(meta.get("input_dim", feats.shape[1]))
    mc = cfg.get("model", {})
    model = UnifiedLMWM(in_dim, latent_dim, num_m, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    greedy_probs_list, trans_probs_list = [], []
    with torch.no_grad():
        for s in range(0, len(feats), 8192):
            x = torch.from_numpy(feats[s:s + 8192]).to(device)
            out = model(x)
            greedy_probs_list.append(F.softmax(out["greedy_logits"], dim=-1).cpu().numpy())
            trans_probs_list.append(F.softmax(out["transition_logits"], dim=-1).cpu().numpy())
    greedy_probs = np.concatenate(greedy_probs_list)
    trans_probs = np.concatenate(trans_probs_list)

    # ---- non-neural baselines ----
    uniform = np.full((len(vi), num_m), 1.0 / num_m, dtype=np.float32)
    empirical = transition_probs[current_m]  # empirical P(next | current milestone)

    summary = {
        "checkpoint": str(args.checkpoint),
        "label_source_trained": meta.get("label_source", "graph_lookup"),
        "held_out_pairs": int(len(vi)),
        "num_milestones": num_m,
        "held_out_episodes": int(len(np.unique(z["episode_id"][vi]))) if "episode_id" in z.files else None,
        # circular metric (for contrast): neural greedy vs the graph table it was compared to before
        "circular_vs_graph": {
            "neural_greedy_top1_vs_graph_lookup": float((greedy_probs.argmax(1) == greedy_next[current_m]).mean()),
        },
        # honest metric: everything vs the REAL observed future
        "vs_real_future": {
            "baseline_uniform": dist_metrics(uniform, future_m),
            "baseline_graph_argmax_greedy_top1": float((greedy_next[current_m] == future_m).mean()),
            "baseline_graph_argmax_maxproduct_top1": float((max_product_next[current_m] == future_m).mean()),
            "baseline_empirical_dist": dist_metrics(empirical, future_m),
            "neural_greedy_head": dist_metrics(greedy_probs, future_m),
            "neural_transition_head": dist_metrics(trans_probs, future_m),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
