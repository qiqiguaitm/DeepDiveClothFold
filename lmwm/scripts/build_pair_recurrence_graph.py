#!/usr/bin/env python
"""Build a recurrence graph directly from exported LMWM pair datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import load_config  # noqa: E402


def smooth_rows(counts: np.ndarray, alpha: float) -> np.ndarray:
    probs = counts.astype(np.float64) + alpha
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def max_product_policy(probs: np.ndarray, target: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = probs.shape[0]
    dp = np.zeros((horizon + 1, n), dtype=np.float64)
    policy = np.full((horizon + 1, n), -1, dtype=np.int64)
    dp[0, target] = 1.0
    for h in range(1, horizon + 1):
        score = probs * dp[h - 1][None, :]
        policy[h] = score.argmax(axis=1)
        dp[h] = score.max(axis=1)
    return policy[horizon], dp[horizon], dp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)

    dataset_npz = Path(cfg["dataset_npz"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg.get("smoothing_alpha", 1e-3))
    horizon = int(cfg.get("max_product_horizon", 10))

    z = np.load(dataset_npz)
    current_m = z["current_milestone"].astype(np.int64)
    future_m = z["future_milestone"].astype(np.int64)
    proto = z["prototype_table"].astype(np.float32)
    pord = z["pord"].astype(np.float32)
    n_m = int(proto.shape[0])

    counts = np.zeros((n_m, n_m), dtype=np.int64)
    np.add.at(counts, (current_m, future_m), 1)
    probs = smooth_rows(counts, alpha)
    greedy_next = probs.argmax(axis=1).astype(np.int64)
    terminal_target = int(cfg.get("terminal_target", int(np.argmax(pord))))
    max_product_next, max_product_prob, dp_table = max_product_policy(probs, terminal_target, horizon)

    starts = np.bincount(current_m, minlength=n_m).astype(np.int64)
    terminals = np.bincount(future_m, minlength=n_m).astype(np.int64)
    out_npz = out_dir / "recurrence_graph.npz"
    np.savez_compressed(
        out_npz,
        transition_counts=counts,
        transition_probs=probs.astype(np.float32),
        greedy_next=greedy_next,
        max_product_next=max_product_next,
        max_product_prob=max_product_prob.astype(np.float32),
        max_product_dp=dp_table.astype(np.float32),
        starts=starts,
        terminals=terminals,
        pord=pord,
        prototype_table=proto.astype(np.float32),
        terminal_target=np.array(terminal_target, dtype=np.int64),
        horizon=np.array(horizon, dtype=np.int64),
    )

    top_edges = []
    nz_from, nz_to = np.where(counts > 0)
    for a, b in zip(nz_from.tolist(), nz_to.tolist()):
        top_edges.append({"from": int(a), "to": int(b), "count": int(counts[a, b]), "prob": float(probs[a, b])})
    top_edges.sort(key=lambda x: x["count"], reverse=True)
    ep = z["episode_id"].astype(np.int64) if "episode_id" in z.files else np.array([], dtype=np.int64)
    meta = {
        "name": cfg.get("name", out_dir.name),
        "dataset_npz": str(dataset_npz),
        "output_npz": str(out_npz),
        "num_pairs": int(len(current_m)),
        "num_episodes": int(len(np.unique(ep))) if len(ep) else None,
        "num_milestones": int(n_m),
        "feature_dim": int(proto.shape[1]),
        "smoothing_alpha": alpha,
        "terminal_target": terminal_target,
        "terminal_target_progress": float(pord[terminal_target]),
        "max_product_horizon": horizon,
        "nonzero_edges": int((counts > 0).sum()),
        "top_edges_by_count": top_edges[:20],
        "graph_source": "exported_pair_current_future_milestone_counts",
    }
    (out_dir / "recurrence_graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
