#!/usr/bin/env python
"""Build a CRAVE recurrence latent state probability graph from DINOv3-H features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import load_config  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_full_features(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.load(feature_dir / "index.npz")
    e = idx["E"].astype(np.int64)
    fr = idx["FR"].astype(np.int64)
    n = int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gidx = z["gidx"].astype(np.int64)
        feat[gidx] = z["feat"]
        valid[gidx] = z["valid"].astype(bool)
    return e[valid], fr[valid], l2(feat[valid].astype(np.float32))


def assign_milestones(features: np.ndarray, proto: np.ndarray, chunk: int = 32768) -> np.ndarray:
    assign = np.empty(len(features), dtype=np.int64)
    for i in range(0, len(features), chunk):
        sim = features[i:i + chunk] @ proto.T
        assign[i:i + chunk] = sim.argmax(axis=1)
    return assign


def compress_runs(ms: np.ndarray) -> np.ndarray:
    if len(ms) == 0:
        return ms
    keep = np.ones(len(ms), dtype=bool)
    keep[1:] = ms[1:] != ms[:-1]
    return ms[keep]


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

    feature_dir = Path(cfg["feature_dir"])
    milestone_file = Path(cfg["milestone_file"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg.get("smoothing_alpha", 1e-3))
    horizon = int(cfg.get("max_product_horizon", 10))

    E, FR, F = load_full_features(feature_dir)
    mz = np.load(milestone_file)
    proto = l2(mz["C"].astype(np.float32))
    pord = mz["Pord"].astype(np.float32)
    assign = assign_milestones(F, proto)
    n_m = len(proto)

    counts = np.zeros((n_m, n_m), dtype=np.int64)
    starts = np.zeros(n_m, dtype=np.int64)
    terminals = np.zeros(n_m, dtype=np.int64)
    stage_len_sum = np.zeros(n_m, dtype=np.int64)
    stage_visit_count = np.zeros(n_m, dtype=np.int64)
    compressed_episode_lengths = []

    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order = loc[np.argsort(FR[loc])]
        ms = assign[order]
        if len(ms) == 0:
            continue
        unique_ms = compress_runs(ms)
        starts[unique_ms[0]] += 1
        terminals[unique_ms[-1]] += 1
        compressed_episode_lengths.append(int(len(unique_ms)))
        for a, b in zip(unique_ms[:-1], unique_ms[1:]):
            counts[a, b] += 1
        run_start = 0
        for i in range(1, len(ms) + 1):
            if i == len(ms) or ms[i] != ms[run_start]:
                stage = ms[run_start]
                stage_len_sum[stage] += i - run_start
                stage_visit_count[stage] += 1
                run_start = i

    probs = smooth_rows(counts, alpha=alpha)
    greedy_next = probs.argmax(axis=1).astype(np.int64)
    terminal_target = int(cfg.get("terminal_target", int(np.argmax(pord))))
    max_product_next, max_product_prob, dp_table = max_product_policy(probs, terminal_target, horizon)

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
        stage_len_sum=stage_len_sum,
        stage_visit_count=stage_visit_count,
        terminal_target=np.array(terminal_target, dtype=np.int64),
        horizon=np.array(horizon, dtype=np.int64),
    )
    top_edges = []
    for a in range(n_m):
        order = np.argsort(-probs[a])[:5]
        for b in order:
            if counts[a, b] > 0:
                top_edges.append({
                    "from": int(a),
                    "to": int(b),
                    "count": int(counts[a, b]),
                    "prob": float(probs[a, b]),
                })
    top_edges.sort(key=lambda x: x["count"], reverse=True)
    meta = {
        "name": cfg.get("name", "recurrence_graph"),
        "feature_dir": str(feature_dir),
        "milestone_file": str(milestone_file),
        "output_npz": str(out_npz),
        "num_valid_frames": int(len(F)),
        "num_episodes": int(len(np.unique(E))),
        "num_milestones": int(n_m),
        "smoothing_alpha": alpha,
        "terminal_target": terminal_target,
        "terminal_target_progress": float(pord[terminal_target]),
        "max_product_horizon": horizon,
        "mean_compressed_episode_length": float(np.mean(compressed_episode_lengths)),
        "median_compressed_episode_length": float(np.median(compressed_episode_lengths)),
        "nonzero_edges": int((counts > 0).sum()),
        "top_edges_by_count": top_edges[:20],
    }
    (out_dir / "recurrence_graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
