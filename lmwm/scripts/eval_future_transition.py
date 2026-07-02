#!/usr/bin/env python
"""Evaluate LMWM predictions against actual exported future transitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.runtime import UnifiedLMWMPredictor  # noqa: E402


def topk_hit(probs: np.ndarray, target: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    pred = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((pred == target[:, None]).any(axis=1).mean())


def mean_proto_cos(proto_table: np.ndarray, pred_m: np.ndarray, target_m: np.ndarray) -> float:
    pred = proto_table[pred_m]
    target = proto_table[target_m]
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return float((pred * target).sum(axis=1).mean())


def summarize_predictions(pred: dict[str, np.ndarray], z: dict[str, np.ndarray], proto_table: np.ndarray) -> dict[str, float]:
    future_m = z["future_milestone"].astype(np.int64)
    current_m = z["current_milestone"].astype(np.int64)
    pord = z["pord"].astype(np.float32)
    progress_future = z["progress_future"].astype(np.float32) if "progress_future" in z else pord[future_m]
    progress_current = z["progress_t"].astype(np.float32) if "progress_t" in z else pord[current_m]

    rows: dict[str, float] = {}
    transition_probs = pred["transition_probs"]
    rows["transition_top1_vs_future"] = topk_hit(transition_probs, future_m, 1)
    rows["transition_top3_vs_future"] = topk_hit(transition_probs, future_m, 3)
    rows["transition_top5_vs_future"] = topk_hit(transition_probs, future_m, 5)
    rows["transition_future_nll"] = float(-np.log(np.take_along_axis(transition_probs, future_m[:, None], axis=1)[:, 0] + 1e-12).mean())

    for name in ["neural_greedy", "neural_max_product", "hybrid_greedy", "hybrid_max_product", "graph_greedy", "graph_max_product"]:
        pred_m = pred[name].astype(np.int64)
        pred_progress = pord[pred_m]
        rows[f"{name}_top1_vs_future"] = float((pred_m == future_m).mean())
        rows[f"{name}_proto_cos_vs_future"] = mean_proto_cos(proto_table, pred_m, future_m)
        rows[f"{name}_progress_mae_vs_future"] = float(np.abs(pred_progress - progress_future).mean())
        rows[f"{name}_progress_delta_mae"] = float(
            np.abs((pred_progress - progress_current) - (progress_future - progress_current)).mean()
        )
        rows[f"{name}_progress_direction_match"] = float(
            (np.sign(pred_progress - progress_current) == np.sign(progress_future - progress_current)).mean()
        )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset_npz", type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    predictor = UnifiedLMWMPredictor.from_yaml(args.config)
    dataset_npz = args.dataset_npz or predictor.config.checkpoint.parent / "unused"
    if args.dataset_npz is None:
        import yaml

        with args.config.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        dataset_npz = Path(raw["dataset_npz"])
    z = np.load(dataset_npz)
    n_total = int(z["current"].shape[0])
    n = min(n_total, args.max_samples) if args.max_samples > 0 else n_total
    chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, n, args.batch_size):
        end = min(n, start + args.batch_size)
        pred = predictor.predict(
            z["current"][start:end].astype(np.float32),
            z["current_milestone"][start:end].astype(np.int64),
        )
        for key, value in pred.items():
            chunks.setdefault(key, []).append(value)
    pred_all = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    z_slice = {key: z[key][:n] for key in z.files}
    summary = summarize_predictions(pred_all, z_slice, predictor.prototype_table_np)
    summary.update(
        {
            "config": str(args.config),
            "dataset_npz": str(dataset_npz),
            "num_samples": int(n),
            "metric_note": "future metrics compare predictions with exported actual future_milestone/progress_future, not only graph-table labels",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
