#!/usr/bin/env python
"""Evaluate a runtime LMWM policy on a full exported dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.runtime import UnifiedLMWMPredictor  # noqa: E402


def concat_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted(chunks[0].keys())
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset_npz", type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    dataset_npz = args.dataset_npz or Path(cfg["dataset_npz"])
    z = np.load(dataset_npz)
    n_total = int(z["current"].shape[0])
    n = min(n_total, args.max_samples) if args.max_samples > 0 else n_total

    predictor = UnifiedLMWMPredictor.from_yaml(args.config)
    chunks: list[dict[str, np.ndarray]] = []
    totals = {
        "n": 0,
        "neural_greedy_ok": 0,
        "neural_max_ok": 0,
        "hybrid_greedy_ok": 0,
        "hybrid_max_ok": 0,
        "greedy_fallback": 0,
        "max_fallback": 0,
        "row_sum": 0.0,
        "row_sum_abs_err_max": 0.0,
        "greedy_latent_norm": 0.0,
        "max_latent_norm": 0.0,
        "greedy_error_probability": 0.0,
        "max_error_probability": 0.0,
        "has_error_probability": False,
    }
    for start in range(0, n, args.batch_size):
        end = min(n, start + args.batch_size)
        pred = predictor.predict(
            z["current"][start:end].astype(np.float32),
            z["current_milestone"][start:end].astype(np.int64),
        )
        if args.summary_only:
            bs = end - start
            totals["n"] += bs
            totals["neural_greedy_ok"] += int((pred["neural_greedy"] == pred["graph_greedy"]).sum())
            totals["neural_max_ok"] += int((pred["neural_max_product"] == pred["graph_max_product"]).sum())
            totals["hybrid_greedy_ok"] += int((pred["hybrid_greedy"] == pred["graph_greedy"]).sum())
            totals["hybrid_max_ok"] += int((pred["hybrid_max_product"] == pred["graph_max_product"]).sum())
            totals["greedy_fallback"] += int(pred["greedy_fallback_mask"].sum())
            totals["max_fallback"] += int(pred["max_product_fallback_mask"].sum())
            row_sum = pred["transition_probs"].sum(axis=1)
            totals["row_sum"] += float(row_sum.sum())
            totals["row_sum_abs_err_max"] = max(totals["row_sum_abs_err_max"], float(np.abs(row_sum - 1.0).max()))
            totals["greedy_latent_norm"] += float(np.linalg.norm(pred["hybrid_greedy_subgoal_latent"], axis=1).sum())
            totals["max_latent_norm"] += float(np.linalg.norm(pred["hybrid_max_product_subgoal_latent"], axis=1).sum())
            if "greedy_error_probability" in pred:
                totals["has_error_probability"] = True
                totals["greedy_error_probability"] += float(pred["greedy_error_probability"].sum())
                totals["max_error_probability"] += float(pred["max_product_error_probability"].sum())
        else:
            chunks.append(pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        out_npz = None
        denom = max(int(totals["n"]), 1)
        summary = {
            "config": str(args.config),
            "dataset_npz": str(dataset_npz),
            "output_npz": None,
            "summary_only": True,
            "num_samples": int(n),
            "batch_size": int(args.batch_size),
            "neural_greedy_top1_vs_graph": float(totals["neural_greedy_ok"] / denom),
            "neural_max_product_top1_vs_graph": float(totals["neural_max_ok"] / denom),
            "hybrid_greedy_top1_vs_graph": float(totals["hybrid_greedy_ok"] / denom),
            "hybrid_max_product_top1_vs_graph": float(totals["hybrid_max_ok"] / denom),
            "greedy_fallback_rate": float(totals["greedy_fallback"] / denom),
            "max_product_fallback_rate": float(totals["max_fallback"] / denom),
            "transition_row_sum_mean": float(totals["row_sum"] / denom),
            "transition_row_sum_abs_err_max": float(totals["row_sum_abs_err_max"]),
            "hybrid_greedy_latent_norm_mean": float(totals["greedy_latent_norm"] / denom),
            "hybrid_max_product_latent_norm_mean": float(totals["max_latent_norm"] / denom),
        }
        if totals["has_error_probability"]:
            summary["greedy_error_probability_mean"] = float(totals["greedy_error_probability"] / denom)
            summary["max_product_error_probability_mean"] = float(totals["max_error_probability"] / denom)
    else:
        pred_all = concat_chunks(chunks)
        out_npz = args.output_dir / "runtime_predictions.npz"
        np.savez_compressed(out_npz, **pred_all)
        summary = {
            "config": str(args.config),
            "dataset_npz": str(dataset_npz),
            "output_npz": str(out_npz),
            "summary_only": False,
            "num_samples": int(n),
            "batch_size": int(args.batch_size),
            "neural_greedy_top1_vs_graph": float((pred_all["neural_greedy"] == pred_all["graph_greedy"]).mean()),
            "neural_max_product_top1_vs_graph": float(
                (pred_all["neural_max_product"] == pred_all["graph_max_product"]).mean()
            ),
            "hybrid_greedy_top1_vs_graph": float((pred_all["hybrid_greedy"] == pred_all["graph_greedy"]).mean()),
            "hybrid_max_product_top1_vs_graph": float(
                (pred_all["hybrid_max_product"] == pred_all["graph_max_product"]).mean()
            ),
            "greedy_fallback_rate": float(pred_all["greedy_fallback_mask"].mean()),
            "max_product_fallback_rate": float(pred_all["max_product_fallback_mask"].mean()),
            "transition_row_sum_mean": float(pred_all["transition_probs"].sum(axis=1).mean()),
            "transition_row_sum_abs_err_max": float(np.abs(pred_all["transition_probs"].sum(axis=1) - 1.0).max()),
            "hybrid_greedy_latent_norm_mean": float(
                np.linalg.norm(pred_all["hybrid_greedy_subgoal_latent"], axis=1).mean()
            ),
            "hybrid_max_product_latent_norm_mean": float(
                np.linalg.norm(pred_all["hybrid_max_product_subgoal_latent"], axis=1).mean()
            ),
        }
        if "greedy_error_probability" in pred_all:
            summary["greedy_error_probability_mean"] = float(pred_all["greedy_error_probability"].mean())
            summary["max_product_error_probability_mean"] = float(pred_all["max_product_error_probability"].mean())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
