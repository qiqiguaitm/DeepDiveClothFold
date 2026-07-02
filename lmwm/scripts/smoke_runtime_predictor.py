#!/usr/bin/env python
"""Smoke-test the package-level LMWM runtime predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.runtime import UnifiedLMWMPredictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset_npz", type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--num_samples", type=int, default=256)
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    dataset_npz = args.dataset_npz or Path(cfg["dataset_npz"])
    z = np.load(dataset_npz)
    n = min(args.num_samples, int(z["current"].shape[0]))
    features = z["current"][:n].astype(np.float32)
    milestones = z["current_milestone"][:n].astype(np.int64)

    predictor = UnifiedLMWMPredictor.from_yaml(args.config)
    pred = predictor.predict(features, milestones)
    inferred = predictor.predict(features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.output_dir / "runtime_predictions.npz"
    np.savez_compressed(out_npz, **pred)

    summary = {
        "config": str(args.config),
        "dataset_npz": str(dataset_npz),
        "output_npz": str(out_npz),
        "num_samples": n,
        "input_dim": int(features.shape[1]),
        "api_fields": sorted(pred.keys()),
        "explicit_current_stage_match": bool(np.array_equal(pred["current_milestone"], milestones)),
        "inferred_current_stage_match": float((inferred["current_milestone"] == milestones).mean()),
        "hybrid_greedy_fallback_rate": float(pred["greedy_fallback_mask"].mean()),
        "hybrid_max_product_fallback_rate": float(pred["max_product_fallback_mask"].mean()),
        "transition_row_sum_mean": float(pred["transition_probs"].sum(axis=1).mean()),
        "transition_row_sum_abs_err_max": float(np.abs(pred["transition_probs"].sum(axis=1) - 1.0).max()),
        "hybrid_greedy_latent_norm_mean": float(np.linalg.norm(pred["hybrid_greedy_subgoal_latent"], axis=1).mean()),
        "hybrid_max_product_latent_norm_mean": float(np.linalg.norm(pred["hybrid_max_product_subgoal_latent"], axis=1).mean()),
        "neural_greedy_top1_vs_graph": float((pred["neural_greedy"] == pred["graph_greedy"]).mean()),
        "neural_max_product_top1_vs_graph": float((pred["neural_max_product"] == pred["graph_max_product"]).mean()),
        "hybrid_greedy_top1_vs_graph": float((pred["hybrid_greedy"] == pred["graph_greedy"]).mean()),
        "hybrid_max_product_top1_vs_graph": float((pred["hybrid_max_product"] == pred["graph_max_product"]).mean()),
    }
    if "greedy_error_probability" in pred:
        summary["greedy_error_probability_mean"] = float(pred["greedy_error_probability"].mean())
        summary["max_product_error_probability_mean"] = float(pred["max_product_error_probability"].mean())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
