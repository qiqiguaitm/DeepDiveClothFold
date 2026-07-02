#!/usr/bin/env python
"""Compute confidence calibration metrics for LMWM milestone heads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def calibration_bins(conf: np.ndarray, correct: np.ndarray, num_bins: int) -> tuple[list[dict], dict]:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    n = len(conf)
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            acc = float("nan")
            avg_conf = float("nan")
            gap = float("nan")
        else:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = abs(acc - avg_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        rows.append({
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "count": count,
            "coverage": float(count / n),
            "accuracy": acc,
            "avg_confidence": avg_conf,
            "abs_gap": gap,
        })
    summary = {
        "num_samples": int(n),
        "num_bins": int(num_bins),
        "accuracy": float(correct.mean()),
        "confidence_mean": float(conf.mean()),
        "ece": float(ece),
        "mce": float(mce),
    }
    return rows, summary


def threshold_curve(conf: np.ndarray, correct: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for t in thresholds:
        accept = conf >= t
        accepted = int(accept.sum())
        if accepted == 0:
            accepted_acc = float("nan")
            rejected_acc = float(correct.mean())
        else:
            accepted_acc = float(correct[accept].mean())
            rejected_acc = float(correct[~accept].mean()) if accepted < len(correct) else float("nan")
        rows.append({
            "threshold": float(t),
            "accept_rate": float(accept.mean()),
            "fallback_rate": float((~accept).mean()),
            "accepted_accuracy": accepted_acc,
            "rejected_accuracy": rejected_acc,
        })
    return rows


def risk_coverage_auc(conf: np.ndarray, correct: np.ndarray) -> dict:
    order = np.argsort(-conf)
    sorted_correct = correct[order].astype(np.float64)
    coverage = np.arange(1, len(correct) + 1, dtype=np.float64) / len(correct)
    risk = 1.0 - np.cumsum(sorted_correct) / np.arange(1, len(correct) + 1, dtype=np.float64)
    aurc = float(np.trapezoid(risk, coverage))
    return {
        "aurc": aurc,
        "risk_at_80pct_coverage": float(risk[max(0, int(np.ceil(0.80 * len(risk))) - 1)]),
        "risk_at_90pct_coverage": float(risk[max(0, int(np.ceil(0.90 * len(risk))) - 1)]),
        "risk_at_95pct_coverage": float(risk[max(0, int(np.ceil(0.95 * len(risk))) - 1)]),
    }


def best_threshold_under_budget(rows: list[dict], max_fallback: float) -> dict | None:
    candidates = [r for r in rows if r["fallback_rate"] <= max_fallback]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["accepted_accuracy"], -r["fallback_rate"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--num_bins", type=int, default=10)
    ap.add_argument("--thresholds", default="0.50,0.70,0.80,0.85,0.90,0.92,0.95,0.97,0.99")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.predictions)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    heads = {
        "greedy": (
            z["greedy_confidence"].astype(np.float32),
            z["greedy_pred"].astype(np.int64) == z["greedy_target"].astype(np.int64),
        ),
        "max_product": (
            z["max_product_confidence"].astype(np.float32),
            z["max_product_pred"].astype(np.int64) == z["max_product_target"].astype(np.int64),
        ),
    }
    summary = {
        "predictions": str(args.predictions),
        "output_dir": str(args.output_dir),
        "num_bins": int(args.num_bins),
        "heads": {},
    }
    for name, (conf, correct) in heads.items():
        bin_rows, head_summary = calibration_bins(conf, correct, args.num_bins)
        curve_rows = threshold_curve(conf, correct, thresholds)
        write_csv(args.output_dir / f"{name}_reliability_bins.csv", bin_rows)
        write_csv(args.output_dir / f"{name}_threshold_curve.csv", curve_rows)
        summary["heads"][name] = {
            **head_summary,
            **risk_coverage_auc(conf, correct),
            "best_threshold_under_15pct_fallback": best_threshold_under_budget(curve_rows, 0.15),
            "best_threshold_under_20pct_fallback": best_threshold_under_budget(curve_rows, 0.20),
            "best_threshold_under_25pct_fallback": best_threshold_under_budget(curve_rows, 0.25),
            "threshold_curve": curve_rows,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
