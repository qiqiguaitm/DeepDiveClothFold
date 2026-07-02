#!/usr/bin/env python
"""Sweep confidence thresholds for LMWM graph fallback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_milestones(text: str) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--weak_milestones", default="")
    ap.add_argument("--thresholds", default="0.80,0.85,0.90,0.92,0.93,0.94,0.95,0.96,0.97,0.98")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p = np.load(args.predictions)
    current = p["current_milestone"].astype(np.int64)
    greedy_pred = p["greedy_pred"].astype(np.int64)
    max_pred = p["max_product_pred"].astype(np.int64)
    greedy_target = p["greedy_target"].astype(np.int64)
    max_target = p["max_product_target"].astype(np.int64)
    greedy_conf = p["greedy_confidence"].astype(np.float32)
    max_conf = p["max_product_confidence"].astype(np.float32)
    weak = parse_milestones(args.weak_milestones)
    weak_mask = np.array([m in weak for m in current], dtype=bool)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    rows = []
    for tg in thresholds:
        for tm in thresholds:
            gf = (greedy_conf < tg) | weak_mask
            mf = (max_conf < tm) | weak_mask
            hybrid_g = np.where(gf, greedy_target, greedy_pred)
            hybrid_m = np.where(mf, max_target, max_pred)
            rows.append({
                "greedy_threshold": tg,
                "max_product_threshold": tm,
                "hybrid_greedy_top1": float((hybrid_g == greedy_target).mean()),
                "hybrid_max_product_top1": float((hybrid_m == max_target).mean()),
                "mean_top1": float(0.5 * ((hybrid_g == greedy_target).mean() + (hybrid_m == max_target).mean())),
                "greedy_fallback_rate": float(gf.mean()),
                "max_product_fallback_rate": float(mf.mean()),
                "mean_fallback_rate": float(0.5 * (gf.mean() + mf.mean())),
            })
    rows.sort(key=lambda r: (-r["mean_top1"], r["mean_fallback_rate"]))
    csv_path = args.output_dir / "hybrid_gate_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "predictions": str(args.predictions),
        "weak_milestones": sorted(weak),
        "output_csv": str(csv_path),
        "best_by_accuracy": rows[:10],
        "best_under_20pct_fallback": [r for r in rows if r["mean_fallback_rate"] <= 0.20][:10],
        "best_under_15pct_fallback": [r for r in rows if r["mean_fallback_rate"] <= 0.15][:10],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
