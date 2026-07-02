#!/usr/bin/env python
"""Analyze unified LMWM predictions by current milestone and transition support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--graph_npz", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p = np.load(args.predictions)
    g = np.load(args.graph_npz)
    counts = g["transition_counts"].astype(np.int64)
    support = counts.sum(axis=1)
    current = p["current_milestone"].astype(np.int64)
    greedy_pred_key = "greedy_pred" if "greedy_pred" in p.files else "neural_greedy"
    max_pred_key = "max_product_pred" if "max_product_pred" in p.files else "neural_max_product"
    greedy_target = p["greedy_target"].astype(np.int64) if "greedy_target" in p.files else p["graph_greedy"].astype(np.int64)
    max_target = (
        p["max_product_target"].astype(np.int64)
        if "max_product_target" in p.files
        else p["graph_max_product"].astype(np.int64)
    )
    greedy_ok = p[greedy_pred_key].astype(np.int64) == greedy_target
    max_ok = p[max_pred_key].astype(np.int64) == max_target
    has_hybrid = "hybrid_greedy" in p.files and "hybrid_max_product" in p.files
    if has_hybrid:
        hybrid_greedy_ok = p["hybrid_greedy"].astype(np.int64) == greedy_target
        hybrid_max_ok = p["hybrid_max_product"].astype(np.int64) == max_target
        greedy_fallback = p["greedy_fallback_mask"].astype(bool)
        max_fallback = p["max_product_fallback_mask"].astype(bool)
    rows = []
    for m in sorted(set(current.tolist())):
        mask = current == m
        greedy_cos = float(p["greedy_proto_cos"][mask].mean()) if "greedy_proto_cos" in p.files else None
        max_cos = (
            float(p["max_product_proto_cos"][mask].mean())
            if "max_product_proto_cos" in p.files
            else None
        )
        row = {
            "milestone": int(m),
            "samples": int(mask.sum()),
            "transition_support": int(support[m]),
            "greedy_top1": float(greedy_ok[mask].mean()),
            "max_product_top1": float(max_ok[mask].mean()),
            "greedy_confidence_mean": float(p["greedy_confidence"][mask].mean()),
            "max_product_confidence_mean": float(p["max_product_confidence"][mask].mean()),
            "transition_entropy_mean": float(p["transition_entropy"][mask].mean()),
            "greedy_proto_cos_mean": greedy_cos,
            "max_product_proto_cos_mean": max_cos,
        }
        if has_hybrid:
            hybrid_greedy_cos = (
                float(p["hybrid_greedy_proto_cos"][mask].mean())
                if "hybrid_greedy_proto_cos" in p.files
                else None
            )
            hybrid_max_cos = (
                float(p["hybrid_max_product_proto_cos"][mask].mean())
                if "hybrid_max_product_proto_cos" in p.files
                else None
            )
            row.update({
                "hybrid_greedy_top1": float(hybrid_greedy_ok[mask].mean()),
                "hybrid_max_product_top1": float(hybrid_max_ok[mask].mean()),
                "greedy_fallback_rate": float(greedy_fallback[mask].mean()),
                "max_product_fallback_rate": float(max_fallback[mask].mean()),
                "hybrid_greedy_proto_cos_mean": hybrid_greedy_cos,
                "hybrid_max_product_proto_cos_mean": hybrid_max_cos,
            })
        rows.append(row)
    rows.sort(key=lambda r: (r["greedy_top1"] + r["max_product_top1"], r["samples"]))
    csv_path = args.output_dir / "per_milestone_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "predictions": str(args.predictions),
        "graph_npz": str(args.graph_npz),
        "output_csv": str(csv_path),
        "num_milestones_seen": len(rows),
        "worst_milestones": rows[:10],
        "low_support_milestones": sorted(rows, key=lambda r: r["transition_support"])[:10],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
