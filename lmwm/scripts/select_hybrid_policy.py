#!/usr/bin/env python
"""Select a validation-driven hybrid fallback policy for LMWM inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in [
            "milestone",
            "samples",
            "transition_support",
            "greedy_top1",
            "max_product_top1",
            "greedy_confidence_mean",
            "max_product_confidence_mean",
            "transition_entropy_mean",
        ]:
            if key in row:
                row[key] = float(row[key])
        row["milestone"] = int(row["milestone"])
        row["samples"] = int(row["samples"])
        row["transition_support"] = int(row["transition_support"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_milestone_csv", required=True, type=Path)
    ap.add_argument("--base_config", required=True, type=Path)
    ap.add_argument("--output_config", required=True, type=Path)
    ap.add_argument("--output_summary", required=True, type=Path)
    ap.add_argument("--min_samples", type=int, default=30)
    ap.add_argument("--min_transition_support", type=int, default=0)
    ap.add_argument("--min_head_top1", type=float, default=0.94)
    ap.add_argument("--greedy_conf_threshold", type=float, default=0.90)
    ap.add_argument("--max_product_conf_threshold", type=float, default=0.92)
    args = ap.parse_args()

    rows = read_rows(args.per_milestone_csv)
    selected = []
    rejected = []
    for row in rows:
        enough_support = row["samples"] >= args.min_samples and row["transition_support"] >= args.min_transition_support
        weak_head = row["greedy_top1"] < args.min_head_top1 or row["max_product_top1"] < args.min_head_top1
        reason = {
            "milestone": row["milestone"],
            "samples": row["samples"],
            "transition_support": row["transition_support"],
            "greedy_top1": row["greedy_top1"],
            "max_product_top1": row["max_product_top1"],
            "selected": bool(enough_support and weak_head),
            "reason": "low_head_top1" if enough_support and weak_head else "not_selected",
        }
        if reason["selected"]:
            selected.append(reason)
        else:
            rejected.append(reason)

    with args.base_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["name"] = args.output_config.stem
    cfg["output_dir"] = str(Path("lmwm/outputs/stage3_unified_inference") / args.output_config.stem)
    cfg["weak_milestones"] = [r["milestone"] for r in selected]
    cfg["greedy_conf_threshold"] = float(args.greedy_conf_threshold)
    cfg["max_product_conf_threshold"] = float(args.max_product_conf_threshold)
    cfg["selection_policy"] = {
        "source_csv": str(args.per_milestone_csv),
        "min_samples": int(args.min_samples),
        "min_transition_support": int(args.min_transition_support),
        "min_head_top1": float(args.min_head_top1),
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    summary = {
        "base_config": str(args.base_config),
        "per_milestone_csv": str(args.per_milestone_csv),
        "output_config": str(args.output_config),
        "selected_weak_milestones": cfg["weak_milestones"],
        "num_selected": len(selected),
        "selection_policy": cfg["selection_policy"],
        "selected_details": selected,
        "top_rejected_low_accuracy": sorted(
            rejected,
            key=lambda r: min(r["greedy_top1"], r["max_product_top1"]),
        )[:10],
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
