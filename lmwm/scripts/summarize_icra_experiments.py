#!/usr/bin/env python
"""Build paper-oriented LMWM experiment summary tables from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_STAGE_RUNS = {
    "stage1ab_fixed": Path("lmwm/logs/stage1ab/20260701_140401+kai0base_dinov3h_stage1ab_fixed"),
    "stage1c_next_unique": Path("lmwm/logs/stage1c/20260701_140401+kai0base_dinov3h_stage1c_next_unique"),
    "stage1d_frame2proto": Path("lmwm/logs/stage1d/20260701_142250+kai0base_dinov3h_stage1d_frame2proto_next_unique"),
    "stage2_graph_policy": Path("lmwm/logs/stage2_graph/20260701_142639+kai0base_dinov3h_stage2_graph_policy"),
    "stage3_unified": Path("lmwm/logs/stage3_unified/20260701_142850+kai0base_dinov3h_stage3_unified"),
    "kai0bd_stage1ab_fixed": Path("lmwm/logs/stage1ab/20260702_040847+kai0bd_stage1ab_fixed"),
    "kai0bd_stage1c_next_unique": Path("lmwm/logs/stage1c/20260702_040849+kai0bd_stage1c_next_unique"),
    "kai0bd_stage2_graph_policy": Path("lmwm/logs/stage2_graph/20260702_041309+kai0bd_stage2_graph_policy"),
    "kai0bd_stage3_unified": Path("lmwm/logs/stage3_unified/20260702_041309+kai0bd_stage3_unified"),
}

DEFAULT_GRAPH_METAS = {
    "kai0base_dinov3h": Path("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_meta.json"),
    "kai0bd_feature_stage1": Path("lmwm/data/recurrence_graphs/kai0bd_feature_stage1/recurrence_graph_meta.json"),
}
DEFAULT_RUNTIME_COMPARISON = Path("lmwm/outputs/runtime_eval/20260702_policy_comparison_summary.json")
DEFAULT_EXTRA_RUNTIME_SUMMARIES = {
    "kai0bd_recommended": Path("lmwm/outputs/runtime_eval/20260702_kai0bd_recommended_full_summary/summary.json"),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def stage_row(name: str, run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "final_metrics.json")
    meta = read_json(run_dir / "run_meta.json")
    return {
        "stage": name,
        "run_dir": str(run_dir),
        "checkpoint_dir": meta.get("checkpoint_dir", "-"),
        "num_pairs": meta.get("num_pairs"),
        "input_dim": meta.get("input_dim", meta.get("raw_dim")),
        "num_milestones": meta.get("num_milestones"),
        "step": metrics.get("step"),
        "val_top1": metrics.get("val_top1"),
        "val_greedy_top1": metrics.get("val_greedy_top1"),
        "val_max_product_top1": metrics.get("val_max_product_top1"),
        "val_kl": metrics.get("val_kl"),
        "val_mse": metrics.get("val_mse"),
        "val_greedy_proto_cos": metrics.get("val_greedy_proto_cos"),
        "val_max_product_proto_cos": metrics.get("val_max_product_proto_cos"),
    }


def graph_summary(path: Path) -> dict[str, Any]:
    meta = read_json(path)
    return {
        "path": str(path),
        "valid_frames": meta.get("valid_frames", meta.get("num_valid_frames", meta.get("num_pairs"))),
        "episodes": meta.get("episodes", meta.get("num_episodes")),
        "num_milestones": meta.get("num_milestones"),
        "nonzero_edges": meta.get("nonzero_edges"),
        "mean_compressed_episode_length": meta.get("mean_compressed_episode_length"),
        "terminal_milestone": meta.get("terminal_milestone", meta.get("terminal_target")),
        "terminal_progress": meta.get("terminal_progress", meta.get("terminal_target_progress")),
        "source": meta.get("graph_source", meta.get("feature_dir", meta.get("dataset_npz"))),
    }


def runtime_rows(path: Path) -> list[dict[str, Any]]:
    summary = read_json(path)
    rows = []
    for item in summary.get("policies", []):
        rows.append(
            {
                "policy": item.get("policy"),
                "summary": item.get("summary"),
                "num_samples": summary.get("num_samples"),
                "hybrid_greedy_top1": item.get("hybrid_greedy_top1"),
                "hybrid_max_product_top1": item.get("hybrid_max_product_top1"),
                "mean_hybrid_top1": item.get("mean_hybrid_top1"),
                "greedy_fallback_rate": item.get("greedy_fallback_rate"),
                "max_product_fallback_rate": item.get("max_product_fallback_rate"),
                "mean_fallback_rate": item.get("mean_fallback_rate"),
            }
        )
    return rows


def runtime_summary_row(name: str, path: Path) -> dict[str, Any]:
    summary = read_json(path)
    return {
        "policy": name,
        "summary": str(path),
        "num_samples": summary.get("num_samples"),
        "hybrid_greedy_top1": summary.get("hybrid_greedy_top1_vs_graph"),
        "hybrid_max_product_top1": summary.get("hybrid_max_product_top1_vs_graph"),
        "mean_hybrid_top1": 0.5
        * (float(summary.get("hybrid_greedy_top1_vs_graph", 0.0)) + float(summary.get("hybrid_max_product_top1_vs_graph", 0.0))),
        "greedy_fallback_rate": summary.get("greedy_fallback_rate"),
        "max_product_fallback_rate": summary.get("max_product_fallback_rate"),
        "mean_fallback_rate": 0.5
        * (float(summary.get("greedy_fallback_rate", 0.0)) + float(summary.get("max_product_fallback_rate", 0.0))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_markdown(
    path: Path,
    graphs: dict[str, dict[str, Any]],
    stages: list[dict[str, Any]],
    policies: list[dict[str, Any]],
) -> None:
    graph_rows = []
    for name, graph in graphs.items():
        graph_rows.append(
            [
                name,
                fmt(graph["valid_frames"]),
                fmt(graph["episodes"]),
                fmt(graph["num_milestones"]),
                fmt(graph["nonzero_edges"]),
                fmt(graph["mean_compressed_episode_length"]),
                f"#{fmt(graph['terminal_milestone'])}",
                fmt(graph["terminal_progress"]),
            ]
        )

    stage_rows = []
    for row in stages:
        stage_rows.append(
            [
                row["stage"],
                fmt(row["num_pairs"]),
                fmt(row["input_dim"]),
                fmt(row["num_milestones"]),
                fmt(row["step"]),
                fmt(row["val_top1"]),
                fmt(row["val_greedy_top1"]),
                fmt(row["val_max_product_top1"]),
                fmt(row["val_kl"]),
                fmt(row["val_greedy_proto_cos"]),
                fmt(row["val_max_product_proto_cos"]),
            ]
        )

    policy_rows = []
    for row in policies:
        policy_rows.append(
            [
                row["policy"],
                fmt(row["num_samples"]),
                fmt(row["mean_hybrid_top1"]),
                fmt(row["hybrid_greedy_top1"]),
                fmt(row["hybrid_max_product_top1"]),
                fmt(row["mean_fallback_rate"]),
                fmt(row["greedy_fallback_rate"]),
                fmt(row["max_product_fallback_rate"]),
            ]
        )

    text = f"""# ICRA-Oriented LMWM Experiment Snapshot

Generated from existing local artifacts. This is a paper-table snapshot, not a
new training run.

## Terminology

- **Greedy**: one-step local prediction, `argmax P(stage_{{t+1}} | stage_t)`.
- **Max-product**: finite-horizon dynamic programming / max-product search
  toward the terminal milestone; report the next step on that path.

## Recurrence Graphs

{markdown_table(
        [
            "Graph",
            "Frames/Pairs",
            "Episodes",
            "Milestones",
            "Edges",
            "Mean compressed len",
            "Terminal",
            "Terminal progress",
        ],
        graph_rows,
    )}

## Training Stages

{markdown_table(
        [
            "Stage",
            "Pairs",
            "Input dim",
            "Milestones",
            "Step",
            "Top1",
            "Greedy top1",
            "Max-product top1",
            "KL",
            "Greedy proto cos",
            "Max-product proto cos",
        ],
        stage_rows,
    )}

## Runtime Policies

{markdown_table(
        [
            "Policy",
            "Samples",
            "Mean top1",
            "Greedy top1",
            "Max-product top1",
            "Mean fallback",
            "Greedy fallback",
            "Max-product fallback",
        ],
        policy_rows,
    )}

## Paper Read

- `recommended` remains the balanced default: high mean top1 with lower fallback
  than the graph-prior-heavy safe policy.
- `validation_selected_safe` is the highest-accuracy safety mode, but it relies
  more heavily on graph fallback.
- `learned_tuned` is the strongest learned-fallback candidate, but it does not
  replace the default until it wins on broader held-out or cross-task data.

## Current Gap To ICRA-Ready Evidence

- Promote the new `kai0bd` evidence from Stage-1/2/3 pipeline validation into
  an independent held-out or cross-task evaluation.
- Add independent criteria beyond graph-table labels.
- Add VLA-side ablations showing whether latent milestone subgoals improve
  downstream execution or data selection.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=Path("lmwm/docs/icra_experiments"))
    ap.add_argument("--runtime_comparison", type=Path, default=DEFAULT_RUNTIME_COMPARISON)
    args = ap.parse_args()

    stages = [stage_row(name, path) for name, path in DEFAULT_STAGE_RUNS.items()]
    graphs = {name: graph_summary(path) for name, path in DEFAULT_GRAPH_METAS.items()}
    policies = runtime_rows(args.runtime_comparison)
    policies.extend(runtime_summary_row(name, path) for name, path in DEFAULT_EXTRA_RUNTIME_SUMMARIES.items())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "graphs": graphs,
        "stages": stages,
        "runtime_policies": policies,
        "source_artifacts": {
            "stage_runs": {name: str(path) for name, path in DEFAULT_STAGE_RUNS.items()},
            "graph_metas": {name: str(path) for name, path in DEFAULT_GRAPH_METAS.items()},
            "runtime_comparison": str(args.runtime_comparison),
            "extra_runtime_summaries": {name: str(path) for name, path in DEFAULT_EXTRA_RUNTIME_SUMMARIES.items()},
        },
    }
    (args.output_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "training_stages.csv", stages)
    write_csv(args.output_dir / "runtime_policies.csv", policies)
    write_markdown(args.output_dir / "README.md", graphs, stages, policies)
    print(json.dumps({"output_dir": str(args.output_dir), "stage_rows": len(stages), "policy_rows": len(policies)}, indent=2))


if __name__ == "__main__":
    main()
