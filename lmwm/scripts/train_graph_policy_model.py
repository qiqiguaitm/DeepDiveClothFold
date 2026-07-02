#!/usr/bin/env python
"""Train a graph-supervised LMWM policy model from current frame features."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import load_config, load_graph_policy_data, split_indices  # noqa: E402
from lmwm.models import GraphSupervisedLMWM, count_params  # noqa: E402
from lmwm.training import (  # noqa: E402
    append_jsonl,
    make_ckpt_dir,
    make_run_dir,
    resolve_device,
    sample_batch,
    save_checkpoint,
    set_seed,
    write_json,
)


@dataclass
class Metrics:
    step: int
    train_loss: float
    val_loss: float
    val_kl: float
    val_greedy_ce: float
    val_max_product_ce: float
    val_greedy_top1: float
    val_max_product_top1: float


def compute_loss(out: dict[str, torch.Tensor], data: dict[str, torch.Tensor], b: torch.Tensor, weights: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(out["transition_logits"], dim=-1)
    kl = F.kl_div(logp, data["transition_target"][b], reduction="batchmean")
    greedy_ce = F.cross_entropy(out["greedy_logits"], data["greedy_target"][b])
    max_product_ce = F.cross_entropy(out["max_product_logits"], data["max_product_target"][b])
    loss = weights["kl"] * kl + weights["greedy"] * greedy_ce + weights["max_product"] * max_product_ce
    return loss, kl, greedy_ce, max_product_ce


@torch.no_grad()
def evaluate(model: nn.Module, data: dict[str, torch.Tensor], idx: torch.Tensor, batch_size: int, weights: dict[str, float]) -> dict[str, float]:
    model.eval()
    accum = {"loss": 0.0, "kl": 0.0, "greedy_ce": 0.0, "max_product_ce": 0.0}
    greedy_hits = max_product_hits = total = 0
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        out = model(data["current"][b])
        loss, kl, greedy_ce, max_product_ce = compute_loss(out, data, b, weights)
        n = int(len(b))
        accum["loss"] += float(loss.item()) * n
        accum["kl"] += float(kl.item()) * n
        accum["greedy_ce"] += float(greedy_ce.item()) * n
        accum["max_product_ce"] += float(max_product_ce.item()) * n
        greedy_hits += int((out["greedy_logits"].argmax(dim=-1) == data["greedy_target"][b]).sum().item())
        max_product_hits += int((out["max_product_logits"].argmax(dim=-1) == data["max_product_target"][b]).sum().item())
        total += n
    return {
        "loss": accum["loss"] / total,
        "kl": accum["kl"] / total,
        "greedy_ce": accum["greedy_ce"] / total,
        "max_product_ce": accum["max_product_ce"] / total,
        "greedy_top1": greedy_hits / total,
        "max_product_top1": max_product_hits / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    run_dir = make_run_dir(cfg, args.config, "lmwm/logs/stage2_graph", "lmwm_graph_run")
    ckpt_dir = make_ckpt_dir(cfg, run_dir, "lmwm/checkpoints/stage2_graph")

    device = resolve_device(cfg)
    data, z, g = load_graph_policy_data(cfg, device, include_proto=False)
    n, in_dim = data["current"].shape
    num_milestones = int(g["transition_probs"].shape[0])
    train_idx, val_idx = split_indices(
        z, n, float(cfg["training"].get("val_ratio", 0.2)), seed, device, str(cfg.get("split_mode", "random"))
    )

    mc = cfg["model"]
    model = GraphSupervisedLMWM(in_dim, num_milestones, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-6)),
    )
    weights = {
        "kl": float(cfg["training"].get("kl_weight", 1.0)),
        "greedy": float(cfg["training"].get("greedy_ce_weight", 1.0)),
        "max_product": float(cfg["training"].get("max_product_ce_weight", 1.0)),
    }
    batch_size = int(cfg["training"].get("batch_size", 2048))
    max_steps = int(cfg["training"].get("max_steps", 1000))
    eval_interval = int(cfg["training"].get("eval_interval", 100))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    meta = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "dataset_npz": cfg["dataset_npz"],
        "graph_npz": cfg["graph_npz"],
        "device": str(device),
        "num_pairs": int(n),
        "input_dim": int(in_dim),
        "num_milestones": int(num_milestones),
        "train_pairs": int(len(train_idx)),
        "val_pairs": int(len(val_idx)),
        "model_params": count_params(model),
        "objective": "current frame feature -> transition row, greedy next, max-product completion next",
    }
    write_json(run_dir / "run_meta.json", meta)
    print(json.dumps(meta, indent=2))

    metrics_path = run_dir / "metrics.jsonl"
    best = -1.0
    last: Metrics | None = None
    for step in range(1, max_steps + 1):
        model.train()
        b = sample_batch(train_idx, batch_size, device)
        out = model(data["current"][b])
        loss, _, _, _ = compute_loss(out, data, b, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step == 1 or step % eval_interval == 0 or step == max_steps:
            ev = evaluate(model, data, val_idx, batch_size, weights)
            last = Metrics(
                step=step,
                train_loss=float(loss.item()),
                val_loss=ev["loss"],
                val_kl=ev["kl"],
                val_greedy_ce=ev["greedy_ce"],
                val_max_product_ce=ev["max_product_ce"],
                val_greedy_top1=ev["greedy_top1"],
                val_max_product_top1=ev["max_product_top1"],
            )
            row = asdict(last)
            append_jsonl(metrics_path, row)
            print(json.dumps(row), flush=True)
            score = 0.5 * (last.val_greedy_top1 + last.val_max_product_top1)
            if score > best:
                best = score
                save_checkpoint(ckpt_dir / "best.pt", model, cfg, meta, row)
    save_checkpoint(ckpt_dir / "last.pt", model, cfg, meta, asdict(last) if last else None)
    if last is not None:
        write_json(run_dir / "final_metrics.json", asdict(last))
    print(f"saved {ckpt_dir / 'best.pt'}")
    print(f"saved {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
