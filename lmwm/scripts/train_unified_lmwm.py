#!/usr/bin/env python
"""Train a unified LMWM from frame features to graph policy and latent subgoals."""

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
from lmwm.models import UnifiedLMWM, count_params  # noqa: E402
from lmwm.training import (  # noqa: E402
    append_jsonl,
    build_param_groups,
    make_ckpt_dir,
    make_run_dir,
    make_warmup_scheduler,
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
    val_greedy_proto_mse: float
    val_max_product_proto_mse: float
    val_greedy_top1: float
    val_max_product_top1: float
    val_greedy_proto_cos: float
    val_max_product_proto_cos: float


def compute_losses(out: dict[str, torch.Tensor], data: dict[str, torch.Tensor], b: torch.Tensor, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # proto_beta: smooth_l1 transition point. LaWM uses 0.1 for DINO-feature
    # regression (errors are tiny, so beta=1.0 leaves them in the weak-gradient
    # quadratic regime); default 1.0 keeps the original LMWM behavior.
    beta = float(weights.get("proto_beta", 1.0))
    logp = F.log_softmax(out["transition_logits"], dim=-1)
    kl = F.kl_div(logp, data["transition_target"][b], reduction="batchmean")
    greedy_ce = F.cross_entropy(out["greedy_logits"], data["greedy_target"][b])
    max_product_ce = F.cross_entropy(out["max_product_logits"], data["max_product_target"][b])
    greedy_proto_mse = F.smooth_l1_loss(out["greedy_proto"], data["greedy_proto_target"][b], beta=beta)
    max_product_proto_mse = F.smooth_l1_loss(out["max_product_proto"], data["max_product_proto_target"][b], beta=beta)
    loss = (
        weights["kl"] * kl
        + weights["greedy_ce"] * greedy_ce
        + weights["max_product_ce"] * max_product_ce
        + weights["greedy_proto"] * greedy_proto_mse
        + weights["max_product_proto"] * max_product_proto_mse
    )
    return loss, {
        "kl": kl,
        "greedy_ce": greedy_ce,
        "max_product_ce": max_product_ce,
        "greedy_proto_mse": greedy_proto_mse,
        "max_product_proto_mse": max_product_proto_mse,
    }


@torch.no_grad()
def evaluate(model: nn.Module, data: dict[str, torch.Tensor], idx: torch.Tensor, batch_size: int, weights: dict[str, float]) -> dict[str, float]:
    model.eval()
    sums = {k: 0.0 for k in ["loss", "kl", "greedy_ce", "max_product_ce", "greedy_proto_mse", "max_product_proto_mse", "greedy_cos", "max_product_cos"]}
    greedy_hits = max_hits = total = 0
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        out = model(data["current"][b])
        loss, parts = compute_losses(out, data, b, weights)
        n = int(len(b))
        sums["loss"] += float(loss.item()) * n
        for k, v in parts.items():
            sums[k] += float(v.item()) * n
        greedy_hits += int((out["greedy_logits"].argmax(dim=-1) == data["greedy_target"][b]).sum().item())
        max_hits += int((out["max_product_logits"].argmax(dim=-1) == data["max_product_target"][b]).sum().item())
        sums["greedy_cos"] += float(F.cosine_similarity(out["greedy_proto"], data["greedy_proto_target"][b], dim=-1).sum().item())
        sums["max_product_cos"] += float(F.cosine_similarity(out["max_product_proto"], data["max_product_proto_target"][b], dim=-1).sum().item())
        total += n
    result = {k: v / total for k, v in sums.items()}
    result["greedy_top1"] = greedy_hits / total
    result["max_product_top1"] = max_hits / total
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    run_dir = make_run_dir(cfg, args.config, "lmwm/logs/stage3_unified", "unified_lmwm")
    ckpt_dir = make_ckpt_dir(cfg, run_dir, "lmwm/checkpoints/stage3_unified")

    device = resolve_device(cfg)
    label_source = str(cfg.get("label_source", "graph_lookup"))
    proto_target_source = str(cfg.get("proto_target_source", "centroid"))
    data, z, g = load_graph_policy_data(cfg, device, include_proto=True, label_source=label_source, proto_target_source=proto_target_source)
    n, in_dim = data["current"].shape
    latent_dim = int(g["prototype_table"].shape[1])
    num_milestones = int(g["transition_probs"].shape[0])
    train_idx, val_idx = split_indices(
        z, n, float(cfg["training"].get("val_ratio", 0.2)), seed, device, str(cfg.get("split_mode", "random"))
    )

    mc = cfg["model"]
    model = UnifiedLMWM(in_dim, latent_dim, num_milestones, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(device)
    weight_decay = float(cfg["training"].get("weight_decay", 1e-6))
    exclude_bias_norm = bool(cfg["training"].get("exclude_bias_norm_from_wd", False))
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    opt = torch.optim.AdamW(
        build_param_groups(model, weight_decay, exclude_bias_norm),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
    )
    scheduler = make_warmup_scheduler(opt, warmup_steps)
    weights = {
        "kl": float(cfg["training"].get("kl_weight", 1.0)),
        "greedy_ce": float(cfg["training"].get("greedy_ce_weight", 1.0)),
        "max_product_ce": float(cfg["training"].get("max_product_ce_weight", 1.0)),
        "greedy_proto": float(cfg["training"].get("greedy_proto_weight", 5.0)),
        "max_product_proto": float(cfg["training"].get("max_product_proto_weight", 5.0)),
        "proto_beta": float(cfg["training"].get("proto_smooth_l1_beta", 1.0)),
    }
    batch_size = int(cfg["training"].get("batch_size", 2048))
    max_steps = int(cfg["training"].get("max_steps", 1200))
    eval_interval = int(cfg["training"].get("eval_interval", 100))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    meta = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "dataset_npz": cfg["dataset_npz"],
        "graph_npz": cfg["graph_npz"],
        "label_source": label_source,
        "proto_target_source": proto_target_source,
        "weight_decay": weight_decay,
        "exclude_bias_norm_from_wd": exclude_bias_norm,
        "warmup_steps": warmup_steps,
        "device": str(device),
        "num_pairs": int(n),
        "input_dim": int(in_dim),
        "latent_dim": latent_dim,
        "num_milestones": num_milestones,
        "train_pairs": int(len(train_idx)),
        "val_pairs": int(len(val_idx)),
        "model_params": count_params(model),
        "objective": "frame feature -> transition distribution + greedy/max-product milestone ids + latent prototype subgoals",
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
        loss, _ = compute_losses(out, data, b, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if step == 1 or step % eval_interval == 0 or step == max_steps:
            ev = evaluate(model, data, val_idx, batch_size, weights)
            last = Metrics(
                step=step,
                train_loss=float(loss.item()),
                val_loss=ev["loss"],
                val_kl=ev["kl"],
                val_greedy_ce=ev["greedy_ce"],
                val_max_product_ce=ev["max_product_ce"],
                val_greedy_proto_mse=ev["greedy_proto_mse"],
                val_max_product_proto_mse=ev["max_product_proto_mse"],
                val_greedy_top1=ev["greedy_top1"],
                val_max_product_top1=ev["max_product_top1"],
                val_greedy_proto_cos=ev["greedy_cos"],
                val_max_product_proto_cos=ev["max_product_cos"],
            )
            row = asdict(last)
            append_jsonl(metrics_path, row)
            print(json.dumps(row), flush=True)
            score = 0.25 * (last.val_greedy_top1 + last.val_max_product_top1 + last.val_greedy_proto_cos + last.val_max_product_proto_cos)
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
