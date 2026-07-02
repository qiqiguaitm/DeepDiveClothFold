#!/usr/bin/env python
"""Train a LaWM-shaped LMWM Stage-1 transition model."""

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
from lmwm.data import load_config, load_state_pair_data, split_indices  # noqa: E402
from lmwm.models import LaWMShapedLMWM, count_params  # noqa: E402
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
    val_mse: float
    val_ce: float
    val_top1: float
    val_top3: float


@torch.no_grad()
def evaluate(model: nn.Module, data: dict[str, torch.Tensor], idx: torch.Tensor, batch_size: int, ce_weight: float) -> tuple[float, float, float, float, float]:
    model.eval()
    loss_sum = mse_sum = ce_sum = 0.0
    top1_hits = top3_hits = total = 0
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        out = model(data["current"][b], data["future"][b])
        mse = F.smooth_l1_loss(out["r_hat"], out["r_future"])
        ce = F.cross_entropy(out["logits"], data["future_milestone"][b])
        loss = mse + ce_weight * ce
        n = int(len(b))
        loss_sum += float(loss.item()) * n
        mse_sum += float(mse.item()) * n
        ce_sum += float(ce.item()) * n
        pred = out["logits"].argmax(dim=-1)
        top1_hits += int((pred == data["future_milestone"][b]).sum().item())
        top3 = out["logits"].topk(k=min(3, out["logits"].shape[-1]), dim=-1).indices
        top3_hits += int((top3 == data["future_milestone"][b, None]).any(dim=-1).sum().item())
        total += n
    return loss_sum / total, mse_sum / total, ce_sum / total, top1_hits / total, top3_hits / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)

    run_dir = make_run_dir(cfg, args.config, "lmwm/logs/stage1ab", "lmwm_run")
    ckpt_dir = make_ckpt_dir(cfg, run_dir, "lmwm/checkpoints/stage1ab")

    device = resolve_device(cfg)
    data, z = load_state_pair_data(cfg, device)
    n, raw_dim = data["current"].shape
    num_milestones = int(z["prototype_table"].shape[0])
    train_idx, val_idx = split_indices(
        z, n, float(cfg["training"].get("val_ratio", 0.2)), seed, device, str(cfg.get("split_mode", "random"))
    )

    mc = cfg["model"]
    model = LaWMShapedLMWM(
        raw_dim=raw_dim,
        num_milestones=num_milestones,
        code_dim=int(mc.get("code_dim", 32)),
        transition_dim=int(mc.get("transition_dim", 32)),
        hidden_dim=int(mc.get("hidden_dim", 128)),
        depth=int(mc.get("depth", 2)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-6)),
    )
    batch_size = int(cfg["training"].get("batch_size", 512))
    max_steps = int(cfg["training"].get("max_steps", 1000))
    eval_interval = int(cfg["training"].get("eval_interval", 100))
    ce_weight = float(cfg["training"].get("ce_weight", 0.5))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    meta = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "dataset_npz": cfg["dataset_npz"],
        "device": str(device),
        "num_pairs": int(n),
        "raw_dim": int(raw_dim),
        "num_milestones": int(num_milestones),
        "train_pairs": int(len(train_idx)),
        "val_pairs": int(len(val_idx)),
        "model_params": count_params(model),
        "note": "Stage-1 LaWM-shaped transition model; prototype semantics are defined by the dataset export.",
    }
    write_json(run_dir / "run_meta.json", meta)
    print(json.dumps(meta, indent=2))

    metrics_path = run_dir / "metrics.jsonl"
    best_top1 = -1.0
    last: Metrics | None = None
    for step in range(1, max_steps + 1):
        model.train()
        b = sample_batch(train_idx, batch_size, device)
        out = model(data["current"][b], data["future"][b])
        mse = F.smooth_l1_loss(out["r_hat"], out["r_future"])
        ce = F.cross_entropy(out["logits"], data["future_milestone"][b])
        loss = mse + ce_weight * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step == 1 or step % eval_interval == 0 or step == max_steps:
            val_loss, val_mse, val_ce, top1, top3 = evaluate(model, data, val_idx, batch_size, ce_weight)
            last = Metrics(step, float(loss.item()), val_loss, val_mse, val_ce, top1, top3)
            row = asdict(last)
            append_jsonl(metrics_path, row)
            print(json.dumps(row), flush=True)
            if top1 > best_top1:
                best_top1 = top1
                save_checkpoint(ckpt_dir / "best.pt", model, cfg, meta, row)

    save_checkpoint(ckpt_dir / "last.pt", model, cfg, meta, asdict(last) if last else None)
    if last is not None:
        write_json(run_dir / "final_metrics.json", asdict(last))
    print(f"saved {ckpt_dir / 'best.pt'}")
    print(f"saved {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
