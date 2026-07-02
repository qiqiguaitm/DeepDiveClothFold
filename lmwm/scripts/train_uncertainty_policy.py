#!/usr/bin/env python
"""Train a lightweight learned uncertainty policy for LMWM fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_features(z: np.lib.npyio.NpzFile, head: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    current = z["current_milestone"].astype(np.int64)
    transition_probs = z["transition_probs"].astype(np.float32)
    num_milestones = transition_probs.shape[1]
    onehot = np.eye(num_milestones, dtype=np.float32)[current]
    transition_conf = transition_probs.max(axis=1, keepdims=True)
    entropy = z["transition_entropy"].astype(np.float32)[:, None]
    if head == "greedy":
        conf = z["greedy_confidence"].astype(np.float32)[:, None]
        correct = z["greedy_pred"].astype(np.int64) == z["greedy_target"].astype(np.int64)
    elif head == "max_product":
        conf = z["max_product_confidence"].astype(np.float32)[:, None]
        correct = z["max_product_pred"].astype(np.int64) == z["max_product_target"].astype(np.int64)
    else:
        raise ValueError(f"unsupported head={head}")
    x = np.concatenate([conf, entropy, transition_conf, onehot], axis=1).astype(np.float32)
    y_error = (~correct).astype(np.float32)
    names = ["head_confidence", "transition_entropy", "transition_confidence"] + [
        f"current_milestone_{i}" for i in range(num_milestones)
    ]
    return x, y_error, names


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
) -> tuple[np.ndarray, float, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xtr = torch.from_numpy(x_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    xva = torch.from_numpy(x_val).to(device)
    yva = torch.from_numpy(y_val).to(device)
    w = torch.zeros(x_train.shape[1], device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    best = {"val_loss": float("inf"), "w": None, "b": None, "step": 0}
    for step in range(1, steps + 1):
        logits = xtr @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, ytr, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                val_loss = F.binary_cross_entropy_with_logits(xva @ w + b, yva).item()
            if val_loss < best["val_loss"]:
                best = {
                    "val_loss": float(val_loss),
                    "w": w.detach().cpu().numpy().copy(),
                    "b": float(b.detach().cpu().item()),
                    "step": step,
                }
    return best["w"], best["b"], {"best_step": best["step"], "best_val_loss": best["val_loss"]}


def eval_policy(prob: np.ndarray, y_error: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    correct = 1.0 - y_error
    for t in thresholds:
        fallback = prob >= t
        accepted = ~fallback
        accepted_acc = float(correct[accepted].mean()) if accepted.any() else float("nan")
        fallback_recall = float(y_error[fallback].sum() / max(y_error.sum(), 1.0))
        rows.append({
            "error_threshold": float(t),
            "fallback_rate": float(fallback.mean()),
            "accepted_accuracy": accepted_acc,
            "error_recall": fallback_recall,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.predictions)
    rng = np.random.default_rng(args.seed)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    summary = {"predictions": str(args.predictions), "heads": {}}
    policy = {"feature_names": None, "heads": {}}

    for head in ["greedy", "max_product"]:
        x, y, names = make_features(z, head)
        idx = rng.permutation(len(x))
        n_val = max(1, int(round(len(x) * args.val_ratio)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        w, b, fit = train_logistic(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        prob = 1.0 / (1.0 + np.exp(-(x[val_idx] @ w + b)))
        rows = eval_policy(prob, y[val_idx], thresholds)
        best_under_20 = max(
            [r for r in rows if r["fallback_rate"] <= 0.20],
            key=lambda r: (r["accepted_accuracy"], r["error_recall"]),
            default=None,
        )
        summary["heads"][head] = {
            **fit,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "val_error_rate": float(y[val_idx].mean()),
            "threshold_curve": rows,
            "best_under_20pct_fallback": best_under_20,
        }
        policy["feature_names"] = names
        policy["heads"][head] = {
            "weights": w.astype(np.float32).tolist(),
            "bias": float(b),
            "recommended_error_threshold": float(best_under_20["error_threshold"] if best_under_20 else 0.5),
        }

    np.savez_compressed(
        args.output_dir / "uncertainty_policy.npz",
        feature_names=np.array(policy["feature_names"], dtype=object),
        greedy_weights=np.array(policy["heads"]["greedy"]["weights"], dtype=np.float32),
        greedy_bias=np.array(policy["heads"]["greedy"]["bias"], dtype=np.float32),
        greedy_error_threshold=np.array(policy["heads"]["greedy"]["recommended_error_threshold"], dtype=np.float32),
        max_product_weights=np.array(policy["heads"]["max_product"]["weights"], dtype=np.float32),
        max_product_bias=np.array(policy["heads"]["max_product"]["bias"], dtype=np.float32),
        max_product_error_threshold=np.array(
            policy["heads"]["max_product"]["recommended_error_threshold"], dtype=np.float32
        ),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
