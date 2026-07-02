#!/usr/bin/env python
"""Phase B — calibration + graph-as-prior fusion under the real-future criterion.

The Phase A real-future greedy head is already a frame-conditional next-milestone
distribution (softmax trained by CE on observed futures). Phase B asks two honest
questions on the held-out episode split, scored against the REAL observed future:

1. Calibration: is the predicted confidence trustworthy? (reliability, ECE,
   Brier), and does a single temperature fix it (fit on a calib half, test on the
   other)?
2. Graph as prior (not label, not hard fallback): does a log-linear pool
   ``log p = (1-lam)*log p_neural + lam*log p_prior`` beat neural-only or
   prior-only on real-future NLL / top-k? ``p_prior`` is the empirical
   milestone-level transition row.

Usage:
    python lmwm/scripts/eval_phase_b.py \
        --checkpoint lmwm/checkpoints/stage3_realfuture/<run>/best.pt \
        --output_dir lmwm/outputs/phase_b_eval/<name> [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402


def topk_acc(probs: np.ndarray, target: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=min(k, probs.shape[1] - 1), axis=1)[:, :k]
    return float((topk == target[:, None]).any(axis=1).mean())


def nll(probs: np.ndarray, target: np.ndarray) -> float:
    p = probs[np.arange(len(target)), target]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def dist_metrics(probs: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"top1": topk_acc(probs, target, 1), "top3": topk_acc(probs, target, 3),
            "top5": topk_acc(probs, target, 5), "nll": nll(probs, target)}


def brier(probs: np.ndarray, target: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(target)), target] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ece(probs: np.ndarray, target: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == target).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(target)
    e = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "conf": None, "acc": None})
            continue
        bconf = float(conf[m].mean())
        bacc = float(correct[m].mean())
        e += (cnt / total) * abs(bconf - bacc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": cnt, "conf": bconf, "acc": bacc})
    return float(e), rows


def fit_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    """Grid + local refine for the scalar T minimizing NLL of softmax(logits/T)."""
    t = torch.from_numpy(logits.astype(np.float64))
    y = torch.from_numpy(target.astype(np.int64))
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 8.0, 26)]):
        loss = F.cross_entropy(t / float(T), y).item()
        if loss < best_nll:
            best_nll, best_T = loss, float(T)
    return best_T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck["config"]
    meta = ck.get("meta", {})
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    z = np.load(cfg["dataset_npz"])
    g = np.load(cfg["graph_npz"])
    n = len(z["current_milestone"])
    seed = int(cfg.get("seed", 2026))
    val_ratio = float(cfg["training"].get("val_ratio", 0.2))
    split_mode = str(cfg.get("split_mode", "random"))
    _, val_idx = split_indices(z, n, val_ratio, seed, device, split_mode)
    vi = val_idx.cpu().numpy()

    current_m = z["current_milestone"].astype(np.int64)[vi]
    future_m = z["future_milestone"].astype(np.int64)[vi]
    feats = z["current"].astype(np.float32)[vi]
    transition_probs = g["transition_probs"].astype(np.float32)
    num_m = int(transition_probs.shape[0])

    latent_dim = int(meta.get("latent_dim", g["prototype_table"].shape[1]))
    in_dim = int(meta.get("input_dim", feats.shape[1]))
    mc = cfg.get("model", {})
    model = UnifiedLMWM(in_dim, latent_dim, num_m, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    logits_list = []
    with torch.no_grad():
        for s in range(0, len(feats), 8192):
            x = torch.from_numpy(feats[s:s + 8192]).to(device)
            logits_list.append(model(x)["greedy_logits"].cpu().numpy())
    logits = np.concatenate(logits_list).astype(np.float64)
    p_neural = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    p_prior = transition_probs[current_m].astype(np.float64)
    p_prior = p_prior / p_prior.sum(axis=1, keepdims=True).clip(1e-12)

    # --- calibration of the frame-conditional neural distribution ---
    e_neural, bins_neural = ece(p_neural, future_m)
    calib = {
        "neural_ece": e_neural,
        "neural_brier": brier(p_neural, future_m),
        "reliability_bins": bins_neural,
    }

    # temperature scaling: fit T on a deterministic calib half, test on the other
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(vi))
    half = len(vi) // 2
    ci, ti = perm[:half], perm[half:]
    T = fit_temperature(logits[ci], future_m[ci])
    p_test_raw = torch.softmax(torch.from_numpy(logits[ti]), dim=-1).numpy()
    p_test_T = torch.softmax(torch.from_numpy(logits[ti] / T), dim=-1).numpy()
    e_raw, _ = ece(p_test_raw, future_m[ti])
    e_T, _ = ece(p_test_T, future_m[ti])
    calib["temperature"] = {
        "fitted_T": T,
        "test_ece_before": e_raw,
        "test_ece_after": e_T,
        "test_nll_before": nll(p_test_raw, future_m[ti]),
        "test_nll_after": nll(p_test_T, future_m[ti]),
    }

    # --- graph-as-prior log-linear fusion sweep ---
    ln_neural = np.log(np.clip(p_neural, 1e-12, 1.0))
    ln_prior = np.log(np.clip(p_prior, 1e-12, 1.0))
    fusion = []
    for lam in np.round(np.linspace(0.0, 1.0, 11), 2):
        logp = (1.0 - lam) * ln_neural + lam * ln_prior
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        m = dist_metrics(p, future_m)
        m["lam"] = float(lam)
        fusion.append(m)
    best = min(fusion, key=lambda r: r["nll"])

    summary = {
        "checkpoint": str(args.checkpoint),
        "label_source_trained": meta.get("label_source"),
        "held_out_pairs": int(len(vi)),
        "num_milestones": num_m,
        "calibration": calib,
        "prior_fusion_sweep": fusion,
        "prior_fusion_best": best,
        "neural_only": fusion[0],   # lam=0
        "prior_only": fusion[-1],   # lam=1
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
