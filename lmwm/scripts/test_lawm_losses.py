#!/usr/bin/env python
"""Exercise LaWM/LAM's loss functions and apply its loss recipe to our LMWM.

LaWM's full training is not runnable locally (needs their dataset + DINOv3
weights at placeholder /mnt/xx/xx paths), so this instead:
  1. imports LaWM's actual loss fns (charbonnier_loss, eef_reconstruction_loss)
     and verifies they run;
  2. computes every LaWM reconstruction-loss variant on OUR LMWM Stage-3
     real-future greedy_proto predictions vs the true next-milestone prototype,
     on the held-out episode split -- so we see what each loss "sees" on our data;
  3. compares smooth_l1 beta=1.0 (our current) vs beta=0.1 (LaWM) gradient scale.

Usage:
    python lmwm/scripts/test_lawm_losses.py \
        --checkpoint lmwm/checkpoints/stage3_realfuture/<run>/best.pt \
        --pairs lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz \
        --out lmwm/outputs/lawm_loss_probe/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402

# Load LaWM's loss utils directly by file path (its package __init__ pulls in
# lightning, which we don't need and isn't installed here).
_lawm_utils_path = Path(__file__).resolve().parents[1] / "vendor/LaWAM/latent_action_model/core/utils/utils.py"
_spec = importlib.util.spec_from_file_location("lawm_utils", _lawm_utils_path)
_lawm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lawm)
charbonnier_loss = _lawm.charbonnier_loss
eef_reconstruction_loss = _lawm.eef_reconstruction_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    # ---- 1. verify LaWM loss fns run on synthetic tensors ----
    a = torch.randn(8, 32)
    b = torch.randn(8, 32)
    smoke = {
        "charbonnier_eps1e-3": float(charbonnier_loss(a, b, eps=1e-3)),
        "eef_l2_masked": float(eef_reconstruction_loss(a, b, torch.ones_like(a), state_loss_type="l2")),
        "eef_l1_masked": float(eef_reconstruction_loss(a, b, torch.ones_like(a), state_loss_type="l1")),
    }

    # ---- 2. LaWM loss variants on our real LMWM proto predictions ----
    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck["config"]
    meta = ck.get("meta", {})
    g = np.load(args.graph_npz)
    proto = g["prototype_table"].astype(np.float32)
    num_m, latent_dim = proto.shape
    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, val_idx = split_indices(z, n, float(cfg["training"].get("val_ratio", 0.2)),
                               int(cfg.get("seed", 2026)), torch.device("cpu"), str(cfg.get("split_mode", "random")))
    vi = val_idx.numpy()
    feats = z["current"][vi].astype(np.float32)
    future_m = z["future_milestone"][vi].astype(np.int64)
    tgt = torch.from_numpy(proto[future_m])  # true next-milestone prototype

    in_dim = int(meta.get("input_dim", feats.shape[1]))
    mc = cfg.get("model", {})
    model = UnifiedLMWM(in_dim, latent_dim, num_m, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2)))
    model.load_state_dict(ck["model"]); model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(feats), 8192):
            preds.append(model(torch.from_numpy(feats[s:s + 8192]))["greedy_proto"])
    recon = torch.cat(preds)  # L2-normalized predicted subgoal latent

    cos = F.cosine_similarity(recon, tgt, dim=-1).mean().item()
    proto_losses = {
        "smooth_l1_beta1.0_(LMWM current)": F.smooth_l1_loss(recon, tgt, beta=1.0).item(),
        "smooth_l1_beta0.1_(LaWM)": F.smooth_l1_loss(recon, tgt, beta=0.1).item(),
        "l1": F.l1_loss(recon, tgt).item(),
        "l2_mse": F.mse_loss(recon, tgt).item(),
        "charbonnier_eps1e-3_(LaWM)": float(charbonnier_loss(recon, tgt, eps=1e-3)),
        "cos_loss_1_minus_cossim_(LaWM 'cos')": float(F.smooth_l1_loss(recon, tgt, beta=0.1) + (1 - cos)),
        "cosine_similarity_metric": cos,
    }

    # ---- 3. beta gradient-scale comparison (why LaWM picks 0.1) ----
    # smooth_l1 gradient magnitude is |x|/beta (capped at 1). Smaller beta => full
    # gradient kicks in at smaller errors => sharper feature-regression signal.
    err = (recon - tgt).abs()
    grad_beta = {}
    for beta in (0.1, 1.0):
        gmag = torch.clamp(err / beta, max=1.0).mean().item()
        grad_beta[f"mean_grad_mag_beta{beta}"] = gmag
    grad_beta["median_abs_err"] = float(err.median())
    grad_beta["frac_err_below_0.1"] = float((err < 0.1).float().mean())
    grad_beta["note"] = ("LaWM beta=0.1: most per-dim errors are tiny (DINO features), so beta=1.0 "
                         "keeps them in the quadratic regime with weak gradient; beta=0.1 gives full "
                         "gradient earlier -> sharper regression.")

    summary = {
        "held_out_pairs": int(len(vi)),
        "latent_dim": int(latent_dim),
        "lawm_loss_smoke_test": smoke,
        "proto_losses_on_our_data": proto_losses,
        "beta_gradient_analysis": grad_beta,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
