#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen Inverse-Dynamics (IDM) probe for wam_fold_v3 cloth-fold video.

L0 of the world-model controllability eval: measure the "action-observability
ceiling" — how recoverable the (normalized delta-) action is from REAL video.

Classic 2-frame inverse dynamics:  a_t = f(o_t, o_{t+1}).
Per dataset window the video has chunk_length+1 frames -> chunk_length
consecutive (frame-pair, action) training examples (one decode -> 32 pairs).

Input to the net: [frame_t, frame_{t+1}, frame_diff] = 9 RGB channels, resized
to 224x224 (concat_view: overhead on top, two wrist cams on bottom). RGB-based
(NOT VAE-latent) so the same frozen probe can later judge videos from different
world models (Cosmos3 Wan-VAE vs Ctrl-World SVD-VAE) fairly.

Model: torchvision resnet18 (conv1 -> 9ch, fc -> 14), GAP -> 14-D. MSE loss vs
the dataset's already-quantile-normalized 14-D delta-action target.

Reports (NORMALIZED action units, ~[-1,1] quantile-normalized delta):
  1. IDM-MAE (mean over 14 dims)  = action-observability ceiling
  2. baseline MAE: predict-zeros and predict-per-dim-train-mean
  3. per-dim MAE breakdown (arm joints vs grippers)
Per rig (visrobot01 / kairobot01).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from cosmos_framework.data.vfm.action.datasets.wam_fold_dataset import WamFoldLeRobotDataset

DATA_ROOT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3"
OUT_DIR = Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/eval/idm")
CHUNK = 32
RES = 224
# index layout: per arm 6 joints + 1 gripper, x2 arms. grippers at 6 and 13.
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPER_DIMS = [6, 13]
DIM_NAMES = (
    [f"L_arm_j{i}" for i in range(6)] + ["L_gripper"]
    + [f"R_arm_j{i}" for i in range(6)] + ["R_gripper"]
)


class IDMWindowWrapper(Dataset):
    """Wrap WamFoldLeRobotDataset: return (resized_frames[T,3,RES,RES] uint8, action[CHUNK,14])."""

    def __init__(self, base: WamFoldLeRobotDataset, res: int = RES):
        self.base = base
        self.res = res

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        video = item["video"]  # uint8 [C=3, T=CHUNK+1, 720, 640]
        v = video.permute(1, 0, 2, 3).float()  # [T,3,H,W]
        v = F.interpolate(v, size=(self.res, self.res), mode="bilinear", align_corners=False)
        v = v.round().clamp(0, 255).to(torch.uint8)  # [T,3,RES,RES]
        return v, item["action"]  # action [CHUNK,14] float


def build_datasets(rig: str):
    common = dict(mode="forward_dynamics", chunk_length=CHUNK, fps=30.0)
    if rig == "visrobot01":
        train_base = WamFoldLeRobotDataset(rig=rig, root=f"{DATA_ROOT}/visrobot01_v3_train", **common)
        val_base = WamFoldLeRobotDataset(rig=rig, root=f"{DATA_ROOT}/visrobot01_v3_val", **common)
        train_idx = list(range(len(train_base)))
        val_idx = list(range(len(val_base)))
        return train_base, val_base, train_idx, val_idx
    elif rig == "kairobot01":
        base = WamFoldLeRobotDataset(rig=rig, root=f"{DATA_ROOT}/kairobot01_v3", **common)
        file_ids = sorted({f for f, _ in base._windows})
        n_val = max(1, int(0.10 * len(file_ids)))
        val_files = set(file_ids[-n_val:])
        train_idx = [i for i, (f, _) in enumerate(base._windows) if f not in val_files]
        val_idx = [i for i, (f, _) in enumerate(base._windows) if f in val_files]
        return base, base, train_idx, val_idx
    raise KeyError(rig)


def make_model() -> nn.Module:
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 14)
    return m


def pairs_from_batch(frames: torch.Tensor, action: torch.Tensor, device):
    """frames [B,T,3,RES,RES] uint8, action [B,CHUNK,14] -> (x[N,9,RES,RES], y[N,14])."""
    frames = frames.to(device, non_blocking=True).float().div_(255.0)
    f_t = frames[:, :CHUNK]      # [B,CHUNK,3,H,W]
    f_t1 = frames[:, 1:]         # [B,CHUNK,3,H,W]
    diff = f_t1 - f_t
    # center frames to ~[-0.5,0.5]; diff already in [-1,1]
    x = torch.cat([f_t - 0.5, f_t1 - 0.5, diff], dim=2)  # [B,CHUNK,9,H,W]
    B = x.shape[0]
    x = x.reshape(B * CHUNK, 9, x.shape[-2], x.shape[-1])
    y = action.to(device, non_blocking=True).reshape(B * CHUNK, 14)
    return x, y


@torch.no_grad()
def evaluate(model, loader, device, max_pairs: int):
    model.eval()
    abs_err = torch.zeros(14, device=device)
    sq_targets_sum = torch.zeros(14, device=device)  # for zero-baseline MAE = mean|y|
    n = 0
    for frames, action in loader:
        x, y = pairs_from_batch(frames, action, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).float()
        abs_err += (pred - y).abs().sum(0)
        sq_targets_sum += y.abs().sum(0)
        n += y.shape[0]
        if n >= max_pairs:
            break
    model.train()
    return (abs_err / n).cpu().numpy(), (sq_targets_sum / n).cpu().numpy(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True, choices=["visrobot01", "kairobot01"])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch_items", type=int, default=8)  # *CHUNK pairs per step
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_pairs", type=int, default=12000)
    ap.add_argument("--mean_est_items", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    t0 = time.time()

    train_base, val_base, train_idx, val_idx = build_datasets(args.rig)
    print(f"[{args.rig}] train windows={len(train_idx)} val windows={len(val_idx)}", flush=True)

    train_ds = Subset(IDMWindowWrapper(train_base), train_idx)
    val_ds = Subset(IDMWindowWrapper(val_base), val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_items, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_items, shuffle=True, num_workers=6,
        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2,
    )

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    loss_fn = nn.L1Loss()  # train directly on MAE objective (probe); robust

    # ---- estimate per-dim TRAIN mean (for mean-baseline) ----
    mean_loader = DataLoader(
        Subset(IDMWindowWrapper(train_base), train_idx[: max(1, len(train_idx))]),
        batch_size=args.batch_items, shuffle=True, num_workers=6, drop_last=False,
    )
    msum = np.zeros(14, dtype=np.float64)
    mcnt = 0
    for frames, action in mean_loader:
        a = action.reshape(-1, 14).numpy()
        msum += a.sum(0)
        mcnt += a.shape[0]
        if mcnt >= args.mean_est_items * CHUNK:
            break
    train_mean = (msum / mcnt).astype(np.float32)
    print(f"[{args.rig}] train-mean estimated on {mcnt} actions", flush=True)

    # ---- train ----
    model.train()
    step = 0
    run_loss = 0.0
    done = False
    while not done:
        for frames, action in train_loader:
            x, y = pairs_from_batch(frames, action, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x).float()
                loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"[{args.rig}] step {step}/{args.steps} loss(L1)={run_loss/50:.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                run_loss = 0.0
            if step >= args.steps:
                done = True
                break

    # ---- evaluate ----
    idm_mae, zero_mae, n_val = evaluate(model, val_loader, device, args.val_pairs)
    # mean-baseline MAE per dim = mean|y - train_mean|; recompute over the same val stream
    model.eval()
    mean_abs = torch.zeros(14, device=device)
    nm = 0
    tm = torch.tensor(train_mean, device=device)
    with torch.no_grad():
        for frames, action in val_loader:
            y = action.to(device).reshape(-1, 14)
            mean_abs += (y - tm).abs().sum(0)
            nm += y.shape[0]
            if nm >= args.val_pairs:
                break
    mean_mae = (mean_abs / nm).cpu().numpy()

    idm_overall = float(idm_mae.mean())
    zero_overall = float(zero_mae.mean())
    mean_overall = float(mean_mae.mean())
    arm_idm = float(idm_mae[ARM_DIMS].mean())
    grip_idm = float(idm_mae[GRIPPER_DIMS].mean())

    print("\n==================== RESULTS:", args.rig, "====================", flush=True)
    print(f"val pairs evaluated: {n_val}", flush=True)
    print(f"IDM-MAE (overall, 14-dim): {idm_overall:.4f}", flush=True)
    print(f"  baseline zero-pred MAE : {zero_overall:.4f}", flush=True)
    print(f"  baseline mean-pred MAE : {mean_overall:.4f}", flush=True)
    print(f"  arm-joint IDM-MAE : {arm_idm:.4f}   gripper IDM-MAE : {grip_idm:.4f}", flush=True)
    print("  per-dim [name: idm / zero / mean]:", flush=True)
    for d in range(14):
        print(f"    {DIM_NAMES[d]:>10}: {idm_mae[d]:.4f} / {zero_mae[d]:.4f} / {mean_mae[d]:.4f}", flush=True)

    metrics = {
        "rig": args.rig,
        "normalization": "quantile-normalized DELTA action (arm=delta vs window-anchor, "
                         "grippers absolute), ~[-1,1]; MAE in these normalized units",
        "steps": args.steps,
        "n_val_pairs": int(n_val),
        "n_train_windows": len(train_idx),
        "n_val_windows": len(val_idx),
        "idm_mae_overall": idm_overall,
        "baseline_zero_mae_overall": zero_overall,
        "baseline_mean_mae_overall": mean_overall,
        "idm_mae_arm_joints": arm_idm,
        "idm_mae_grippers": grip_idm,
        "improvement_vs_zero_pct": float(100.0 * (zero_overall - idm_overall) / zero_overall),
        "improvement_vs_mean_pct": float(100.0 * (mean_overall - idm_overall) / mean_overall),
        "per_dim": {
            DIM_NAMES[d]: {
                "idm_mae": float(idm_mae[d]),
                "zero_mae": float(zero_mae[d]),
                "mean_mae": float(mean_mae[d]),
            }
            for d in range(14)
        },
        "train_mean": train_mean.tolist(),
        "elapsed_sec": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"metrics_{args.rig}.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), OUT_DIR / f"idm_{args.rig}.pt")
    print(f"\nsaved metrics -> {OUT_DIR / f'metrics_{args.rig}.json'}", flush=True)
    print(f"saved model   -> {OUT_DIR / f'idm_{args.rig}.pt'}", flush=True)


if __name__ == "__main__":
    main()
