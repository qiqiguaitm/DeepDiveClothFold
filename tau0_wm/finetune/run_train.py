"""Distributed tau0 joint-space fine-tune loop (accelerate DDP, multi-node).

Per-GPU micro-batch=1 (tau0 WanModel.forward is single-sample) + gradient accumulation.
Data: visrobot01_train upsampled x3 + kairobot01 (per-embodiment stats), reusing the
verified-compatible GigaWorld latent + t5 caches. Phases P1 (freeze trunk) / P2 (unfreeze
action_blocks). Saves trainable-param checkpoints to a shared PFS dir.

Launch (2 nodes x 8 GPU = 16), from each node:
  accelerate launch --multi_gpu --num_machines 2 --num_processes 16 \
    --machine_rank {0|1} --main_process_ip 192.168.20.128 --main_process_port 29501 \
    finetune/run_train.py --phase p1_warm --max_steps 4000 --ckpt_dir <shared>
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finetune.model_joint import build_joint_wanmodel, set_trainable  # noqa: E402
from finetune.train_tau0 import TauFlowTrainer, ACTION_CHUNK  # noqa: E402
from finetune.data_joint import LatentJointDataset  # noqa: E402

DATA = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1"
ASSETS = os.path.join(ROOT, "finetune", "assets")
CKPT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/checkpoints/tau-0-wm"


def build_dataset(vis_upsample=3, include_kai=True):
    parts = []
    for _ in range(vis_upsample):
        parts.append(LatentJointDataset(f"{DATA}/visrobot01_train",
                                        f"{ASSETS}/statistics_visrobot01.json", ACTION_CHUNK, embed_id=0))
    if include_kai:
        parts.append(LatentJointDataset(f"{DATA}/kairobot01",
                                        f"{ASSETS}/statistics_kairobot01.json", ACTION_CHUNK, embed_id=1))
    return ConcatDataset(parts)


def collate(b):
    return b[0]  # micro-batch=1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="p1_warm", choices=["p1_warm", "p2_specialize", "all"])
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lambda_v", type=float, default=0.0)   # action-focused FT by default
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--ckpt_dir", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/tau0_fold_p1")
    ap.add_argument("--ckpt_interval", type=int, default=1000)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init_ckpt", default=CKPT, help="pretrained tau0 dir (or a prior phase ckpt)")
    ap.add_argument("--vis_upsample", type=int, default=3)
    ap.add_argument("--no_kai", action="store_true")
    ap.add_argument("--random_init", action="store_true", help="skip pretrained load (infra test only)")
    ap.add_argument("--warmup_steps", type=int, default=0, help=">0 enables warmup-cosine LR")
    ap.add_argument("--cosine_steps", type=int, default=0, help="cosine decay horizon (default=max_steps)")
    args = ap.parse_args()

    from accelerate import Accelerator
    accel = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    dev = accel.device
    is_main = accel.is_main_process

    def log(*a):
        if is_main:
            print("[train]", *a, flush=True)

    log(f"world_size={accel.num_processes} phase={args.phase} lr={args.lr} "
        f"grad_accum={args.grad_accum} lambda_a={args.lambda_a} lambda_v={args.lambda_v}")

    model, rep = build_joint_wanmodel(action_in_dim=14, ckpt_dir=args.init_ckpt,
                                      load_pretrained=not args.random_init,
                                      dtype=torch.float32, device="cpu", verbose=is_main)
    resume_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        # continue step/LR schedule when resuming from a step_<N>.pt checkpoint
        # (P2's final.pt → 0, i.e. fresh schedule; P3's step_8000.pt → 8000)
        import re as _re
        m = _re.search(r"step_(\d+)\.pt$", os.path.basename(args.resume))
        resume_step = int(m.group(1)) if m else 0
        log(f"resumed trainable params from {args.resume} (start_step={resume_step})")
    n_tr, n_all = set_trainable(model, args.phase)
    if model.config.get("gradient_checkpointing", False) is False:
        model.gradient_checkpointing = True   # activation ckpt to fit
    log(f"trainable={n_tr/1e6:.3f}M / {n_all/1e9:.3f}B")

    ds = build_dataset(args.vis_upsample, include_kai=not args.no_kai)
    log(f"dataset size (episodes, upsampled)={len(ds)}")
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-2)
    # manual warmup-cosine LR (set on param_groups each step) — robust under DeepSpeed/DDP
    import math as _m
    cos_T = args.cosine_steps or args.max_steps
    def compute_lr(s):
        if args.warmup_steps <= 0:
            return args.lr
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        p = min(1.0, (s - args.warmup_steps) / max(1, cos_T - args.warmup_steps))
        return args.lr * 0.5 * (1 + _m.cos(_m.pi * p))
    if args.warmup_steps > 0:
        log(f"warmup-cosine LR: warmup={args.warmup_steps} cosine_T={cos_T} peak={args.lr}")
    model, opt, dl = accel.prepare(model, opt, dl)
    raw = accel.unwrap_model(model)
    tr = TauFlowTrainer(raw, dev, lambda_v=args.lambda_v, lambda_a=args.lambda_a)
    model.train()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    step = resume_step
    t0 = time.time()
    running = 0.0
    running_v = 0.0
    data_iter = iter(dl)
    while step < args.max_steps:
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            b = next(data_iter)
        with accel.accumulate(model):
            z0 = b["video_latent"].to(dev, torch.bfloat16)
            ref = b["ref"].to(dev, torch.bfloat16)
            a0 = b["action"].unsqueeze(0).to(dev, torch.bfloat16)
            state = b["state"].unsqueeze(0).to(dev, torch.bfloat16)
            ctx = b["t5"].to(dev, torch.bfloat16)
            loss, parts = tr.forward_step(z0, a0, state, ctx, ref=ref)
            accel.backward(loss)
            if accel.sync_gradients and args.warmup_steps > 0:
                lr_now = compute_lr(step)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            opt.step()
            opt.zero_grad()
        running += parts["a_loss"]
        running_v += parts.get("v_loss", 0.0)
        if accel.sync_gradients:
            step += 1
            if step % args.log_interval == 0:
                sps = (step - resume_step) / (time.time() - t0)
                cur_lr = opt.param_groups[0]["lr"]
                denom = args.log_interval * args.grad_accum
                log(f"step {step}/{args.max_steps}  a_loss={running/denom:.4f}  v_loss={running_v/denom:.4f}  "
                    f"lr={cur_lr:.2e}  {sps:.2f} step/s")
                running = 0.0
                running_v = 0.0
            if is_main and step % args.ckpt_interval == 0:
                full = accel.unwrap_model(model).state_dict()
                sd = full if args.phase == "all" else {k: v for k, v in full.items()
                      if k.startswith(("action_proj_in", "action_head", "action_blocks", "action_time"))}
                path = os.path.join(args.ckpt_dir, f"step_{step}.pt")
                torch.save(sd, path)
                log(f"saved {path} ({len(sd)} tensors)")
    if is_main:
        full = accel.unwrap_model(model).state_dict()
        sd = full if args.phase == "all" else {k: v for k, v in full.items()
              if k.startswith(("action_proj_in", "action_head", "action_blocks", "action_time"))}
        torch.save(sd, os.path.join(args.ckpt_dir, "final.pt"))
        log("done.")


if __name__ == "__main__":
    main()
