"""Distributed eval (accelerate, multi-node) of a tau0 joint checkpoint on visrobot01_val.

Shards the 200 val episodes across ranks; each rank averages the action flow-loss over
sampled windows; results are all-reduced. Reports overall mean val action-loss + per-dim,
and writes JSON on rank 0. Used for the P1 go/no-go ablation and P2 final eval, on 2 nodes x 8 GPU.

Launch (16 GPU = 2 nodes): see finetune/launch_eval_2node.sh
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finetune.model_joint import build_joint_wanmodel  # noqa: E402
from finetune.train_tau0 import TauFlowTrainer, ACTION_CHUNK, ACTION_DIM  # noqa: E402
from finetune.data_joint import LatentJointDataset  # noqa: E402

VAL = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val"
ASSETS = os.path.join(ROOT, "finetune", "assets")
CKPT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/checkpoints/tau-0-wm"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="trained joint-head/action ckpt (.pt)")
    ap.add_argument("--random_trunk", action="store_true", help="control: random-init trunk")
    ap.add_argument("--windows_per_ep", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/eval_report.json")
    ap.add_argument("--tag", default="p1")
    args = ap.parse_args()

    from accelerate import Accelerator
    accel = Accelerator(mixed_precision="bf16")
    dev = accel.device
    R, W = accel.process_index, accel.num_processes
    is_main = accel.is_main_process

    def log(*a):
        if is_main:
            print("[eval]", *a, flush=True)

    log(f"world={W} ckpt={args.ckpt or '(none)'} random_trunk={args.random_trunk} tag={args.tag}")
    model, _ = build_joint_wanmodel(action_in_dim=ACTION_DIM, ckpt_dir=CKPT,
                                    load_pretrained=not args.random_trunk,
                                    dtype=torch.bfloat16, device=str(dev), verbose=is_main)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        miss, unexp = model.load_state_dict(sd, strict=False)
        log(f"loaded ckpt: {len(sd)} tensors (missing {len(miss)}, unexpected {len(unexp)})")
    model.eval()
    tr = TauFlowTrainer(model, dev)

    ds = LatentJointDataset(VAL, f"{ASSETS}/statistics_visrobot01.json", ACTION_CHUNK, embed_id=0)
    # shard episodes across ranks
    my_eps = list(range(R, len(ds), W))
    log(f"episodes total={len(ds)}; this rank evaluates {len(my_eps)}")

    vel_sq = torch.zeros(ACTION_DIM, device=dev, dtype=torch.float64)   # velocity MSE per-dim (ablation)
    den_sq = torch.zeros(ACTION_DIM, device=dev, dtype=torch.float64)   # denoised action MSE per-dim (norm)
    hor_sq = torch.zeros(ACTION_CHUNK, device=dev, dtype=torch.float64)  # denoised MSE per horizon
    cnt = torch.zeros(1, device=dev, dtype=torch.float64)
    a_std = torch.tensor(ds.stats["a_std"], device=dev, dtype=torch.float64)  # for physical (rad) MSE
    rng = np.random.RandomState(1234 + R)
    for ei in my_eps:
        ep = ds._load_episode(ei)
        nw = ep["visual"].shape[0]
        for _ in range(args.windows_per_ep):
            w = rng.randint(0, nw)
            t = ep["starts"][w]
            n = len(ep["s"])
            idx = np.clip(np.arange(t, t + ACTION_CHUNK), 0, n - 1)
            from finetune.data_joint import proprio_sample
            ns, na = proprio_sample(ep["s"][t], ep["a"][idx], ds.stats)
            z0 = ep["visual"][w].float().to(dev, torch.bfloat16)
            ref = ep["ref"][w].float().to(dev, torch.bfloat16)
            a0 = torch.from_numpy(na).unsqueeze(0).to(dev, torch.bfloat16)
            state = torch.from_numpy(ns).unsqueeze(0).to(dev, torch.bfloat16)
            ctx = ep["t5"].to(dev, torch.bfloat16)
            for _s in range(args.seeds):
                # recompute per-dim squared velocity error
                tsr, sig = tr._sigma(1)
                mask = torch.ones_like(z0); mask[:, 0] = 0
                z0c = z0.clone(); z0c[:, 0:1] = ref
                nv = torch.randn_like(z0c); nap = torch.randn_like(a0)
                at = (nap * sig.view(1, 1, 1) + a0 * (1 - sig.view(1, 1, 1))).to(a0.dtype)
                zt = ((1 - mask) * z0c + mask * (nv * sig.view(1, 1, 1, 1) + z0c * (1 - sig.view(1, 1, 1, 1)))).to(z0c.dtype)
                a_tar = nap - a0
                ts_val = float(tsr.item())
                temp = (mask[0][:, ::2, ::2] * ts_val).flatten()
                if temp.numel() < tr.seq_len:
                    temp = torch.cat([temp, temp.new_full((tr.seq_len - temp.numel(),), ts_val)])
                v_ts = temp[: tr.seq_len].unsqueeze(0)
                a_ts = tsr.view(1, 1).repeat(1, ACTION_CHUNK).float()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x=[zt], t=v_ts, context=[ctx], seq_len=tr.seq_len,
                                action_states=at, action_timestep=a_ts, return_video=True,
                                return_action=True, store_buffer=False, history_action_state=state)
                ap_pred = out["action"].float()
                a0f = a0.float()
                # velocity MSE (training objective)
                vel_sq += ((ap_pred - a_tar.float()) ** 2).mean(dim=(0, 1)).double()
                # denoised action = noised - sigma*velocity  (GigaWorld action_mse style)
                pred_clean = at.float() - float(sig.item()) * ap_pred
                d2 = (pred_clean - a0f) ** 2                     # [1,33,14]
                den_sq += d2.mean(dim=(0, 1)).double()           # per-dim [14]
                hor_sq += d2.mean(dim=(0, 2)).double()           # per-horizon [33]
                cnt += 1

    accel.wait_for_everyone()
    for tns in (vel_sq, den_sq, hor_sq, cnt):
        torch.distributed.all_reduce(tns)
    vel_per_dim = (vel_sq / cnt).cpu().numpy()
    den_per_dim = (den_sq / cnt).cpu().numpy()
    den_phys_per_dim = (den_per_dim * (a_std.cpu().numpy() ** 2))   # physical (rad^2) MSE per-dim
    hor = (hor_sq / cnt).cpu().numpy()
    overall = float(vel_per_dim.mean())
    if is_main:
        import math
        res = {"tag": args.tag, "ckpt": args.ckpt, "random_trunk": args.random_trunk,
               "world_size": W, "n_window_evals": int(cnt.item()),
               "val_action_loss_mean": overall,
               "val_action_loss_per_dim": [round(float(x), 5) for x in vel_per_dim],
               "action_mse_norm_mean": float(den_per_dim.mean()),
               "action_mse_norm_per_dim": [round(float(x), 5) for x in den_per_dim],
               "action_mse_phys_mean_rad2": float(den_phys_per_dim.mean()),
               "action_rmse_phys_mean_rad": float(math.sqrt(den_phys_per_dim.mean())),
               "action_mse_phys_per_dim_rad2": [round(float(x), 6) for x in den_phys_per_dim],
               "action_mse_per_horizon": [round(float(x), 5) for x in hor]}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        # append to a list file
        allres = []
        if os.path.exists(args.out):
            try:
                allres = json.load(open(args.out))
            except Exception:
                allres = []
        allres.append(res)
        json.dump(allres, open(args.out, "w"), indent=2)
        print(f"[eval] === RESULT tag={args.tag} ===", flush=True)
        print(f"[eval] velocity-loss={overall:.4f}  action_mse(norm)={res['action_mse_norm_mean']:.4f}  "
              f"action_RMSE(phys)={res['action_rmse_phys_mean_rad']:.4f} rad  over {int(cnt.item())} evals (16 GPU)", flush=True)
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
