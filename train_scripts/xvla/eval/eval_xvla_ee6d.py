#!/usr/bin/env python
"""Eval an X-VLA EE6D cloth-fold checkpoint: action-chunk MAE on a fixed held-out
val subset of the vis domain (A_0423_0527).

Controlled-variable ablation across X3.C/X3.B/X3.A: identical val windows + metric,
only the trained checkpoint differs.

Model construction mirrors train_scripts/xvla/launch/xvla_train.py exactly:
    model = XVLAPolicy.from_pretrained(CKPT_INIT).to(device)
    model.load_state_dict(torch.load(ckpt)["model_state"])

Val window selection (FIXED, identical across all 3 ckpts):
    - dataset = LeRobotEE6DDataset(val_root, domain_id) -> flat list of (ep_idx,f_idx)
      windows in episode order.
    - held-out region = samples whose ep_idx is in the LAST 50 episodes
      (ep_idx >= total_episodes - 50).
    - from that contiguous region take a deterministic strided subset of n_windows:
      stride = max(1, len(region)//n_windows), windows = region[::stride][:n_windows].
    - NO randomness in window selection.

Determinism of the flow-matching denoiser: generate_actions() draws random init
noise x1 ~ randn. To make all 3 models see IDENTICAL noise per batch we set
torch.manual_seed(GEN_SEED + batch_index) right before each predict call.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
# train_scripts/xvla/eval/eval_xvla_ee6d.py -> data dir is ../data
sys.path.insert(0, str(HERE.parent.parent / "data"))
from multi_domain_dataset import LeRobotEE6DDataset  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CKPT_INIT = "/data/shared/ubuntu/workspace/xvla_ckpts"
PROMPT = "Flatten and fold the cloth."
GEN_SEED = 12345
N_HELDOUT_EP = 50
HORIZONS = [1, 10, 25, 30]


def build_collate():
    tok = AutoTokenizer.from_pretrained("facebook/bart-large")
    cached_tokens = tok(
        [PROMPT], padding="max_length", max_length=50, truncation=True,
        return_tensors="pt",
    )["input_ids"][0]

    def collate(batch):
        out = {}
        for k in batch[0].keys():
            if isinstance(batch[0][k], torch.Tensor):
                out[k] = torch.stack([s[k] for s in batch])
        out["observation.language.tokens"] = (
            cached_tokens.unsqueeze(0).expand(len(batch), -1).contiguous()
        )
        return out

    return collate


def select_windows(ds, n_windows):
    total_ep = len(ds.episodes)
    cutoff = total_ep - N_HELDOUT_EP
    region = [i for i, (ep_idx, f_idx) in enumerate(ds.samples) if ep_idx >= cutoff]
    stride = max(1, len(region) // n_windows)
    chosen = region[::stride][:n_windows]
    meta = {
        "total_episodes": total_ep,
        "heldout_ep_start": cutoff,
        "heldout_ep_end": total_ep - 1,
        "region_n_windows": len(region),
        "stride": stride,
        "n_selected": len(chosen),
        "first_global_idx": chosen[0] if chosen else None,
        "last_global_idx": chosen[-1] if chosen else None,
        "first_sample": list(ds.samples[chosen[0]]) if chosen else None,
        "last_sample": list(ds.samples[chosen[-1]]) if chosen else None,
    }
    return chosen, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-root", default="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/self_built/A_0423_0527")
    ap.add_argument("--domain-id", type=int, default=20)
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="/tmp/xvla_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print(f"[load] from_pretrained({CKPT_INIT}) ...", flush=True)
    model = XVLAPolicy.from_pretrained(CKPT_INIT).to(device)
    print(f"[load] torch.load({args.ckpt}) ...", flush=True)
    sd = torch.load(args.ckpt, map_location="cpu")
    step = sd.get("step", None)
    missing, unexpected = model.load_state_dict(sd["model_state"], strict=True)
    model.eval()
    print(f"[load] loaded model_state (step={step}) in {time.time()-t_start:.1f}s", flush=True)

    chunk_size = model.config.chunk_size
    n_denoise = model.config.num_denoising_steps
    print(f"[cfg] chunk_size={chunk_size} num_denoising_steps={n_denoise}", flush=True)

    print(f"[data] LeRobotEE6DDataset({args.val_root}, domain_id={args.domain_id})", flush=True)
    ds = LeRobotEE6DDataset(args.val_root, domain_id=args.domain_id, task_prompt=PROMPT)
    print(f"[data] total windows in dataset: {len(ds)}", flush=True)
    chosen, sel_meta = select_windows(ds, args.n_windows)
    print(f"[data] selected {len(chosen)} val windows: {json.dumps(sel_meta)}", flush=True)

    collate = build_collate()

    # Accumulate sum of abs error per timestep, averaged over action dims, summed over windows
    eff_h = min(chunk_size, ds.action_chunk)  # GT chunk len = 30
    per_step_abs_sum = np.zeros(eff_h, dtype=np.float64)  # over the eff_h timesteps
    n_seen = 0

    bs = args.batch_size
    n_batches = (len(chosen) + bs - 1) // bs
    pred_chunk_len = None
    for b in range(n_batches):
        idxs = chosen[b * bs:(b + 1) * bs]
        samples = [ds[i] for i in idxs]
        batch = collate(samples)
        gt = batch["action"].clone()  # (B,30,20) cpu
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        torch.manual_seed(GEN_SEED + b)  # identical init noise across the 3 models
        if device.type == "cuda":
            torch.cuda.manual_seed_all(GEN_SEED + b)
        with torch.no_grad():
            pred = model.predict_action_chunk(batch)  # (B, chunk, 20)
        pred = pred.float().cpu()
        if pred_chunk_len is None:
            pred_chunk_len = pred.shape[1]
            print(f"[pred] predicted chunk length = {pred_chunk_len} (GT chunk = {gt.shape[1]})", flush=True)
        h = min(pred.shape[1], gt.shape[1], eff_h)
        # abs err per (B, h, dim) -> mean over dims -> (B,h) -> sum over B
        ae = (pred[:, :h, :] - gt[:, :h, :]).abs().mean(dim=2)  # (B,h)
        per_step_abs_sum[:h] += ae.sum(dim=0).numpy()
        n_seen += pred.shape[0]
        if b % 10 == 0:
            print(f"[pred] batch {b+1}/{n_batches} (windows {n_seen}) elapsed={time.time()-t_start:.0f}s", flush=True)

    per_step_mae = per_step_abs_sum / max(1, n_seen)  # (eff_h,) MAE at each timestep

    results = {}
    for hh in HORIZONS:
        use_h = min(hh, eff_h)
        results[f"MAE@{hh}"] = float(per_step_mae[:use_h].mean())

    out = {
        "ckpt": args.ckpt,
        "step": step,
        "val_root": args.val_root,
        "domain_id": args.domain_id,
        "n_windows_requested": args.n_windows,
        "n_windows_eval": n_seen,
        "batch_size": bs,
        "chunk_size": chunk_size,
        "pred_chunk_len": pred_chunk_len,
        "num_denoising_steps": n_denoise,
        "gen_seed": GEN_SEED,
        "horizons": HORIZONS,
        "window_selection": sel_meta,
        "per_step_mae": per_step_mae.tolist(),
        "results": results,
        "elapsed_s": time.time() - t_start,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n========== EVAL SUMMARY ==========", flush=True)
    print(f"ckpt: {args.ckpt} (step={step})", flush=True)
    print(f"val: {args.val_root} domain={args.domain_id}  windows={n_seen}  denoise_steps={n_denoise}", flush=True)
    for hh in HORIZONS:
        print(f"MAE@{hh}: {results[f'MAE@{hh}']:.4f}", flush=True)
    print(f"json -> {args.out}", flush=True)
    print("==================================", flush=True)


if __name__ == "__main__":
    main()
