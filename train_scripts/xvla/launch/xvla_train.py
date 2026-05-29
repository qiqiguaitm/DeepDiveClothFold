"""X-VLA multi-domain training script for Track X (X3.A/X3.B).

Usage (single node 8 GPU):
  cd /data/shared/ubuntu/workspace/xvla_scripts
  torchrun --nproc_per_node=8 xvla_train.py \\
    --config X3B_stage_a   --output_dir /data/shared/ubuntu/local_ckpts/xvla_x3b_stage_a

Multi-node 2-node × 8 GPU:
  on master (uc01): torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
    --master_addr=10.x.x.x --master_port=29500 xvla_train.py --config X3B_stage_a
  on worker (uc02): same with --node_rank=1
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from multi_domain_dataset import LeRobotEE6DDataset, MultiDomainDataset, XVLAHdf5Dataset
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

# ==================== CONFIGS ====================
DATA_ROOT = "/data/shared/ubuntu/workspace/dataset_ee6d"
CKPT_INIT = "/data/shared/ubuntu/workspace/xvla_ckpts"
PROMPT = "Flatten and fold the cloth."

CONFIGS = {
    "X3B_stage_a": dict(
        datasets=[
            dict(root=f"{DATA_ROOT}/Kai0_official/Task_A/base", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{DATA_ROOT}/Kai0_official/Task_A/dagger", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{DATA_ROOT}/Task_A/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=7.0),
        ],
        steps=20_000,
        lr=1e-4,
        warmup_steps=1000,
        freeze_steps=1000,  # backbone frozen for first 1000 steps
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3A_stage_a": dict(
        datasets=[
            dict(root=f"{DATA_ROOT}/Kai0_official/Task_A/base", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{DATA_ROOT}/Kai0_official/Task_A/dagger", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{DATA_ROOT}/Task_A/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=7.0, type="parquet"),
            dict(root="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold",
                 action_cache_dir=f"{DATA_ROOT}/xvla_soft_fold_action_cache",
                 domain_id=21, prompt=PROMPT, weight=2.0, type="hdf5"),
        ],
        steps=20_000,
        lr=1e-4,
        warmup_steps=1000,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3C_vis_only_direct": dict(
        # Ablation: skip Stage A continual pretrain, init from lerobot/xvla-base directly,
        # finetune on vis-only. Measures the actual contribution of Stage A multi-domain pretrain.
        datasets=[
            dict(root=f"{DATA_ROOT}/Task_A/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=20_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,  # longer freeze than Stage B (500): raw base needs more warmup
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3_stage_b": dict(
        datasets=[
            dict(root=f"{DATA_ROOT}/Task_A/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=10_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=500,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "A_0423_0527": dict(
        # Single-dataset direct finetune from xvla-base on A_0423_0527 (cloth-fold, 1085 ep,
        # 1.37M chunk-samples). EE6D built with FIXED converter (interleaved rot6d + binarized
        # gripper). Mirrors X3C_vis_only_direct (direct-from-base). 30k steps ≈ 1.40 epoch at
        # eff batch 64 — A_0423_0527 is ~32% larger than vis_v2_merged, so 30k matches/exceeds
        # X3.C's validated 1.23-epoch exposure. save every 2k → 15 ckpts, pick best by eval.
        datasets=[
            dict(root="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/self_built/A_0423_0527", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
}

# ==================== TRAIN ====================

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def is_main(rank: int): return rank == 0


def build_dataset(cfg):
    datasets = []
    weights = []  # per-dataset weight
    for d in cfg["datasets"]:
        dtype = d.get("type", "parquet")
        if dtype == "hdf5":
            ds = XVLAHdf5Dataset(
                root=d["root"], action_cache_dir=d["action_cache_dir"],
                domain_id=d["domain_id"], task_prompt=d["prompt"],
            )
        else:
            ds = LeRobotEE6DDataset(d["root"], domain_id=d["domain_id"], task_prompt=d["prompt"])
        datasets.append(ds)
        weights.append(d["weight"])
    multi = MultiDomainDataset(datasets)
    # Per-sample weights (normalized within dataset, then weighted by domain weight)
    per_sample_w = []
    for w, ds in zip(weights, datasets):
        per_sample_w.extend([w / len(ds)] * len(ds))
    return multi, torch.tensor(per_sample_w, dtype=torch.float32)


def main(args):
    rank, world, local_rank = setup_distributed()
    cfg = CONFIGS[args.config]
    device = torch.device(f"cuda:{local_rank}")

    if is_main(rank):
        print(f"=== Track X {args.config} ===")
        print(f"world_size={world}, rank={rank}, local_rank={local_rank}")
        cfg_log = {k: v for k, v in cfg.items() if k != "datasets"}
        print(f"config: {json.dumps(cfg_log, indent=2)}")
        for d in cfg["datasets"]:
            print("  domain " + str(d["domain_id"]) + " (" + Path(d["root"]).name + ") weight=" + str(d["weight"]))

    # Data
    multi, sample_weights = build_dataset(cfg)
    if is_main(rank):
        print(f"total samples: {len(multi)}")

    # Sampler: WeightedRandom for single-node; DistributedSampler for multi
    # For balanced multi-domain in DDP, we use WeightedRandomSampler with per-rank slice
    # Simpler: just shuffle + random sampling (assume domain ratio approximated over many batches)
    if world > 1:
        sampler = DistributedSampler(multi, num_replicas=world, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(multi), replacement=True
        )

    # Tokenizer
    tok = AutoTokenizer.from_pretrained("facebook/bart-large")

    fixed_prompt = "Flatten and fold the cloth."
    cached_tokens = tok([fixed_prompt], padding="max_length", max_length=50, truncation=True, return_tensors="pt")["input_ids"][0]
    def collate(batch):
        out = {}
        for k in batch[0].keys():
            if isinstance(batch[0][k], torch.Tensor):
                out[k] = torch.stack([s[k] for s in batch])
        # Add language tokens (same prompt for all samples - cheap broadcast)
        out["observation.language.tokens"] = cached_tokens.unsqueeze(0).expand(len(batch), -1).contiguous()
        return out

    loader = DataLoader(
        multi, batch_size=cfg["batch_size_per_gpu"], sampler=sampler,
        num_workers=args.workers, collate_fn=collate, pin_memory=True, drop_last=True,
    )

    # Model
    if is_main(rank): print(f"loading {CKPT_INIT}...")
    model = XVLAPolicy.from_pretrained(CKPT_INIT).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer: separate VLM LR (lower) and rest
    inner = model.module if world > 1 else model
    vlm_params, other_params = [], []
    for n, p in inner.named_parameters():
        if not p.requires_grad: continue
        if "florence" in n.lower() or "vision" in n.lower():
            vlm_params.append(p)
        else:
            other_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": vlm_params, "lr": cfg["lr"] * cfg["vlm_lr_scale"]},
        {"params": other_params, "lr": cfg["lr"]},
    ], weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], cfg["steps"])

    # Output dir
    if is_main(rank):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        json.dump({**cfg, "world_size": world}, open(f"{args.output_dir}/config.json", "w"), indent=2)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < cfg["steps"]:
        try: batch = next(data_iter)
        except StopIteration:
            if world > 1 and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(step)
            data_iter = iter(loader); batch = next(data_iter)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Freeze backbone for first freeze_steps
        if step == 0 and cfg["freeze_steps"] > 0:
            for p in vlm_params:
                p.requires_grad = False
            if is_main(rank): print("VLM frozen (first " + str(cfg["freeze_steps"]) + " steps)")
        if step == cfg["freeze_steps"] and cfg["freeze_steps"] > 0:
            for p in vlm_params:
                p.requires_grad = True
            if is_main(rank): print(f"VLM UNFROZEN at step {step}")

        loss, log_dict = (model.module.forward(batch) if world > 1 else model.forward(batch))
        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 and is_main(rank):
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta_h = (cfg["steps"] - step) / rate / 3600
            n_steps = cfg["steps"]
            print(f"step {step}/{n_steps} loss={loss.item():.4f} gnorm={gnorm.item():.3f} rate={rate:.2f}it/s eta={eta_h:.1f}h")

        if step % 2000 == 0 and step > 0 and is_main(rank):
            ckpt_path = f"{args.output_dir}/step_{step:06d}"
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            # Use torch.save to bypass draccus serialization bug
            torch.save({"model_state": inner.state_dict(), "step": step}, f"{ckpt_path}/state_dict.pt")
            print(f"saved {ckpt_path}")
        step += 1

    if is_main(rank):
        ckpt_path = f"{args.output_dir}/step_final"
        Path(ckpt_path).mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": inner.state_dict(), "step": step}, f"{ckpt_path}/state_dict.pt")
        print(f"=== DONE: {ckpt_path} ===")

    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(args)
