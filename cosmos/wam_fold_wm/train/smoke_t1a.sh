#!/bin/bash
# M0 smoke for the wam_fold_v3 AC-WM data path (single-node 8xA100, 30 steps).
# Validates: v3 per-rig camera auto-detect + index-decode (start_time fix) + FD token flow.
set -e
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
CF=$COS/packages/cosmos3
VENV=$CF/.venv
RUNS=$COS/wam_fold_wm_runs
cd "$CF"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export no_proxy='*'
export PYTHONPATH="$CF" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib:${LD_LIBRARY_PATH:-}
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export BASE_CKPT_DCP=$RUNS/checkpoints/Cosmos3-Nano-dcp
export WAM_WM_LATENT_CACHE=$RUNS/latent_cache_t1a      # fresh: v3 top_head != v1 cam_high, same cache key
export CKPT_DIR=$RUNS/smoke_t1a
export IMAGINAIRE_OUTPUT_ROOT="$CKPT_DIR"
export WANDB_MODE=offline WANDB_DIR="$CKPT_DIR/wandb" WANDB__SERVICE_WAIT=300
mkdir -p "$CKPT_DIR/wandb" "$WAM_WM_LATENT_CACHE"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN
export MALLOC_ARENA_MAX=2

STEPS=${STEPS:-30}
TOML=$COS/wam_fold_wm/train/recipe_wm_nano_t1a.toml
echo "[smoke-v3] $(date) steps=$STEPS ckpt=$CKPT_DIR latent_cache=$WAM_WM_LATENT_CACHE"
"$VENV/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29511 \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter=$STEPS checkpoint.save_iter=$STEPS \
  scheduler.cycle_lengths=[$STEPS] \
  model.config.parallelism.data_parallel_replicate_degree=1
echo "[smoke-v3] EXIT $? $(date)"
