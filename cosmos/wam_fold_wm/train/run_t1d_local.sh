#!/bin/bash
# T1d = T1a (L1: ~10hz strided + chunk16 + low-σ) + L3 (EVAC/Ctrl-World-style channel-concat
# action conditioning via WAM_COND_CONCAT=14). LOCAL 8xA100, ~3k steps, save every 250.
# Tests whether a dedicated zero-init action-concat pathway lifts ΔPSNR(GT-wrong) above the
# +0.16 L1 plateau toward Ctrl-World's +8.17. Reuses t1a latent cache (latents unchanged;
# cond_tokens are built fresh from action at pack time, not cached).
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
export WAM_WM_LATENT_CACHE=$RUNS/latent_cache_t1a
export WAM_COND_CONCAT=${WAM_COND_CONCAT:-14}     # L3: 14-dim action channel-concat conditioning
export CKPT_DIR=$RUNS/train_out_t1d_local
export IMAGINAIRE_OUTPUT_ROOT="$CKPT_DIR"
export WANDB_MODE=offline WANDB_DIR="$CKPT_DIR/wandb" WANDB__SERVICE_WAIT=300
mkdir -p "$CKPT_DIR/wandb" "$WAM_WM_LATENT_CACHE"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=0

STEPS=${STEPS:-3000}
SAVE=${SAVE:-250}
WORKERS=${WORKERS:-1}
TOML=$COS/wam_fold_wm/train/recipe_wm_nano_t1a.toml
CKDIR="$CKPT_DIR/cosmos3/action/wam_fold_wm_nano_t1a/checkpoints"
ATTEMPT=0
while [ "$ATTEMPT" -lt 40 ]; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[t1d-local] attempt $ATTEMPT $(date) COND_CONCAT=$WAM_COND_CONCAT steps=$STEPS save=$SAVE (resumes from latest ckpt)"
  "$VENV/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29515 \
    -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
    trainer.max_iter=$STEPS checkpoint.save_iter=$SAVE \
    scheduler.cycle_lengths=[$STEPS] \
    dataloader_train.num_workers=$WORKERS \
    dataloader_train.prefetch_factor=1 \
    dataloader_train.pool_size=4 \
    model.config.parallelism.data_parallel_replicate_degree=1 \
    2>&1 | tee -a "$CKPT_DIR/train.log"
  if ls -d "$CKDIR"/iter_0000030* >/dev/null 2>&1; then
    echo "[t1d-local] reached iter_3000 — DONE $(date)"; break
  fi
  echo "[t1d-local] attempt $ATTEMPT exited (likely CPU-OOM); retrying in 15s..."; sleep 15
done
