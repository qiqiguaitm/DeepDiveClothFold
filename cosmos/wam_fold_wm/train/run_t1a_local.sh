#!/bin/bash
# T1a A/B run on the LOCAL 8xA100 box (preemption-free). ~3k steps with checkpoints
# every 500 so L0 (IDM-MAE + motion-ΔPSNR) can score iter_2000/2500/3000 vs the M1
# control. L1 levers: ~10hz strided + random skip + chunk16 + low-σ (config t1a).
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
export CKPT_DIR=$RUNS/train_out_t1a_local
export IMAGINAIRE_OUTPUT_ROOT="$CKPT_DIR"
export WANDB_MODE=offline WANDB_DIR="$CKPT_DIR/wandb" WANDB__SERVICE_WAIT=300
mkdir -p "$CKPT_DIR/wandb" "$WAM_WM_LATENT_CACHE"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN
export MALLOC_ARENA_MAX=2

STEPS=${STEPS:-3000}
SAVE=${SAVE:-250}      # frequent ckpts → a CPU-OOM restart loses <=250 steps
WORKERS=${WORKERS:-1}  # minimize dataloader RAM leak on the shared local box
TOML=$COS/wam_fold_wm/train/recipe_wm_nano_t1a.toml
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=0
CKDIR="$CKPT_DIR/cosmos3/action/wam_fold_wm_nano_t1a/checkpoints"
# Self-healing loop: the dataloader leaks ~0.9G/step and OOM-kills a worker after a while;
# torchrun then dies with no auto-restart. Re-launch on failure — the framework auto-resumes
# from the latest checkpoint in CKPT_DIR, and a fresh process resets the leaked RAM.
ATTEMPT=0
while [ "$ATTEMPT" -lt 40 ]; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[t1a-local] attempt $ATTEMPT $(date) steps=$STEPS save=$SAVE workers=$WORKERS (resumes from latest ckpt)"
  "$VENV/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29512 \
    -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
    trainer.max_iter=$STEPS checkpoint.save_iter=$SAVE \
    scheduler.cycle_lengths=[$STEPS] \
    dataloader_train.num_workers=$WORKERS \
    dataloader_train.prefetch_factor=1 \
    dataloader_train.pool_size=4 \
    model.config.parallelism.data_parallel_replicate_degree=1 \
    2>&1 | tee -a "$CKPT_DIR/train.log"
  if ls -d "$CKDIR"/iter_0000030* >/dev/null 2>&1; then
    echo "[t1a-local] reached iter_3000 — DONE $(date)"; break
  fi
  echo "[t1a-local] attempt $ATTEMPT exited (likely CPU-OOM); retrying in 15s..."; sleep 15
done
