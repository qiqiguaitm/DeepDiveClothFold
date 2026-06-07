#!/bin/bash
# AIHC multi-node launcher for Cosmos3-Nano-Policy -> wam_fold cross-rig full-FT.
# Mirrors giga_world_policy/scripts/aihc/run_train_aihc.sh + tau-0-wm run_train_aihc_tau0.sh:
# AIHC PyTorchJob injects per-pod WORLD_SIZE(#nodes), RANK(node rank), MASTER_ADDR, MASTER_PORT.
# Cosmos uses the framework's native FSDP launcher (torchrun -m cosmos_framework.scripts.train),
# NOT accelerate. The gpfs venv (cu128/torch2.10) is reused; its uv-managed python is staged on
# PFS and populated into each pod's /root/.local (same idea as tau0 staging ibverbs).
set -e
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
CF=$COS/packages/cosmos3
WFP=$COS/wam_fold_policy
VENV=$CF/.venv
RUNS=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs
cd "$CF"

# ---- populate the venv's node-local python on this pod (venv symlinks to /root/.local) ----
UVPY=/root/.local/share/uv/python/cpython-3.13.0-linux-x86_64-gnu
if [ ! -x "$UVPY/bin/python3.13" ]; then
  mkdir -p /root/.local/share/uv/python
  cp -an /mnt/pfs/p46h4f/cosmos/uvpy/cpython-3.13.0-linux-x86_64-gnu /root/.local/share/uv/python/ 2>/dev/null || true
fi

# ---- distributed topology (AIHC-injected) ----
export NUM_GPUS=${NUM_GPUS:-8}
export NNODES=${NNODES:-${WORLD_SIZE:-4}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29504}

# ---- cosmos env (mirror wam_fold_policy/train/env.sh; offline: all caches on PFS) ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export no_proxy='*'
export PYTHONPATH="$CF" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export BASE_CKPT_DCP=$RUNS/checkpoints/Cosmos3-Nano-Policy-DROID-dcp
export CKPT_DIR=${CKPT_DIR:-$RUNS/train_out_4n8g}
export IMAGINAIRE_OUTPUT_ROOT="$CKPT_DIR"; mkdir -p "$CKPT_DIR"

# ---- RDMA/IB (reuse dreamzero userspace rdma-core staged on PFS, like GigaWorld/tau0) ----
IBROOT=/mnt/pfs/p46h4f/cosmos/dreamzero/ibverbs/root
if [ -d "$IBROOT" ] && ! ldconfig -p 2>/dev/null | grep -qi libibverbs; then
  cp -an "$IBROOT/usr/lib/x86_64-linux-gnu/." /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
  cp -an "$IBROOT/lib/x86_64-linux-gnu/."     /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
  mkdir -p /etc/libibverbs.d && cp -an "$IBROOT/etc/libibverbs.d/." /etc/libibverbs.d/ 2>/dev/null || true
  ldconfig 2>/dev/null || true
fi
# ffmpeg (torchcodec video decode) + IB libs on the loader path
export LD_LIBRARY_PATH="/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib:$IBROOT/usr/lib/x86_64-linux-gnu:$IBROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if ls /sys/class/infiniband >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -qi libibverbs; then
  export NCCL_IB_DISABLE=0 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
else
  export NCCL_IB_DISABLE=1
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ---- training args (override via AIHC envs) ----
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_ITER=${SAVE_ITER:-1000}
TOML=$WFP/train/recipe_nano.toml
EXTRA=${EXTRA:-}

echo "[aihc-cosmos] node $NODE_RANK/$NNODES gpus=$NUM_GPUS total=$((NNODES*NUM_GPUS)) master=$MASTER_ADDR:$MASTER_PORT"
echo "[aihc-cosmos] max_iter=$MAX_STEPS save_iter=$SAVE_ITER ckpt_dir=$CKPT_DIR (cross-rig visrobot01+kairobot01)"
"$VENV/bin/python" -c 'import torch;print("[aihc-cosmos] torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.device_count())' || true

exec "$VENV/bin/torchrun" \
  --nnodes="$NNODES" --nproc_per_node="$NUM_GPUS" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter="$MAX_STEPS" checkpoint.save_iter="$SAVE_ITER" $EXTRA
