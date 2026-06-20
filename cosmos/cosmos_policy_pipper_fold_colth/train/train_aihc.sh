#!/bin/bash
# Per-pod AIHC launcher for cosmos_policy_pipper_fold_colth — multi-node torchrun.
# Topology: NNODES pods x 8xA100 (default 5 nodes = 40 ranks). AIHC PyTorchJob injects
# WORLD_SIZE(=#nodes), RANK(=node rank), MASTER_ADDR, MASTER_PORT into each pod.
# Submit: python scripts ... submit_raw.py train/aijob_pipper_fold_5n8g.json
set -uo pipefail
PROJ=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/cosmos_policy_pipper_fold_colth
source "$PROJ/setup/env.sh"          # venv, COSMOS_LOCAL_MODELS, LD_LIBRARY_PATH(cuda libs), etc.
cd "$CP_ROOT"

# ---- AIHC-injected topology ----
export NUM_GPUS=${NUM_GPUS:-8}
export NNODES=${NNODES:-${WORLD_SIZE:-5}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

# ---- pods have no external net -> offline, drop dead proxies ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export WANDB_MODE=disabled WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$CP_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
# cluster output dir (kept separate from the single-node run so they don't clobber)
export IMAGINAIRE_OUTPUT_ROOT="$RUNS/train_out_aihc"
export BASE_DATASETS_DIR="$RUNS/datasets"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT" "$RUNS/logs"

# ---- triton runtime kernel compile needs python3.10 dev headers (pod image lacks them) ----
HDR="$PROJ/setup/py310_headers"
if [ -d "$HDR/python3.10" ] && [ ! -f /usr/include/python3.10/Python.h ]; then
    mkdir -p /usr/include/python3.10 /usr/include/x86_64-linux-gnu/python3.10
    cp -an "$HDR/python3.10/." /usr/include/python3.10/ 2>/dev/null || true
    cp -f "$HDR/arch_pyconfig.h" /usr/include/x86_64-linux-gnu/python3.10/pyconfig.h 2>/dev/null || true
fi

# ---- RDMA/IB: stage jammy rdma-core userland from PFS (cosmos image lacks libibverbs) ----
IBROOT=/mnt/pfs/p46h4f/cosmos/dreamzero/ibverbs/root
if [ -d "$IBROOT" ] && ! ldconfig -p 2>/dev/null | grep -qi libibverbs; then
    cp -an "$IBROOT/usr/lib/x86_64-linux-gnu/." /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    cp -an "$IBROOT/lib/x86_64-linux-gnu/."     /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    mkdir -p /etc/libibverbs.d && cp -an "$IBROOT/etc/libibverbs.d/." /etc/libibverbs.d/ 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi
export LD_LIBRARY_PATH="$IBROOT/usr/lib/x86_64-linux-gnu:$IBROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
HAS_IB=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')
if [ -n "$HAS_IB" ] && ldconfig -p 2>/dev/null | grep -qi libibverbs; then
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
else
    export NCCL_IB_DISABLE=1
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

NPROC_TOTAL=$((NNODES * NUM_GPUS))
EXP=${EXP:-cosmos_predict2_2b_480p_pipper_fold_colth}
MAXITER=${MAXITER:-6000}; SAVEITER=${SAVEITER:-500}; BATCH=${BATCH:-16}
LOG_DIR="$IMAGINAIRE_OUTPUT_ROOT/aihc_logs_$EXP"; mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pod_${NODE_RANK}.stdout") 2>&1

echo "[aihc] node $NODE_RANK/$NNODES gpus/node=$NUM_GPUS total=$NPROC_TOTAL master=$MASTER_ADDR:$MASTER_PORT exp=$EXP batch=$BATCH"
echo "[aihc] IB=$([ "${NCCL_IB_DISABLE:-1}" = 0 ] && echo on || echo off) ibdev='$HAS_IB'"
"$CP_ROOT/.venv/bin/python" -c 'import torch;print("[aihc] torch",torch.__version__,"cuda",torch.cuda.is_available(),"gpus",torch.cuda.device_count())' || true

exec "$CP_ROOT/.venv/bin/python" -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NUM_GPUS" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --max-restarts=0 \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
  experiment="$EXP" job.wandb_mode=disabled \
  trainer.max_iter="$MAXITER" checkpoint.save_iter="$SAVEITER" \
  dataloader_train.batch_size="$BATCH"
