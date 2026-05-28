#!/bin/bash
# gf0 AWBC dagger-only (ablation): same config as gf0_awbc_baseline_v2 but trained
# on Task_A/dagger_advantage (3457 ep) instead of Task_A/kai0_advantage (3055 ep).
# Aligned to 30K steps to match awbc_v2 A/B series (vanilla, robust).
# Purpose: isolate the base↔dagger contribution to AWBC eval MAE.

CONFIG="pi05_flatten_fold_awbc_daggeronly"
EXP_NAME="gf0_awbc_daggeronly_v2"
BATCH_SIZE=256
FSDP_DEVICES=8

KAI0_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0"
LOG_DIR="/vePFS/tim/workspace/deepdive_kai0/logs"
PYTHON="$KAI0_ROOT/.venv/bin/python3"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

TRAIN_CMD="$PYTHON -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp_name=$EXP_NAME \
  --fsdp-devices $FSDP_DEVICES \
  --batch-size $BATCH_SIZE \
  --no-wandb-enabled"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset JAX_COORDINATOR_ADDRESS JAX_NUM_PROCESSES JAX_PROCESS_INDEX
unset NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
      NCCL_IB_ROUTABLE_FLID_GID_INDEX NCCL_NET_PLUGIN NCCL_IB_TIMEOUT \
      NCCL_IB_RETRY_CNT NCCL_IB_ADDR_FAMILY NCCL_NET_GDR_LEVEL NCCL_ALGO \
      NCCL_IB_PCI_RELAXED_ORDERING
export NCCL_DEBUG=WARN
export KAI0_DATA_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

cd $KAI0_ROOT
nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
echo "[gf0-daggeronly] pid=$! log=$LOG_FILE"
