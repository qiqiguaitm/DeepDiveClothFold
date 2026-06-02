#!/bin/bash
# Exp-1b (A_0522_0526_raw, data_root_cause 对照组) — uc 2-host 16-GPU JAX 启动 (单节点版)。
# 在【每个空闲节点】各跑一次 (从控制端 ssh 进去跑), 先 proc0 (coordinator) 再 proc1。
#
#   用法:  bash run_raw_uc16.sh <COORD_ErH1_IP> <PROC_IDX> [NUM_PROCS=2]
#   例:    # 设空闲节点为 uc03(eth1 .1.4, 当 coordinator/proc0) + uc01(eth1 .1.2, proc1)
#          ssh uc03 'bash .../run_raw_uc16.sh 192.168.1.4 0 2'   # 先起 proc0, 确认监听 :15830
#          ssh uc01 'bash .../run_raw_uc16.sh 192.168.1.4 1 2'   # 再起 proc1
#
# ⚠️ 关键修复 (2026-06-02, 见 uc_cluster_jobs.md §12.11 坑 9):
#   --checkpoint-base-dir 指向【共享 NFS】/data/shared/ubuntu/workspace/multinode_ckpts,
#   绝不能用默认 kai0/checkpoints (uc 上 symlink→节点本地 local_ckpts, 多机第一次 save 必崩)。
set -uo pipefail

COORD_IP="${1:?need coordinator eth1 IP, e.g. 192.168.1.4}"
PROC_IDX="${2:?need process index 0 or 1}"
NUM_PROCS="${3:-2}"

REPO=/data/shared/ubuntu/workspace/deepdive_kai0
CONFIG=pi05_flatten_fold_A_0522_0526_raw
EXP=A_0522_0526_raw_uc16
DATA=/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0522_0526_raw
INIT=/data/shared/ubuntu/workspace/shared_ckpt/Task_A/mixed_1_clean/params
CKPT_BASE=/data/shared/ubuntu/workspace/multinode_ckpts          # 坑9: 共享 NFS, 非节点本地
LOG_DIR=/data/shared/ubuntu/workspace/logs
TS=$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR" "$CKPT_BASE"
LOG="$LOG_DIR/${EXP}_${TS}_proc${PROC_IDX}.log"

cd "$REPO/kai0"
source .venv/bin/activate

# --- JAX 多机协调 ---
export JAX_COORDINATOR_ADDRESS="${COORD_IP}:15830"
export JAX_NUM_PROCESSES="$NUM_PROCS"
export JAX_PROCESS_INDEX="$PROC_IDX"
export JAX_ENABLE_EMPTY_ARRAYS=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
unset XLA_FLAGS

# --- NCCL RDMA + GDR (§12.2) ---
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23 NCCL_IB_RETRY_CNT=7 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1 WANDB_MODE=offline

echo "[proc${PROC_IDX}/${NUM_PROCS}] coord=$JAX_COORDINATOR_ADDRESS ckpt_base=$CKPT_BASE log=$LOG"
nohup .venv/bin/python -u scripts/train.py "$CONFIG" \
  --exp-name "$EXP" \
  --data.repo_id "$DATA" \
  --weight-loader.params-path "$INIT" \
  --checkpoint-base-dir "$CKPT_BASE" \
  --fsdp-devices 16 \
  --overwrite \
  --no-wandb-enabled \
  > "$LOG" 2>&1 &
echo "[proc${PROC_IDX}] pid=$! log=$LOG"
disown
