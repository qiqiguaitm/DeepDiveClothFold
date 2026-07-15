#!/usr/bin/env bash
# 训练完成后本机 LIBERO 评测双臂 (Arm M milestone vs Arm B baseline)。
# 用法: bash eval_dualarm_libero.sh <M_ckpt.pt> <B_ckpt.pt> [SUITES] [NTRIALS]
# 关键: M 臂 server 需 LMWM_CKPT 重建 LMWMDecoder 架构再加载训练权重(推理不需 MILESTONE_TARGET);
#       B 臂无 LMWM env(released decoder)。server=lawam env, client=libero env。
set -u
L=/home/tim/workspace/deepdive_kai0/lmvla
LAWAM=$L/lawam
LMWM=$L/lmwm
BENCH=$LAWAM/examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh
M_CKPT="${1:?M checkpoint path}"
B_CKPT="${2:?B checkpoint path}"
SUITES_="${3:-libero_10}"
NTRIALS="${4:-50}"

export STAR_VLA_PYTHON=/home/tim/miniconda3/envs/lawam/bin/python   # server: starVLA/torch
export LIBERO_PYTHON=/home/tim/miniconda3/envs/libero/bin/python    # client: libero sim
export SUITES="$SUITES_"
export NUM_TRIALS_PER_TASK="$NTRIALS"
cd "$LAWAM"

echo "=== [Arm M milestone] GPU0, LMWM 架构 ==="
CUDA_VISIBLE_DEVICES=0 \
  LMWM_CKPT=$LMWM/checkpoints/lmwm_libero_dinov3base/lmwm.pt \
  LMWM_ADAPTER_DIR=$LAWAM LMWM_SWAP_TEACHER=1 \
  LIBERO_CKPT_ALIAS=milestone_pathA LIBERO_RUN_GROUP=lmwm_cmp \
  bash "$BENCH" "$M_CKPT" > $LMWM/eval_armM.log 2>&1 &
MPID=$!

echo "=== [Arm B baseline] GPU1, released decoder ==="
CUDA_VISIBLE_DEVICES=1 \
  LIBERO_CKPT_ALIAS=baseline_noswap LIBERO_RUN_GROUP=lmwm_cmp \
  bash "$BENCH" "$B_CKPT" > $LMWM/eval_armB.log 2>&1 &
BPID=$!

wait $MPID; echo "Arm M eval done"
wait $BPID; echo "Arm B eval done"
echo "=== 日志: $LMWM/eval_arm{M,B}.log ==="
