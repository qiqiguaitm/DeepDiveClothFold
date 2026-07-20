#!/bin/bash
# 官方 LIBERO eval (LaWAM released libero SFT ckpt). scoped 首跑先验流水线+SR.
set -x
cd /home/tim/workspace/deepdive_kai0/lmvla/lawam
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export LIBERO_HOME=/home/tim/workspace/deepdive_kai0/lmvla/LIBERO
export LIBERO_PYTHON=/home/tim/miniconda3/envs/libero/bin/python
export STAR_VLA_PYTHON=/home/tim/miniconda3/envs/lawam/bin/python
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com

CKPT_PATH=results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt

SUITES="${SUITES:-libero_10}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-20}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
GPU_IDS="${GPU_IDS:-0 1}" \
OUTPUT_ROOT=results/eval_runs/libero \
LIBERO_CKPT_ALIAS=lawam_libero_sft \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$CKPT_PATH"
echo "===== LIBERO EVAL DONE ====="
