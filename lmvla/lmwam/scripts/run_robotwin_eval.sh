#!/bin/bash
set -x
cd /home/tim/workspace/deepdive_kai0/lmvla/lawam
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=1
export STAR_VLA_PYTHON=/home/tim/miniconda3/envs/lawam/bin/python
export ROBOTWIN_PATH=/home/tim/workspace/RoboTwin
export ROBOTWIN_PYTHON=/home/tim/workspace/deepdive_kai0/lmvla/lawam/robotwin_python_wrapper.sh
export ROBOTWIN_EVAL_ROOT=results/eval_runs/robotwin
export ROBOTWIN_TASKS="${ROBOTWIN_TASKS:-beat_block_hammer}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-10}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-2}"
CKPT=results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh "$CKPT" demo_clean
echo "===== ROBOTWIN EVAL DONE ====="
