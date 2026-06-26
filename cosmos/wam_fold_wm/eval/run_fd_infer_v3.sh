#!/usr/bin/env bash
# v3 AC-WM controllability eval on visrobot01_v3_val.
# Usage: bash run_fd_infer_v3.sh --export-dir <HF_export> [--n-episodes 10 --num-steps 8 --guidance 3.0 --out-dir ...]
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv; PY=$VENV/bin/python
export PYTHONPATH="$CF:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
# v3: visrobot val set (camera keys auto-detected = top_head/hand_left/hand_right); v1 delta stats reused.
export FD_VAL_ROOT=${FD_VAL_ROOT:-/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/visrobot01_v3_val}
"$PY" /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/eval/fd_infer.py "$@"
