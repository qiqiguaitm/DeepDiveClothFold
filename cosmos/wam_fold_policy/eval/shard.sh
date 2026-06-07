#!/usr/bin/env bash
# One eval shard pinned to one GPU. Args: <CUDA_DEV> <SHARD_ID> <NUM_SHARDS> [extra args...]
set -uo pipefail
DEV="$1"; SID="$2"; NUM="$3"; shift 3
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
export CUDA_VISIBLE_DEVICES="$DEV"
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
export PYTHONPATH="$CF"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=600
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$CF"
exec "$CF/.venv/bin/python" /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/eval/eval_report.py \
  --shard_id "$SID" --num_shards "$NUM" "$@"
