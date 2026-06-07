#!/usr/bin/env bash
# Full fine-tune: Cosmos3-Nano-Policy-DROID -> wam_fold, 8-GPU FSDP single node.
# Warm-start from DCP; action head reset (keys_to_skip_loading); rectified-flow.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv
TOML=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/recipe_nano.toml
export PYTHONPATH="$CF:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=600
export UV_PROJECT_ENVIRONMENT=$VENV
export IMAGINAIRE_OUTPUT_ROOT=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/train_out_single
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
export BASE_CKPT_DCP=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/checkpoints/Cosmos3-Nano-Policy-DROID-dcp
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
cd "$CF"
NGPU=${NGPU:-8}; MAXITER=${MAXITER:-5000}; SAVEITER=${SAVEITER:-500}
PORT=$(( ( $$ % 20000 ) + 31000 ))
echo "=== FULL TRAIN start $(date) | ngpu=$NGPU max_iter=$MAXITER save_iter=$SAVEITER ==="
"$VENV/bin/torchrun" --nproc_per_node=$NGPU --master_port=$PORT \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter=$MAXITER checkpoint.save_iter=$SAVEITER
echo "=== FULL_TRAIN_DONE rc=$? $(date) ==="
