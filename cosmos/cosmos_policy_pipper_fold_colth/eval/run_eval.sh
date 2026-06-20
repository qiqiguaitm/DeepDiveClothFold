#!/usr/bin/env bash
# Offline action-MAE eval on the val set, using the latest (or given) checkpoint.
#   bash eval/run_eval.sh [CKPT_DIR]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../setup/env.sh"
cd "$CP_ROOT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
export WANDB_MODE=disabled WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export IMAGINAIRE_OUTPUT_ROOT="$RUNS/train_out" BASE_DATASETS_DIR="$RUNS/datasets"
export PYTHONPATH="$CP_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # eval on GPU0 (training uses 1-7)
PYBIN="$CP_ROOT/.venv/bin/python"

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  # newest checkpoints/iter_* dir under the training output
  CKPT=$(ls -d "$IMAGINAIRE_OUTPUT_ROOT"/*/*/cosmos_predict2_2b_480p_pipper_fold_colth/checkpoints/iter_* 2>/dev/null | sort -V | tail -1)
fi
[ -n "$CKPT" ] || { echo "no checkpoint found; pass CKPT_DIR explicitly"; exit 1; }
echo "=== EVAL ckpt=$CKPT ==="
OUT="$RUNS/reports/mae_$(basename "$CKPT").json"; mkdir -p "$RUNS/reports"
"$PYBIN" "$PROJ/eval/offline_eval.py" \
  --ckpt "$CKPT" \
  --val_dir "$PIPPER_DATA/fold_cloth/val" \
  --stats "$PIPPER_DATA/dataset_statistics.json" \
  --t5 "$PIPPER_DATA/t5_embeddings.pkl" \
  --n_episodes "${N_EPISODES:-20}" --stride "${STRIDE:-50}" \
  --out "$OUT"
echo "=== report -> $OUT ==="
