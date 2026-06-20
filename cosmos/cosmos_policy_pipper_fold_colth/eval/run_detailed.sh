#!/usr/bin/env bash
# Detailed full-val-set eval for one checkpoint .pt.  Usage: bash eval/run_detailed.sh <ckpt.pt> <out.json>
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../setup/env.sh"
cd "$CP_ROOT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export IMAGINAIRE_OUTPUT_ROOT="$RUNS/train_out_aihc" BASE_DATASETS_DIR="$RUNS/datasets"
export PYTHONPATH="$CP_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CKPT="$1"; OUT="$2"
"$CP_ROOT/.venv/bin/python" "$PROJ/eval/detailed_eval.py" \
  --ckpt "$CKPT" \
  --val_dir "$PIPPER_DATA/fold_cloth/val" \
  --stats "$PIPPER_DATA/dataset_statistics.json" \
  --t5 "$PIPPER_DATA/t5_embeddings.pkl" \
  --n_episodes "${N_EPISODES:-100}" --stride "${STRIDE:-50}" --max_q "${MAXQ:-12}" \
  --out "$OUT"
