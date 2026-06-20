#!/usr/bin/env bash
# Train cosmos_policy_pipper_fold_colth: warm-start from Cosmos-Policy-ALOHA-Predict2-2B,
# single node, 8x A100. Outputs -> $RUNS/train_out/<project>/<group>/<name>/checkpoints.
#
#   bash train/train_8gpu.sh                 # full training
#   MAXITER=20 NGPU=2 bash train/train_8gpu.sh   # short smoke
#   DRYRUN=1 bash train/train_8gpu.sh        # build+validate config only (no training)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../setup/env.sh"
cd "$CP_ROOT"

# All checkpoints are local (ModelScope mirror via COSMOS_LOCAL_MODELS) -> no network needed.
# Bypass the flaky proxy entirely and run the venv python directly (NOT `uv run`, which would
# trigger uv sync and try to re-download torch from the proxy).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_API_KEY=local-dummy
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGINAIRE_OUTPUT_ROOT="$RUNS/train_out"
export BASE_DATASETS_DIR="$RUNS/datasets"
export PYTHONPATH="$CP_ROOT:${PYTHONPATH:-}"
PYBIN="$CP_ROOT/.venv/bin/python"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT" "$RUNS/logs"

EXP=${EXP:-cosmos_predict2_2b_480p_pipper_fold_colth}
NGPU=${NGPU:-8}
PORT=$(( ( $$ % 20000 ) + 30000 ))
OVERRIDES=( "experiment=$EXP" "job.wandb_mode=disabled" )
[ -n "${MAXITER:-}" ]  && OVERRIDES+=( "trainer.max_iter=$MAXITER" )
[ -n "${SAVEITER:-}" ] && OVERRIDES+=( "checkpoint.save_iter=$SAVEITER" )
[ -n "${BATCH:-}" ]    && OVERRIDES+=( "dataloader_train.batch_size=$BATCH" )

DRY=()
[ "${DRYRUN:-0}" = "1" ] && DRY=( --dryrun )

echo "=== TRAIN $EXP | ngpu=$NGPU | overrides=${OVERRIDES[*]} | $(date) ==="
set -x
"$PYBIN" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
  --max-restarts=0 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py "${DRY[@]}" -- \
  "${OVERRIDES[@]}"
echo "=== TRAIN_DONE rc=$? $(date) ==="
