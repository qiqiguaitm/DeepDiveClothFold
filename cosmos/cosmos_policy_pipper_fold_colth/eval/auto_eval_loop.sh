#!/usr/bin/env bash
# Autonomous eval loop: watch the training checkpoints dir; for each new iter_* checkpoint,
# consolidate DCP->.pt and run offline action-MAE eval on the val set (GPU0, while training
# uses 1-7). Append results to a curve file. Stops when training is done AND all ckpts evaled.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../setup/env.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
PYBIN="$CP_ROOT/.venv/bin/python"
# CKD/CURVE overridable so the same loop can track the local run OR the 5-node cluster run.
CKD="${CKD:-$RUNS/train_out/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_pipper_fold_colth/checkpoints}"
CURVE="${CURVE:-$RUNS/reports/mae_curve.jsonl}"
STOP_WHEN_NO_TRAIN="${STOP_WHEN_NO_TRAIN:-1}"   # 0 = keep looping (cluster trains on other nodes)
mkdir -p "$RUNS/reports"
N_EPISODES="${N_EPISODES:-10}"; STRIDE="${STRIDE:-50}"

evaled() { grep -q "\"iter\": $1" "$CURVE" 2>/dev/null; }

while true; do
  for d in $(ls -d "$CKD"/iter_* 2>/dev/null | sort -V); do
    it=$(basename "$d" | sed 's/iter_0*//')
    [ -z "$it" ] && it=0
    evaled "$it" && continue
    [ -d "$d/model" ] || continue
    echo "[$(date '+%T')] eval iter $it"
    # consolidate DCP -> .pt (skip if exists)
    [ -f "$d/model.pt" ] || CUDA_VISIBLE_DEVICES="" "$PYBIN" "$PROJ/eval/dcp_to_pt.py" "$d" >>"$RUNS/logs/auto_eval.log" 2>&1
    OUT="$RUNS/reports/mae_iter_${it}.json"
    CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 \
      IMAGINAIRE_OUTPUT_ROOT="$RUNS/train_out" BASE_DATASETS_DIR="$RUNS/datasets" \
      PYTHONPATH="$CP_ROOT" \
      "$PYBIN" "$PROJ/eval/offline_eval.py" \
        --ckpt "$d/model.pt" --val_dir "$PIPPER_DATA/fold_cloth/val" \
        --stats "$PIPPER_DATA/dataset_statistics.json" --t5 "$PIPPER_DATA/t5_embeddings.pkl" \
        --n_episodes "$N_EPISODES" --stride "$STRIDE" --out "$OUT" \
        >>"$RUNS/logs/auto_eval.log" 2>&1 \
      && "$PYBIN" -c "import json;r=json.load(open('$OUT'));r['iter']=$it;print(json.dumps(r))" >> "$CURVE" \
      && echo "[$(date '+%T')] iter $it -> MAE $($PYBIN -c "import json;print(round(json.load(open('$OUT'))['mae_overall'],4))")"
  done
  # exit when local training finished and no unevaled ckpts remain (skip for cluster: STOP_WHEN_NO_TRAIN=0)
  if [ "$STOP_WHEN_NO_TRAIN" = "1" ] && ! pgrep -f "cosmos_policy.scripts.train" >/dev/null 2>&1; then
    sleep 60
    pgrep -f "cosmos_policy.scripts.train" >/dev/null 2>&1 || { echo "[$(date '+%T')] training done; auto-eval exiting"; break; }
  fi
  sleep 120
done
echo "=== MAE CURVE ==="; cat "$CURVE" 2>/dev/null
