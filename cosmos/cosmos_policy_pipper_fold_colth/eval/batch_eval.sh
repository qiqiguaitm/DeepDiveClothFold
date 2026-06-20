#!/usr/bin/env bash
# Comprehensive eval: run offline_eval (mae@1/12/25) on EVERY checkpoint (local + cluster) over
# the val set, parallelized across the given GPUs, then print one comparison table.
#   GPUS="3 4 5 6 7" N_EPISODES=30 bash eval/batch_eval.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../setup/env.sh"
cd "$CP_ROOT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export IMAGINAIRE_OUTPUT_ROOT="$RUNS/train_out" BASE_DATASETS_DIR="$RUNS/datasets" PYTHONPATH="$CP_ROOT"
PYBIN="$CP_ROOT/.venv/bin/python"
N_EPISODES="${N_EPISODES:-30}"; STRIDE="${STRIDE:-50}"
read -ra GPUS <<< "${GPUS:-3 4 5 6 7}"
CMP="$RUNS/reports/cmp"; mkdir -p "$CMP"
VAL="$PIPPER_DATA/fold_cloth/val"; STATS="$PIPPER_DATA/dataset_statistics.json"; T5="$PIPPER_DATA/t5_embeddings.pkl"

LOCAL_C="$RUNS/train_out/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_pipper_fold_colth/checkpoints"
CL_C="$RUNS/train_out_aihc/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_pipper_fold_colth/checkpoints"

# collect (tag, ckptdir) pairs
declare -a TAGS DIRS
for d in "$LOCAL_C"/iter_* ; do [ -d "$d/model" ] && { TAGS+=("local_$(basename "$d"|sed 's/iter_0*//')"); DIRS+=("$d"); }; done
for d in "$CL_C"/iter_*    ; do [ -d "$d/model" ] && { TAGS+=("cluster_$(basename "$d"|sed 's/iter_0*//')"); DIRS+=("$d"); }; done
echo "[batch] ${#DIRS[@]} checkpoints on GPUs ${GPUS[*]}, $N_EPISODES episodes each"

i=0
for idx in "${!DIRS[@]}"; do
  d="${DIRS[$idx]}"; tag="${TAGS[$idx]}"; gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  out="$CMP/$tag.json"
  [ -f "$out" ] && { echo "[batch] skip $tag (done)"; i=$((i+1)); continue; }
  ( [ -f "$d/model.pt" ] || CUDA_VISIBLE_DEVICES="" "$PYBIN" "$PROJ/eval/dcp_to_pt.py" "$d" >/dev/null 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" "$PROJ/eval/offline_eval.py" --ckpt "$d/model.pt" \
      --val_dir "$VAL" --stats "$STATS" --t5 "$T5" --n_episodes "$N_EPISODES" --stride "$STRIDE" \
      --out "$out" >"$CMP/$tag.log" 2>&1 && echo "[batch] done $tag" ) &
  i=$((i+1))
  [ $((i % ${#GPUS[@]})) -eq 0 ] && wait
done
wait

echo "=== COMPREHENSIVE val MAE (rad) — mae@1 / mae@12 / mae@25 ==="
"$PYBIN" - "$CMP" << 'PY'
import sys, glob, json, os
cmp = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(cmp, "*.json")):
    r = json.load(open(f)); tag = os.path.basename(f)[:-5]
    run, it = tag.split("_"); rows.append((run, int(it), r))
rows.sort(key=lambda x: (x[0], x[1]))
print(f"{'run':8} {'iter':>5} {'mae@1':>7} {'mae@12':>7} {'mae@25':>7} {'overall':>7}")
for run, it, r in rows:
    print(f"{run:8} {it:>5} {r.get('mae@1',0):7.4f} {r.get('mae@12',0):7.4f} {r.get('mae@25',0):7.4f} {r.get('mae_overall',0):7.4f}")
best = min(rows, key=lambda x: x[2].get('mae@12', 9))
print(f"\nBEST by mae@12: {best[0]}_iter_{best[1]:06d}  mae@12={best[2]['mae@12']:.4f}")
PY
