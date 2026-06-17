#!/usr/bin/env bash
# One-shot clean sequential re-eval of fastwam v4 final steps (22500/25000/25510).
# All 8 GPUs per step, no contention. Aggregates + prints MAE per step.
set -uo pipefail
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true
W=runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v4/checkpoints/weights

for N in 22500 25000 25510; do
  CK=$(printf "%s/step_%06d.pt" "$W" "$N")
  [ -f "$CK" ] || { echo "[skip] $CK missing"; continue; }
  OUT="runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v4/report_step${N}"
  rm -f "$OUT/.failed" "$OUT/summary.json" "$OUT/shards/"*.json
  mkdir -p "$OUT/shards" "$OUT/logs"
  echo "[re-eval] $(date +%T) step $N (8 shards, all GPUs)"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/eval_offline_fold.py \
      --shard_id $g --num_shards 8 --weights "$CK" --out_dir "$OUT" --nfe 20 \
      > "$OUT/logs/s${g}_final.log" 2>&1 &
  done
  wait
  PYTHONPATH=src python scripts/eval_offline_fold.py --aggregate --num_shards 8 --out_dir "$OUT" \
    > "$OUT/logs/aggregate_final.log" 2>&1
  if [ -f "$OUT/summary.json" ]; then
    python3 -c "import json;m=json.load(open('$OUT/summary.json'))['raw_mae'];print(f'[done] step $N: @1={m[\"1\"]:.4f} @10={m[\"10\"]:.4f} @24={m[\"24\"]:.4f} @48={m[\"48\"]:.4f}')"
  else
    echo "[FAIL] step $N (see $OUT/logs/aggregate_final.log)"
    tail -3 "$OUT/logs/s0_final.log"
  fi
done
echo "[re-eval] all done $(date +%T)"
