#!/usr/bin/env bash
# FastWAM v4 per-ckpt eval watcher: poll aihc_5n8g_v4 checkpoints, eval each,
# generate summary.json + report_cmp.html (v3 vs v4 trend chart).
# Usage: setsid nohup bash scripts/eval_watch_v4.sh \
#          > runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v4/eval_watch.log 2>&1 &
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
cd "$REPO"; source .venv/bin/activate
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true

RUN=${RUN:-runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v4}
V3_RUN=runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v3
W="$RUN/checkpoints/weights"
GWP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
NFE=${NFE:-20}; FINAL_STEP=${FINAL_STEP:-999999}; POLL=${POLL:-300}
echo "[watch] start $(date +%F_%T) RUN=$RUN nfe=$NFE"

latest_uneval() {
  local best="" f N
  for f in "$W"/step_*.pt; do
    [ -e "$f" ] || continue
    N=$(basename "$f" | grep -oE "[0-9]+" | sed 's/^0*//'); [ -z "$N" ] && N=0
    [ -f "$RUN/report_step${N}/summary.json" ] && continue
    [ -f "$RUN/report_step${N}/.failed" ] && continue
    [ -z "$best" ] || [ "$N" -gt "$best" ] && best=$N
  done
  echo "$best"
}

eval_step() {
  local N=$1 MAX_RETRY=3
  local CK; CK=$(printf "%s/step_%06d.pt" "$W" "$N")
  local OUT="$RUN/report_step${N}"; mkdir -p "$OUT/shards" "$OUT/logs"
  local attempt g
  for attempt in $(seq 1 $MAX_RETRY); do
    [ "$attempt" -gt 1 ] && echo "[retry $attempt/$MAX_RETRY] step $N"
    echo "[watch] $(date +%T) eval step $N attempt $attempt (8-shard) -> $OUT"
    rm -f "$OUT/shards/"*.json "$OUT/summary.json"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/eval_offline_fold.py \
        --shard_id $g --num_shards 8 --weights "$CK" --out_dir "$OUT" --nfe "$NFE" \
        > "$OUT/logs/s${g}_try${attempt}.log" 2>&1 &
    done
    wait
    PYTHONPATH=src python scripts/eval_offline_fold.py --aggregate --num_shards 8 --out_dir "$OUT" \
      > "$OUT/logs/aggregate_try${attempt}.log" 2>&1
    if [ -f "$OUT/summary.json" ]; then
      echo "[watch] step $N DONE (attempt $attempt): $(grep -oE 'raw mae@.*' "$OUT/logs/aggregate_try${attempt}.log" | tail -1)"
      echo "[ref ] gwp_ans .0063/.0288/.0574/.0918@283ms | gwp_ori .0053/.0298/.0595/.0916@532ms | delta .1128@48 | pi05 .1155@48"
      # generate trend report.html (v3 vs v4 comparison)
      ( cd "$GWP" && source env.sh >/dev/null 2>&1 || true
        PYTHONPATH=. python -m scripts.wam_pipeline.cmp_report \
          --baseline_run "$REPO/$V3_RUN" \
          --target_run "$REPO/$RUN" \
          --label_baseline "fastwam-v3 (V1, run3)" --label_target "fastwam-v4 (V1, run4)" \
          --title "FastWAM V1-fold:v3 vs v4 复现对比(visrobot01_val)" \
          --desc "同配方同 V1 数据的两次迭代(复现性)。↓ 越小越好。" \
          --out "$REPO/$RUN/report.html" \
        && echo "[watch] report.html updated at $REPO/$RUN/report.html" \
        || echo "[watch] WARNING: report.html generation failed (non-fatal)" >&2
      )
      return 0
    fi
    echo "[watch] step $N attempt $attempt FAILED"
    tail -3 "$OUT/logs/s0_try${attempt}.log" | sed 's/^/[s0] /'
    sleep 5
  done
  echo "[watch] step $N FAILED after $MAX_RETRY attempts; 标记跳过"
  touch "$OUT/.failed"
}

mkdir -p "$RUN"
while :; do
  N=$(latest_uneval)
  if [ -n "$N" ]; then
    eval_step "$N"
    [ "$N" -ge "$FINAL_STEP" ] && { echo "[watch] reached final $FINAL_STEP, exit"; break; }
  else
    sleep "$POLL"
  fi
done
echo "[watch] end $(date +%F_%T)"
