#!/usr/bin/env bash
# FastWAM v5 per-ckpt eval watcher: poll aihc_5n8g_v5 (V3 data) checkpoints, eval each on
# v3 val (visrobot01_v3_val), generate summary.json + report.html (v4=v1 vs v5=v3 trend chart).
# v5 = v4 recipe (LR/scheduler/batch) on V3 data (wam_fold_v3/visrobot01_v3_train), 50k steps.
# Usage: setsid nohup bash scripts/eval_watch_v5.sh \
#          > runs/visrobot01_v3_fold_1e-4/aihc_5n8g_v5/eval_watch.log 2>&1 &
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
cd "$REPO"; source .venv/bin/activate
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true

# ---- V3 eval overrides for eval_offline_fold.py (default是 v1) ----
export EVAL_VAL_ROOT="$REPO/../kai0/data/wam_fold_v3/visrobot01_v3_val"
export EVAL_VIEW_KEYS="top_head,hand_left,hand_right"   # 角色顺序 [top, left_wrist, right_wrist]
export EVAL_DATA="visrobot01_v3_fold"
export EVAL_TASK="visrobot01_v3_fold_1e-4"
export EVAL_TEXT_EMB="visrobot01_v3_fold"
STATS="$REPO/data/visrobot01_v3_fold/dataset_stats.json"

RUN=${RUN:-runs/visrobot01_v3_fold_1e-4/aihc_5n8g_v5}
W="$RUN/checkpoints/weights"
GWP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
GWP_ABS_V4="${BASELINE_RUN:-$GWP/runs/gwp_abs_v5}"   # report.html 对比基线(默认修正版 gwp_abs_v5)
NFE=${NFE:-20}; FINAL_STEP=${FINAL_STEP:-50000}; POLL=${POLL:-300}
echo "[watch] start $(date +%F_%T) RUN=$RUN nfe=$NFE val=$EVAL_VAL_ROOT stats=$STATS"

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
    echo "[watch] $(date +%T) eval step $N attempt $attempt (8-shard, V3) -> $OUT"
    rm -f "$OUT/shards/"*.json "$OUT/summary.json"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/eval_offline_fold.py \
        --shard_id $g --num_shards 8 --weights "$CK" --out_dir "$OUT" --nfe "$NFE" --stats "$STATS" \
        > "$OUT/logs/s${g}_try${attempt}.log" 2>&1 &
    done
    wait
    PYTHONPATH=src python scripts/eval_offline_fold.py --aggregate --num_shards 8 --out_dir "$OUT" --stats "$STATS" \
      > "$OUT/logs/aggregate_try${attempt}.log" 2>&1
    if [ -f "$OUT/summary.json" ]; then
      echo "[watch] step $N DONE (attempt $attempt): $(grep -oE 'raw mae@.*' "$OUT/logs/aggregate_try${attempt}.log" | tail -1)"
      echo "[ref ] gwp_abs_v4(V3) @48=0.104 | fastwam_v4(V1) @48=0.091 | delta(V1) @48=0.113 | pi05 @48=0.1155"
      # report.html: fastwam-v5(独立ActionDiT@v3, abs系列) vs gwp_abs_v4(共享transformer@v3, delta系列)
      # —— 同 v3 数据 + 同 v3_val 评测,唯一变量=架构。
      ( cd "$GWP" && source env.sh >/dev/null 2>&1 || true
        PYTHONPATH=. python -m scripts.wam_pipeline.cmp_report \
          --baseline_run "$GWP_ABS_V4" \
          --target_run "$REPO/$RUN" \
          --label_baseline "gwp_abs_v4 (共享transformer@v3)" \
          --label_target "fastwam-v5 (独立ActionDiT@v3)" \
          --title "fastwam-v5 vs gwp_abs_v4 @ v3(同数据,唯一变量=架构)" \
          --desc "同 v3 数据(visrobot01_v3_train)+ 同 v3_val 评测;唯一变量=架构(独立 ActionDiT vs 共享 transformer)。↓ 越小越好。" \
          --out "$REPO/$RUN/report.html" \
        && echo "[watch] report.html updated -> $REPO/$RUN/report.html (target=fastwam-v5, baseline=gwp_abs_v4)" \
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
