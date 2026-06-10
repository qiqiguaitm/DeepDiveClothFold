#!/usr/bin/env bash
# 自主完成 delta vs abs 的 1k-step ckpt 评测:等 delta 8 片 -> 聚合 -> 在 b0 跑 abs 8 片 -> 聚合 -> 汇总。
# delta 评测已在外部启动(b0 8卡)。abs 评测因 b1 不可达,改在 b0(delta 评完释放后)跑。
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
cd "$REPO"; source env.sh >/dev/null 2>&1 || true
W=scripts/wam_pipeline/_eval_worker.sh
MODEL_ID=../checkpoints/Wan2.2-TI2V-5B-Diffusers
VAL_ROOT=../kai0/data/wam_fold_v1/visrobot01_val
T5_PKL=../kai0/data/wam_fold_v1/visrobot01_val/t5_embedding/episode_000000.pt
STEP=1000; NS=8; TIMEOUT=2400

wait_shards () {  # <output_dir>
  local OUT=$1 sdir t=0 n=0
  sdir="$OUT/eval_shards/step_$STEP"
  while :; do
    n=$(ls "$sdir"/shard_*.json 2>/dev/null | wc -l)
    [ "$n" -ge "$NS" ] && { echo "[orch] $OUT: $n/$NS shards ready"; return 0; }
    # 死亡检测:无 worker 在跑且不足片数
    local alive; alive=$(ps -e -o cmd | grep -c "[e]val_watch.*$OUT")
    if [ "$alive" -eq 0 ] && [ "$t" -gt 60 ]; then
      echo "[orch] $OUT: workers gone with $n/$NS shards — checking logs"; grep -liE "Error|Traceback" "$sdir"/logs/*.log 2>/dev/null | head; return 1
    fi
    [ "$t" -ge "$TIMEOUT" ] && { echo "[orch] $OUT TIMEOUT $n/$NS"; return 1; }
    sleep 20; t=$((t+20))
  done
}

aggregate () {  # <output_dir> <stats>
  python -m scripts.wam_pipeline.eval_watch --aggregate --step "$STEP" --num_shards "$NS" \
    --output_dir "$1" --model_id "$MODEL_ID" --stats_path "$2" --val_root "$VAL_ROOT" --t5_pkl "$T5_PKL"
}

launch_eval () {  # <output_dir> <stats> <ckpt>
  local OUT=$1 STATS=$2 CKPT=$3 g
  mkdir -p "$OUT/eval_shards/step_$STEP/logs"
  for g in 0 1 2 3 4 5 6 7; do
    setsid env OUTPUT_DIR="$OUT" MODEL_ID="$MODEL_ID" STATS_PATH="$STATS" VAL_ROOT="$VAL_ROOT" \
      T5_PKL="$T5_PKL" COVERAGE=exec EXEC_HORIZON=16 \
      bash "$W" "$g" "$g" "$NS" "$STEP" "$CKPT" \
      > "$OUT/eval_shards/step_$STEP/logs/g${g}.log" 2>&1 < /dev/null &
  done
  echo "[orch] launched 8 eval shards for $OUT"
}

echo "===== [orch] WAIT delta shards (already running on b0) ====="
wait_shards runs/cmp_delta_1k && aggregate runs/cmp_delta_1k assets_visrobot01/norm_stats_vis.json && echo "[orch] DELTA aggregated"

echo "===== [orch] RUN abs eval on b0 (b1 unreachable) ====="
launch_eval runs/cmp_abs_1k assets_visrobot01/norm_stats_vis_abs.json runs/cmp_abs_1k/models/checkpoint_epoch_1_step_1000/transformer
sleep 30
wait_shards runs/cmp_abs_1k && aggregate runs/cmp_abs_1k assets_visrobot01/norm_stats_vis_abs.json && echo "[orch] ABS aggregated"

echo "===== [orch] RESULTS ====="
echo "--- DELTA eval_log.jsonl ---"; tail -1 runs/cmp_delta_1k/eval_log.jsonl 2>/dev/null
echo "--- ABS   eval_log.jsonl ---"; tail -1 runs/cmp_abs_1k/eval_log.jsonl 2>/dev/null
echo "[orch] ALL DONE"
