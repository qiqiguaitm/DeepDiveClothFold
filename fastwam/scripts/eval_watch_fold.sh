#!/usr/bin/env bash
# 本机 per-ckpt eval watcher(latest-only):盯 v3 训练的 weights/step_*.pt,新 ckpt →
# 8 卡 eval_offline_fold.py(200-ep 同协议)→ aggregate → 打印曲线行(带 gwp/delta/pi05 参照)。
# 失败的 step 落 .failed 标记后跳过(不死循环);评完 final step 退出。
# 用法:setsid nohup bash scripts/eval_watch_fold.sh > runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v3/eval_watch.log 2>&1 &
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
cd "$REPO"; source .venv/bin/activate
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true

RUN=${RUN:-runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v3}
W="$RUN/checkpoints/weights"
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

eval_step() {  # <N>
  local N=$1
  local CK; CK=$(printf "%s/step_%06d.pt" "$W" "$N")
  local OUT="$RUN/report_step${N}"; mkdir -p "$OUT/shards" "$OUT/logs"
  echo "[watch] $(date +%T) eval step $N (8-shard) -> $OUT"
  local g
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/eval_offline_fold.py \
      --shard_id $g --num_shards 8 --weights "$CK" --out_dir "$OUT" --nfe "$NFE" \
      > "$OUT/logs/s$g.log" 2>&1 &
  done
  wait
  PYTHONPATH=src python scripts/eval_offline_fold.py --aggregate --num_shards 8 --out_dir "$OUT" \
    > "$OUT/logs/aggregate.log" 2>&1
  if [ -f "$OUT/summary.json" ]; then
    echo "[watch] step $N DONE: $(grep -oE 'raw mae@.*' "$OUT/logs/aggregate.log" | tail -1)"
    echo "[ref ] gwp_ans .0063/.0288/.0574/.0918@283ms | gwp_ori .0053/.0298/.0595/.0916@532ms | delta .1128@48 | pi05 .1155@48"
  else
    echo "[watch] step $N FAILED(详见 $OUT/logs);标记跳过"
    tail -5 "$OUT/logs/s0.log" | sed 's/^/[s0] /'
    touch "$OUT/.failed"
  fi
}

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
