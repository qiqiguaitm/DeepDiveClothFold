#!/usr/bin/env bash
# b0 单机 per-1k-step eval/report watcher —— 盯 abs 50k 训练的新 ckpt,每个出 report.html + summary.json。
# 设计:latest-only(永远评最新未评 ckpt,跳积压);EMA off → 用 raw transformer(不传 --ema_dir / n_ema=0);
# 把 transformer/ 子目录拷到 eval_ckpts/step_N(~10G)以防训练侧 GC(total_limit=30)在评完前删掉;
# 轻量档(n_metric_eps=60)→ ~5min/ckpt,远快于 ~100min 的 1k-ckpt 落盘节奏。
# mask 由 --stats_path 内嵌(abs=全 False)自动解析。报告 summary.json 的 raw_mae 供 cmp_report 画曲线。
# 用法(b0 上 setsid nohup 后台):RUN=runs/visrobot01_fold_abs_50k bash scripts/wam_pipeline/abs_eval_report_watch.sh
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy; cd "$REPO"; source env.sh >/dev/null 2>&1 || true

RUN=${RUN:-runs/visrobot01_fold_abs_50k}
MODEL_ID=${MODEL_ID:-../checkpoints/Wan2.2-TI2V-5B-Diffusers}
STATS=${STATS:-assets_visrobot01/norm_stats_vis_abs.json}
VAL=${VAL:-../kai0/data/wam_fold_v1/visrobot01_val}
T5=${T5:-$VAL/t5_embedding/episode_000000.pt}
S=scripts/wam_pipeline/episode_report.py
NMETRIC=${NMETRIC:-60}; NVIZ=${NVIZ:-3}; NVID=${NVID:-1}
FINAL_STEP=${FINAL_STEP:-50000}; POLL=${POLL:-120}
COMMON="--model_id $MODEL_ID --stats_path $STATS --val_root $VAL --t5_pkl $T5 \
  --n_metric_eps $NMETRIC --n_viz_eps $NVIZ --n_vid_per_ep $NVID --n_ema_eps 0"
mkdir -p "$RUN/eval_ckpts"
echo "[watch] start $(date +%F_%T) RUN=$RUN stats=$STATS metric_eps=$NMETRIC final=$FINAL_STEP"

latest_uneval () {   # echo newest step N (mult of 1000) whose transformer exists and report.html absent
  local best="" d N
  for d in "$RUN"/models/checkpoint_*_step_*; do
    [ -d "$d/transformer" ] || continue
    N=$(echo "$d" | grep -oE "step_[0-9]+" | grep -oE "[0-9]+")
    [ -z "$N" ] && continue
    [ $((N % 1000)) -ne 0 ] && continue
    [ -f "$RUN/report_step${N}/report.html" ] && continue
    [ -z "$best" ] || [ "$N" -gt "$best" ] && best=$N
  done
  echo "$best"
}

eval_step () {   # <N>
  local N=$1 src OUT g
  src=$(ls -d "$RUN"/models/checkpoint_*_step_${N} 2>/dev/null | head -1)
  [ -z "$src" ] && { echo "[watch] step $N ckpt vanished, skip"; return 1; }
  OUT="$RUN/report_step${N}"; mkdir -p "$OUT/shards" "$OUT/logs"
  # 拷 transformer 出来防 GC(只拷一次)
  local TD="$RUN/eval_ckpts/step_${N}/transformer"
  if [ ! -f "$TD/diffusion_pytorch_model.safetensors" ]; then
    mkdir -p "$TD"; cp -a "$src/transformer/." "$TD/" 2>/dev/null || { echo "[watch] copy failed step $N"; return 1; }
  fi
  echo "[watch] $(date +%T) eval step $N (8-shard b0) -> $OUT"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python "$S" --shard_id $g --num_shards 8 \
      --transformer_dir "$TD" --out_dir "$OUT" $COMMON > "$OUT/logs/s$g.log" 2>&1 &
  done
  wait
  PYTHONPATH=. python "$S" --aggregate --num_shards 8 --transformer_dir "$TD" --out_dir "$OUT" $COMMON \
    > "$OUT/logs/aggregate.log" 2>&1
  if [ -f "$OUT/report.html" ]; then
    echo "[watch] step $N DONE: $(grep -oE 'raw mae@.*' "$OUT/logs/aggregate.log" | tail -1)"
    # 每评完一步就刷新 delta-vs-abs 对比报告(自更新 report_cmp.html)
    PYTHONPATH=. python -m scripts.wam_pipeline.cmp_report \
      --delta_run "${DELTA_RUN:-runs/visrobot01_fold_aihc_latent_5x}" --abs_run "$RUN" \
      --out "${CMP_OUT:-runs/report_cmp.html}" >> "$RUN/eval_watch.log" 2>&1 || true
  else
    echo "[watch] step $N aggregate FAILED (see $OUT/logs/aggregate.log)"
  fi
}

while :; do
  N=$(latest_uneval)
  if [ -n "$N" ]; then
    eval_step "$N" || true
    [ "$N" -ge "$FINAL_STEP" ] && { echo "[watch] reached final step $FINAL_STEP, exit"; break; }
  else
    sleep "$POLL"
  fi
done
echo "[watch] end $(date +%F_%T)"
