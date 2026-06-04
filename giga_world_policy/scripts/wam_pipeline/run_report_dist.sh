#!/usr/bin/env bash
# 分布式 episode 报告(在 b2 上跑):把 metric episodes 分 16 片 —— b2 本地 8 卡(shard 0-7)+
# b1 经 ssh 8 卡(shard 8-15),各 worker 出 shards/shard_<id>.json + 自己 viz ep 的图/视频(共享 PFS);
# 凑齐 16 片后 aggregate 成 report.html。一次性(单 ckpt)。
# 用法: CKPT=runs/..._5x/models/checkpoint_epoch_1_step_22000 OUT=runs/..._5x/report_step22000 \
#       bash scripts/wam_pipeline/run_report_dist.sh
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy; cd "$REPO"; source env.sh >/dev/null 2>&1 || true
CKPT=${CKPT:?set CKPT(含 transformer + transformer_ema 的 ckpt 目录)}
OUT=${OUT:?set OUT}
NUM_SHARDS=16; TIMEOUT=${TIMEOUT:-3600}
MODEL_ID=${MODEL_ID:-$WAN_DIFFUSERS}; STATS=assets_visrobot01/norm_stats_vis.json
VAL=${VAL:-$GWP_DATA/visrobot01_val}; T5=${T5:-$VAL/t5_embedding/episode_000000.pt}
B1="ssh -p 429 -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@120.48.99.93"
S=scripts/wam_pipeline/episode_report.py
COMMON="--transformer_dir $CKPT/transformer --ema_dir $CKPT/transformer_ema --model_id $MODEL_ID --stats_path $STATS --val_root $VAL --t5_pkl $T5 --out_dir $OUT --num_shards $NUM_SHARDS --n_metric_eps 200 --n_viz_eps 20 --n_vid_per_ep 3 --n_ema_eps 8 --max_win_per_ep 6"
mkdir -p "$OUT/shards" "$OUT/episodes" "$OUT/logs"
echo "[report-orch] $CKPT -> $OUT, 16 shards (b2 0-7, b1 8-15)"
for g in 0 1 2 3 4 5 6 7; do
  setsid bash -c "CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python $S --shard_id $g $COMMON" > "$OUT/logs/b2_s$g.log" 2>&1 &
done
$B1 "cd $REPO && source env.sh >/dev/null 2>&1 && for g in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=\$g PYTHONPATH=. nohup python $S --shard_id \$((8+g)) $COMMON > $OUT/logs/b1_s\$g.log 2>&1 & done; sleep 1" > "$OUT/logs/b1_dispatch.log" 2>&1 &
t=0
while :; do
  n=$(ls "$OUT"/shards/shard_*.json 2>/dev/null | wc -l)
  [ "$n" -ge "$NUM_SHARDS" ] && break
  [ "$t" -ge "$TIMEOUT" ] && { echo "[report-orch] TIMEOUT: $n/$NUM_SHARDS shards"; break; }
  sleep 20; t=$((t+20))
done
echo "[report-orch] $(ls $OUT/shards/shard_*.json 2>/dev/null|wc -l)/$NUM_SHARDS shards -> aggregate"
PYTHONPATH=. python $S --aggregate $COMMON
