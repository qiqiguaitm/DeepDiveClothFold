#!/usr/bin/env bash
# 分布式 eval orchestrator(在 b2 上跑):轮询训练 OUTPUT_DIR 的新 checkpoint,每个 ckpt
# 把 held-out 窗口分 16 片 —— b2 本地 8 卡(shard 0-7)+ b1 经 ssh 8 卡(shard 8-15),
# 各 worker 写 partial 到共享 PFS;凑齐 16 片后聚合成一条记录入 eval_log.jsonl + tensorboard。
#
# coverage(默认 C=exec):A=episode(~6k,~5min)  C=exec(部署执行步长采样)  B=frames(全量,仅终评)
#
# 用法:
#   bash scripts/wam_pipeline/run_eval_dist.sh                 # 默认 C(exec_horizon=8),循环
#   COVERAGE=episode bash .../run_eval_dist.sh                 # A
#   COVERAGE=frames ONCE=1 STEP=50000 bash .../run_eval_dist.sh  # B 终评(只评指定 step)
#   ONCE=1 bash .../run_eval_dist.sh                           # 评完当前未评 ckpt 即退出
set -uo pipefail

REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
cd "$REPO"
source env.sh >/dev/null 2>&1 || true

export OUTPUT_DIR=${OUTPUT_DIR:-runs/visrobot01_fold_aihc_latent}
export MODEL_ID=${MODEL_ID:-../checkpoints/Wan2.2-TI2V-5B-Diffusers}
export STATS_PATH=${STATS_PATH:-assets_visrobot01/norm_stats_vis.json}
export VAL_ROOT=${VAL_ROOT:-../kai0/data/wam_fold_v1/visrobot01_val}
export T5_PKL=${T5_PKL:-../kai0/data/wam_fold_v1/visrobot01_val/t5_embedding/episode_000000.pt}
export COVERAGE=${COVERAGE:-exec}
export EXEC_HORIZON=${EXEC_HORIZON:-16}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}   # 实测:2/GPU 反而更慢(两份全量 diffusion 抢同一 GPU,
                                      # 单窗 2.35→5s)。1/GPU+帧缓存=1.78s/窗最优。保持 1。
NUM_SHARDS=${NUM_SHARDS:-$((16 * SHARDS_PER_GPU))}   # 16 GPU × 每卡片数
POLL=${POLL:-90}
ONCE=${ONCE:-0}
LATEST_ONLY=${LATEST_ONLY:-1}   # 1=每轮只评最新未评 ckpt(跳过积压;88min/ckpt 远超 cadence 时保持最新)
TIMEOUT=${TIMEOUT:-7200}        # 单 ckpt 等齐 16 片的超时(秒;exec_horizon=8 实测 ~88min/片)
B1=${B1:-"ssh -p 429 -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@120.48.99.93"}
W=scripts/wam_pipeline/_eval_worker.sh
PASSENV="OUTPUT_DIR='$OUTPUT_DIR' MODEL_ID='$MODEL_ID' STATS_PATH='$STATS_PATH' VAL_ROOT='$VAL_ROOT' T5_PKL='$T5_PKL' COVERAGE='$COVERAGE' EXEC_HORIZON='$EXEC_HORIZON'"

agg () { python -m scripts.wam_pipeline.eval_watch --aggregate --step "$1" --num_shards "$NUM_SHARDS" \
    --output_dir "$OUTPUT_DIR" --model_id "$MODEL_ID" --stats_path "$STATS_PATH" \
    --val_root "$VAL_ROOT" --t5_pkl "$T5_PKL"; }

run_step () {
  local STEP=$1 SUBDIR=$2
  local sdir="$OUTPUT_DIR/eval_shards/step_$STEP"
  mkdir -p "$sdir/logs"
  echo "[orch $(date +%H:%M:%S)] step $STEP coverage=$COVERAGE: $NUM_SHARDS shards (16 GPU × ${SHARDS_PER_GPU}/GPU) <- $SUBDIR"
  # b2 本地:gpu g 跑 shard {g + r*16}(setsid 脱离进程组 → orchestrator 崩溃不连带杀 worker)
  for g in 0 1 2 3 4 5 6 7; do
    for r in $(seq 0 $((SHARDS_PER_GPU-1))); do
      sid=$((g + r*16))
      setsid bash "$W" "$g" "$sid" "$NUM_SHARDS" "$STEP" "$SUBDIR" > "$sdir/logs/b2_g${g}_s${sid}.log" 2>&1 &
    done
  done
  # b1 远程:gpu g 跑 shard {8+g + r*16};单条 ssh 后台拉起即返回
  $B1 "cd $REPO && $PASSENV SHARDS_PER_GPU=$SHARDS_PER_GPU bash -c 'for g in 0 1 2 3 4 5 6 7; do for r in \$(seq 0 \$((SHARDS_PER_GPU-1))); do sid=\$((8+g + r*16)); nohup bash $W \$g \$sid $NUM_SHARDS $STEP \"$SUBDIR\" > $sdir/logs/b1_g\${g}_s\${sid}.log 2>&1 & done; done; sleep 1'" \
    > "$sdir/logs/b1_dispatch.log" 2>&1 &
  # 等齐 16 片 partial
  local t=0 n=0
  while :; do
    n=$(ls "$sdir"/shard_*.json 2>/dev/null | wc -l)
    [ "$n" -ge "$NUM_SHARDS" ] && break
    if [ "$t" -ge "$TIMEOUT" ]; then echo "[orch] step $STEP TIMEOUT: $n/$NUM_SHARDS shards"; break; fi
    sleep 15; t=$((t+15))
  done
  echo "[orch $(date +%H:%M:%S)] step $STEP: $n/$NUM_SHARDS shards -> aggregate"
  agg "$STEP"
}

echo "[orch] watch $OUTPUT_DIR  coverage=$COVERAGE exec_horizon=$EXEC_HORIZON shards=$NUM_SHARDS poll=${POLL}s once=$ONCE"
while :; do
  if [ -n "${STEP:-}" ]; then
    # 显式指定 step(终评):取该 step 的 transformer 子目录
    SUB=$(python -m scripts.wam_pipeline.eval_watch --list --output_dir "$OUTPUT_DIR" --model_id "$MODEL_ID" \
            --stats_path "$STATS_PATH" --val_root "$VAL_ROOT" --t5_pkl "$T5_PKL" 2>/dev/null | awk -v s="$STEP" '$1==s{print $2}')
    [ -z "$SUB" ] && SUB="$OUTPUT_DIR/models/checkpoint_epoch_1_step_$STEP/transformer_ema"
    run_step "$STEP" "$SUB"; break
  fi
  mapfile -t LINES < <(python -m scripts.wam_pipeline.eval_watch --list --output_dir "$OUTPUT_DIR" --model_id "$MODEL_ID" \
                         --stats_path "$STATS_PATH" --val_root "$VAL_ROOT" --t5_pkl "$T5_PKL" 2>/dev/null | sort -n)
  [ "$LATEST_ONLY" = "1" ] && [ "${#LINES[@]}" -gt 0 ] && LINES=("${LINES[-1]}")  # 只评最新
  for line in "${LINES[@]}"; do
    [ -z "$line" ] && continue
    run_step "${line%%$'\t'*}" "${line#*$'\t'}"
  done
  [ "$ONCE" = "1" ] && break
  sleep "$POLL"
done
