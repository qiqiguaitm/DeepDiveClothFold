#!/usr/bin/env bash
# 单个 eval 分片 worker(b2/b1 通用,脚本在共享 PFS 上,b1 经 ssh 跑同一路径)。
# 用法: _eval_worker.sh <local_gpu> <shard_id> <num_shards> <step> <ckpt_subdir>
# 配置走环境变量,缺省=生产路径(ssh 不带环境,故内置默认)。所有路径相对 REPO(脚本内 cd)。
set -uo pipefail
GPU=$1; SHARD=$2; NUM_SHARDS=$3; STEP=$4; CKPT_SUBDIR=$5

REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
cd "$REPO"
source env.sh >/dev/null 2>&1 || true

OUTPUT_DIR=${OUTPUT_DIR:-runs/visrobot01_fold_aihc_latent}
MODEL_ID=${MODEL_ID:-../checkpoints/Wan2.2-TI2V-5B-Diffusers}
STATS_PATH=${STATS_PATH:-assets_visrobot01/norm_stats_vis.json}
VAL_ROOT=${VAL_ROOT:-../kai0/data/wam_fold_v1/visrobot01_val}
T5_PKL=${T5_PKL:-../kai0/data/wam_fold_v1/visrobot01_val/t5_embedding/episode_000000.pt}
COVERAGE=${COVERAGE:-exec}
EXEC_HORIZON=${EXEC_HORIZON:-16}
# 仅 shard 0 存几条并排 mp4,其余 shard 关掉(n_mp4=0 → 不存)
NMP4=3; [ "$SHARD" != "0" ] && NMP4=0

CUDA_VISIBLE_DEVICES=$GPU python -m scripts.wam_pipeline.eval_watch \
  --ckpt_subdir "$CKPT_SUBDIR" --step "$STEP" --shard_id "$SHARD" --num_shards "$NUM_SHARDS" \
  --output_dir "$OUTPUT_DIR" --model_id "$MODEL_ID" --stats_path "$STATS_PATH" \
  --val_root "$VAL_ROOT" --t5_pkl "$T5_PKL" \
  --coverage "$COVERAGE" --exec_horizon "$EXEC_HORIZON" --n_mp4 "$NMP4" --no_lpips
