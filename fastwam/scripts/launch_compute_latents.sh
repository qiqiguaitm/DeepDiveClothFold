#!/usr/bin/env bash
# 8 卡并行预计算 VAE latent 缓存(nohup bash 本脚本 & 即可存活)。断点续跑(已存在的 ep 跳过)。
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
cd "$REPO"; source .venv/bin/activate
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/compute_latents.py \
    --shard $g --total 8 > .precompute_shard${g}.log 2>&1 &
done
wait
echo "[launcher] all shards finished $(date +%F_%T)"
