#!/usr/bin/env bash
# 16 进程(每 GPU 2)并行预计算 wam_fold_v3/visrobot01_v3_train VAE latent 缓存。
# 断点续跑(已存在 ep 跳过)。
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
DATA=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/visrobot01_v3_train
cd "$REPO"; source .venv/bin/activate
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
TOTAL=16
for s in $(seq 0 $((TOTAL-1))); do
  g=$((s % 8))
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/compute_latents.py \
    --shard $s --total $TOTAL --data_path "$DATA" \
    > .precompute_v3_shard${s}.log 2>&1 &
done
wait
echo "[launcher] all shards finished $(date +%F_%T)"
