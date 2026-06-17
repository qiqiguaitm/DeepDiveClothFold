#!/bin/bash
# AIHC 单节点(8×A100)重算 wam_fold_v3 VAE latent —— 修正相机顺序版。
# 旧 latent 用字母序探测 → top_head(俯视)被压进腕部小槽(2026-06-17 根因)。
# 本次显式 --cameras top_head,hand_left,hand_right(俯视=256x320 主图),写到新目录 vae_latent_v3fix,
# 不动旧 vae_latent(gwp_abs_v4 训练用过)。16 shard(2/GPU)并行,断点续跑。
set -e
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
DATA=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/visrobot01_v3_train
cd "$REPO"; source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export DIFFSYNTH_MODEL_BASE_PATH="$REPO/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1

OUT="$DATA/vae_latent_v3fix"
LOG_DIR="$REPO/runs/visrobot01_v3_fold_1e-4/precompute_v3fix_logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/launcher.stdout") 2>&1
echo "[precompute] start $(date +%F_%T) out=$OUT cameras=top_head,hand_left,hand_right"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

TOTAL=16
for s in $(seq 0 $((TOTAL-1))); do
  g=$((s % 8))
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src python scripts/compute_latents.py \
    --shard "$s" --total "$TOTAL" --data_path "$DATA" \
    --cameras top_head,hand_left,hand_right --out_dir "$OUT" \
    > "$LOG_DIR/shard${s}.log" 2>&1 &
done
wait
echo "[precompute] done $(date +%F_%T); latents=$(ls "$OUT"/*.pt 2>/dev/null | wc -l) (expect ~2351)"
