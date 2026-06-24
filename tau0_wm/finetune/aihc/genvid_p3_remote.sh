#!/bin/bash
# Regenerate §5 closed-loop rollout GIFs + video_metrics.json from a P3 checkpoint
# on the b1 remote node (1 GPU). chunk=9 / vae_latent_c9 to match the P3 model.
set -u
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm
VENV=/mnt/pfs/p46h4f/cosmos/.venv
CKPT=${CKPT:-$REPO/runs/tau0_fold_p3_32g/final.pt}
OUT=${OUT:-$REPO/runs/report_assets_p3}
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TAU0_CHUNK=9 TAU0_LATENT_DIR=vae_latent_c9
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"
echo "[genvid-p3] ckpt=$CKPT out=$OUT chunk=9 $(date)"
"$VENV/bin/python" finetune/gen_video_compare.py --ckpt "$CKPT" \
  --n_windows 4 --steps 10 --rollout_k 8 --out_dir "$OUT"
echo "[genvid-p3] DONE $(date)"
