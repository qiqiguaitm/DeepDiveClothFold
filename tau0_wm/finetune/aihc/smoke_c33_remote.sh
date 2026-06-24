#!/bin/bash
# 2-GPU chunk=33 (t_lat=9) full-FT smoke on the b1 remote node — validate memory + step time.
set -u
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm
VENV=/mnt/pfs/p46h4f/cosmos/.venv
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TAU0_CHUNK=33 TAU0_LATENT_DIR=vae_latent_c33 TAU0_COND_NOISE=0.05
export CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN
"$VENV/bin/accelerate" launch --use_deepspeed --zero_stage 2 \
  --gradient_accumulation_steps 1 --num_machines 1 --num_processes 2 --mixed_precision bf16 --dynamo_backend no \
  "$REPO/finetune/run_train.py" --phase all --max_steps 6 --lr 1.1e-4 --grad_accum 1 \
  --lambda_v 1.0 --lambda_a 1.0 --warmup_steps 2 --cosine_steps 6 \
  --ckpt_dir "$REPO/runs/_smoke_c33" --ckpt_interval 999999 \
  --init_ckpt "$REPO/checkpoints/tau-0-wm" --resume "$REPO/runs/tau0_fold_p3_32g/final.pt" \
  --no_kai --log_interval 1
echo "C33_SMOKE_TRAIN_DONE rc=$?"
