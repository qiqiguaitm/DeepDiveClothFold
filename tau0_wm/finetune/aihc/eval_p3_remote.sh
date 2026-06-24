#!/bin/bash
# Run P3 eval (action MSE + video PSNR/SSIM) on the b1 remote node's 8 free GPUs.
# chunk=9 / vae_latent_c9 to match the P3 model. Launched via ssh from the driver.
set -u
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm
VENV=/mnt/pfs/p46h4f/cosmos/.venv
CKPT=${CKPT:-$REPO/runs/tau0_fold_p3_32g/step_10000.pt}
# derive a tag from the checkpoint step (step_10000.pt -> p3_step10000; final.pt -> p3_final)
_base=$(basename "$CKPT" .pt)
if [ "$_base" = "final" ]; then
  TAG="p3_final"
else
  _n=$(echo "$_base" | grep -oE '[0-9]+' | tail -1)   # step_20000 -> 20000
  TAG="p3_step${_n}"
fi
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TAU0_CHUNK=9 TAU0_LATENT_DIR=vae_latent_c9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN
ACC="$VENV/bin/accelerate"
COMMON="--num_machines 1 --num_processes 8 --mixed_precision bf16 --dynamo_backend no --main_process_port 29677"

echo "[eval-p3] ckpt=$CKPT chunk=9 latdir=vae_latent_c9  $(date)"
echo "[eval-p3] === action MSE eval (run_eval_dist) ==="
$ACC launch $COMMON finetune/run_eval_dist.py --ckpt "$CKPT" --tag "$TAG" --out "$REPO/runs/eval_report.json"
echo "[eval-p3] === video PSNR/SSIM eval (eval_gigaworld_dist) ==="
$ACC launch $COMMON finetune/eval_gigaworld_dist.py --ckpt "$CKPT" --tag "$TAG" --out "$REPO/runs/eval_gigaworld.json"
echo "[eval-p3] DONE $(date)"
