#!/bin/bash
# Fine-tune Ctrl-World on the cloth-fold datasets.
# Train on visrobot01_v3_train + kairobot01_v3, validate on visrobot01_v3_val.
# Override via env vars; defaults are for the FULL run.
set -e
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/ctrl_wm
export no_proxy='*' NO_PROXY='*'; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export WANDB_MODE=${WANDB_MODE:-offline}
export SWANLAB_MODE=${SWANLAB_MODE:-local}
export TOKENIZERS_PARALLELISM=false

OUT_ROOT=${OUT_ROOT:-/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3_cw}
META=${META:-dataset_meta_info_clothfold}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PORT=${PORT:-29533}
TAG=${TAG:-clothfold_v3}
OUTPUT_DIR=${OUTPUT_DIR:-model_ckpt/${TAG}}
CKPT=${CKPT:-pretrained/Ctrl-World/checkpoint-10000.pt}
BS=${BS:-4}
LR=${LR:-1e-5}
MAXSTEPS=${MAXSTEPS:-100000}
CKPTSTEPS=${CKPTSTEPS:-5000}
VALSTEPS=${VALSTEPS:-2500}
VIDEONUM=${VIDEONUM:-4}
PROB=${PROB:-0.265,0.735}

mkdir -p "$OUTPUT_DIR"
echo "=== fine-tune cloth-fold: OUT_ROOT=$OUT_ROOT META=$META GPUS=$GPUS TAG=$TAG ==="
CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/accelerate launch --num_processes $NPROC --num_machines 1 --mixed_precision fp16 --main_process_port $PORT \
  scripts/train_wm.py \
  --svd_model_path pretrained/svd_diffusers \
  --clip_model_path pretrained/clip-vit-base-patch32 \
  --ckpt_path "$CKPT" \
  --dataset_root_path "$OUT_ROOT" \
  --dataset_meta_info_path "$META" \
  --dataset_names visrobot01_v3_train+kairobot01_v3 \
  --dataset_cfgs visrobot01_v3_train+kairobot01_v3 \
  --prob "$PROB" \
  --val_dataset_names visrobot01_v3_val \
  --val_dataset_cfgs visrobot01_v3_val \
  --action_dim 14 \
  --down_sample 3 \
  --tag "$TAG" \
  --output_dir "$OUTPUT_DIR" \
  --learning_rate $LR \
  --train_batch_size $BS \
  --max_train_steps $MAXSTEPS \
  --checkpointing_steps $CKPTSTEPS \
  --validation_steps $VALSTEPS \
  --video_num $VIDEONUM
