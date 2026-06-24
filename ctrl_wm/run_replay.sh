#!/bin/bash
# Reproduce Ctrl-World inference (1): replay recorded trajectories within the world model.
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/ctrl_wm
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path pretrained/svd_diffusers \
  --clip_model_path pretrained/clip-vit-base-patch32 \
  --ckpt_path pretrained/Ctrl-World/checkpoint-10000.pt
