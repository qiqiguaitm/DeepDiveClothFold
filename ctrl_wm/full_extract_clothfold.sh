#!/bin/bash
# Full SVD-VAE latent extraction for the cloth-fold datasets.
# Uses 8 INDEPENDENT single-GPU processes with manual sharding (no accelerate/NCCL),
# so uneven per-episode runtime can't trigger a distributed barrier timeout.
# Resumable: extract skips episodes whose annotation already exists.
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/ctrl_wm
export no_proxy='*' NO_PROXY='*'; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false

SRC=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3
OUT=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3_cw
SVD=pretrained/svd_diffusers
VIS=observation.images.top_head,observation.images.hand_left,observation.images.hand_right
KAI=observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist
NS=${NS:-8}

run() {  # name cameras split
  local name=$1 cams=$2 split=$3
  echo "================ EXTRACT $name ($split) $(date) ================"
  pids=()
  for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$i .venv/bin/python dataset_example/extract_latent_agilex.py \
      --src_path $SRC/$name --out_path $OUT/$name --svd_path $SVD --cameras $cams \
      --split $split --rgb_skip 3 --shard $i --num_shards $NS \
      > logs/extract_${name}_shard${i}.log 2>&1 &
    pids+=($!)
  done
  echo "launched shards: ${pids[*]}"
  fail=0
  for pid in "${pids[@]}"; do wait $pid || fail=1; done
  echo "[$name] all shards finished (fail=$fail) $(date)"
}

run visrobot01_v3_val   "$VIS" val
run visrobot01_v3_train "$VIS" train
run kairobot01_v3       "$KAI" train
echo "================ ALL EXTRACTION DONE $(date) ================"
