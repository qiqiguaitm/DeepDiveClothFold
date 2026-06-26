#!/bin/bash
# 百度 AIHC 多节点【统一 latent 重抽 vae_latent_uni】:每 pod 8 分片(每 GPU 一个),全集群 NNODES*8 路。
# 三个 cloth-fold 数据集,各按正确相机序;13 帧窗(OFFS)→ T_lat=4(12×48);写共享 PFS,断点续抽。
# 独立分片(无 NCCL/rendezvous)。提交:submit_raw.py aijob_unify_5n8g.json。
set -e
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
source "$REPO/env.sh"; cd "$REPO"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 统一重抽参数
export GWP_DATA=../kai0/data/wam_fold_v3
export WAN_DIFFUSERS=../checkpoints/Wan2.2-TI2V-5B-Diffusers
export GWP_OUT_SUBDIR=vae_latent_uni
export GWP_OFFS="0,4,8,12,16,20,24,28,32,36,40,44,48"   # 13 帧窗 → T_lat=4
VIS="observation.images.top_head,observation.images.hand_left,observation.images.hand_right"
KAI="observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist"

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-${WORLD_SIZE:-5}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
NUM_SHARDS=$((NNODES * NUM_GPUS))
STRIDE=${STRIDE:-4}; WORKERS=${WORKERS:-8}
LOGD="$REPO/.wam_run"; mkdir -p "$LOGD"
echo "[unify] node $NODE_RANK/$NNODES gpus=$NUM_GPUS num_shards=$NUM_SHARDS out=$GWP_OUT_SUBDIR"

run_emb() {  # emb view_keys
  local emb=$1 vk=$2
  echo "[unify] === $emb (views=$vk) node $NODE_RANK ==="
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD=$((NODE_RANK * NUM_GPUS + g))
    CUDA_VISIBLE_DEVICES=$g GWP_VIEW_KEYS="$vk" python -m scripts.wam_pipeline.compute_latents \
      --emb "$emb" --stride "$STRIDE" --shard "$SHARD" --num-shards "$NUM_SHARDS" \
      --workers "$WORKERS" --batch 8 > "$LOGD/unify_${emb}_shard${SHARD}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || echo "[unify][warn] $emb shard exited nonzero"; done
  echo "[unify] $emb done on node $NODE_RANK"
}

run_emb visrobot01_v3_train "$VIS"
run_emb kairobot01_v3       "$KAI"
run_emb visrobot01_v3_val   "$VIS"
echo "UNIFY_NODE_DONE $NODE_RANK"
