#!/bin/bash
# ② 统一重抽 visrobot + kairobot 的 Wan VAE latent,使两库布局一致(12×48 visual/ref)且时序窗更长
# (OFFS=13 帧 → T_lat=4 → 支持 K>1 带噪多帧 history),输出到 vae_latent_uni(不覆盖现有 vae_latent)。
# 每库 8 分片(单卡独立进程,无 NCCL);可断点续传(跳过已存在 episode)。
set -e
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
source env.sh >/dev/null 2>&1
export no_proxy='*' NO_PROXY='*'; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export GWP_DATA=../kai0/data/wam_fold_v3
export WAN_DIFFUSERS=../checkpoints/Wan2.2-TI2V-5B-Diffusers
export GWP_OUT_SUBDIR=vae_latent_uni
export GWP_OFFS="0,4,8,12,16,20,24,28,32,36,40,44,48"   # 13 帧窗 → T_lat=4(支持 K≤3 history)
NS=${NS:-8}
LOGD=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/ctrl_wm/logs

run() {  # emb view_keys
  local emb=$1 vk=$2
  echo "===== UNIFY EXTRACT $emb (views=$vk) $(date) ====="
  pids=()
  for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$i GWP_VIEW_KEYS="$vk" \
      python -m scripts.wam_pipeline.compute_latents --emb "$emb" --stride 4 --shard $i --num-shards $NS \
      > "$LOGD/unify_${emb}_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || echo "[warn] shard $p exited nonzero"; done
  echo "[$emb] done $(date)"
}

run visrobot01_v3_train "observation.images.top_head,observation.images.hand_left,observation.images.hand_right"
run kairobot01_v3       "observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist"
# 验证集(eval 用)
run visrobot01_v3_val   "observation.images.top_head,observation.images.hand_left,observation.images.hand_right"
echo "===== UNIFY ALL DONE $(date) ====="
