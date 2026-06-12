#!/bin/bash
# FastWAM AIHC 多节点 launcher —— visrobot01 叠衣服 abs-angle uncond 训练。
# 拓扑:NNODES pod × 8×A100;AIHC PyTorchJob 注入 WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT。
# 与 GWP run_train_aihc.sh 同模式:PFS venv + 离线 + ibverbs 用户态库注入。
# 注意:fastwam 的 train_zero1.sh 只同步 RUN_ID,不传 accelerate 多机参数 → 本脚本直接调
# accelerate launch 并补 --num_machines/--machine_rank/--main_process_ip/--main_process_port。
set -e
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
cd "$REPO"
source .venv/bin/activate

# ---- AIHC 注入 env → 训练所需 ----
export NUM_GPUS=${NUM_GPUS:-8}
export NNODES=${NNODES:-${WORLD_SIZE:-1}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
NPROC_TOTAL=$((NNODES * NUM_GPUS))

# ---- 离线 + 本地权重 ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export DIFFSYNTH_MODEL_BASE_PATH="$REPO/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$REPO/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"   # torchcodec 的 FFmpeg6(conda-forge,PFS)
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# ---- RDMA/IB(复用 dreamzero 暂存的 jammy rdma-core 用户态库)----
IBROOT=/mnt/pfs/p46h4f/cosmos/dreamzero/ibverbs/root
if [ -d "$IBROOT" ] && ! ldconfig -p 2>/dev/null | grep -qi libibverbs; then
    cp -an "$IBROOT/usr/lib/x86_64-linux-gnu/." /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    cp -an "$IBROOT/lib/x86_64-linux-gnu/."     /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    mkdir -p /etc/libibverbs.d && cp -an "$IBROOT/etc/libibverbs.d/." /etc/libibverbs.d/ 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi
HAS_IB=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')
if [ -n "$HAS_IB" ] && ldconfig -p 2>/dev/null | grep -qi libibverbs; then
    export NCCL_IB_DISABLE=0 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
else
    export NCCL_IB_DISABLE=1
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

TASK=${TASK:-visrobot01_fold_uncond_1e-4}
RUN_NAME=${RUN_NAME:-aihc_5n8g}
OUT="$REPO/runs/${TASK}/${RUN_NAME}"
mkdir -p "$OUT"
echo "[aihc] node $NODE_RANK/$NNODES gpus/node=$NUM_GPUS total=$NPROC_TOTAL master=$MASTER_ADDR:$MASTER_PORT task=$TASK out=$OUT"
python -c 'import torch;print("[aihc] torch",torch.__version__,"cuda",torch.cuda.is_available(),"gpus",torch.cuda.device_count())' || true

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "$NPROC_TOTAL" \
  --num_machines "$NNODES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  scripts/train.py \
  "task=${TASK}" \
  "output_dir=${OUT}" \
  "wandb.name=${TASK}" \
  model.mot_checkpoint_mixed_attn=true \
  num_workers=${NUM_WORKERS:-10} \
  num_epochs=${NUM_EPOCHS:-5} \
  save_every=${SAVE_EVERY:-2500} \
  eval_every=${EVAL_EVERY:-100000} \
  ${EXTRA_OVERRIDES:-}
