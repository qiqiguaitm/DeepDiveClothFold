#!/bin/bash
# AIHC multi-node launcher for tau0 joint-space fine-tune (run_train.py).
# Mirrors giga_world_policy/scripts/aihc/run_train_aihc.sh: AIHC PyTorchJob injects
# WORLD_SIZE(#nodes), RANK(node rank), MASTER_ADDR, MASTER_PORT per pod.
# Job command: bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/finetune/aihc/run_train_aihc_tau0.sh
set -e
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm
VENV=/mnt/pfs/p46h4f/cosmos/.venv
cd "$REPO"

export NUM_GPUS=${NUM_GPUS:-8}
export NNODES=${NNODES:-${WORLD_SIZE:-4}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
NPROC_TOTAL=$((NNODES * NUM_GPUS))

# pods have no external net; local PFS weights/data only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$REPO"

# RDMA/IB (reuse dreamzero's userspace rdma-core staged on PFS, like GigaWorld)
IBROOT=/mnt/pfs/p46h4f/cosmos/dreamzero/ibverbs/root
if [ -d "$IBROOT" ] && ! ldconfig -p 2>/dev/null | grep -qi libibverbs; then
    cp -an "$IBROOT/usr/lib/x86_64-linux-gnu/." /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    cp -an "$IBROOT/lib/x86_64-linux-gnu/."     /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    mkdir -p /etc/libibverbs.d && cp -an "$IBROOT/etc/libibverbs.d/." /etc/libibverbs.d/ 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi
export LD_LIBRARY_PATH="$IBROOT/usr/lib/x86_64-linux-gnu:$IBROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
HAS_IB=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')
if [ -n "$HAS_IB" ] && ldconfig -p 2>/dev/null | grep -qi libibverbs; then
    export NCCL_IB_DISABLE=0 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
else
    export NCCL_IB_DISABLE=1
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ---- tau0 training args (override via AIHC envs) ----
PHASE=${PHASE:-p2_specialize}          # P3 = "all" (full FT)
MAX_STEPS=${MAX_STEPS:-20000}
LR=${LR:-3e-5}                         # P3 warmup-cosine peak ~2**-13.5 ≈ 1.1e-4
GRAD_ACCUM=${GRAD_ACCUM:-2}
LAMBDA_V=${LAMBDA_V:-0.0}              # P3 = 1.0
LAMBDA_A=${LAMBDA_A:-1.0}              # P3 = 5.0  (GigaWorld 5:1 action:video)
WARMUP_STEPS=${WARMUP_STEPS:-0}
COSINE_STEPS=${COSINE_STEPS:-0}
CKPT_DIR=${CKPT_DIR:-/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/tau0_fold_${PHASE}_32g}
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
INIT_CKPT=${INIT_CKPT:-/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/checkpoints/tau-0-wm}
RESUME=${RESUME:-}
EXTRA=${EXTRA:-}
export TAU0_CHUNK=${TAU0_CHUNK:-5}              # P3 = 9 (native contiguous, T_lat=3); P4 = 33 (T_lat=9)
export TAU0_LATENT_DIR=${TAU0_LATENT_DIR:-vae_latent}   # P3 = vae_latent_c9; P4 = vae_latent_c33
export TAU0_COND_NOISE=${TAU0_COND_NOISE:-0.0}  # P4 = 0.05 (exposure-bias mitigation)
USE_DEEPSPEED=${USE_DEEPSPEED:-0}               # P3 = 1 (full-FT needs ZeRO-2)

echo "[aihc-tau0] node $NODE_RANK/$NNODES gpus=$NUM_GPUS total=$NPROC_TOTAL master=$MASTER_ADDR:$MASTER_PORT"
echo "[aihc-tau0] phase=$PHASE steps=$MAX_STEPS lr=$LR λa=$LAMBDA_A λv=$LAMBDA_V warmup=$WARMUP_STEPS chunk=$TAU0_CHUNK latdir=$TAU0_LATENT_DIR ds=$USE_DEEPSPEED"
"$VENV/bin/python" -c 'import torch;print("[aihc-tau0] torch",torch.__version__,"gpus",torch.cuda.device_count())' || true

if [ "$USE_DEEPSPEED" = 1 ]; then
  DIST_FLAGS="--use_deepspeed --zero_stage 2 --deepspeed_multinode_launcher standard --gradient_accumulation_steps $GRAD_ACCUM"
else
  DIST_FLAGS="--multi_gpu"
fi

exec "$VENV/bin/accelerate" launch \
  $DIST_FLAGS --num_machines "$NNODES" --num_processes "$NPROC_TOTAL" --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --mixed_precision bf16 --dynamo_backend no \
  finetune/run_train.py --phase "$PHASE" --max_steps "$MAX_STEPS" --lr "$LR" \
  --grad_accum "$GRAD_ACCUM" --lambda_v "$LAMBDA_V" --lambda_a "$LAMBDA_A" \
  --warmup_steps "$WARMUP_STEPS" --cosine_steps "$COSINE_STEPS" \
  --ckpt_dir "$CKPT_DIR" --ckpt_interval "$CKPT_INTERVAL" --init_ckpt "$INIT_CKPT" \
  ${RESUME:+--resume "$RESUME"} $EXTRA
