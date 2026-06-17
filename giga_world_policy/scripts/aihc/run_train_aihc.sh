#!/bin/bash
# 百度 AIHC 多节点 launcher —— GigaWorld-Policy(WAM)叠衣服 full-FT,Wan2.2-TI2V-5B backbone。
# 拓扑:NNODES 个 pod × 8×A100-80G(默认 2 节点 = 16 卡)。
# 提交方式(PyTorchJob):POST /api/v1/aijobs,每 pod 命令:
#   bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy/scripts/aihc/run_train_aihc.sh
# AIHC PyTorchJob 给每个 pod 注入:WORLD_SIZE(=#节点)、RANK(=节点序)、MASTER_ADDR、MASTER_PORT。
# 参考 /mnt/pfs/p46h4f/cosmos/dreamzero/tools/run_train_aihc.sh。
set -e
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
source "$REPO/env.sh"          # uv venv(torch2.6)+ HF 镜像/离线 + 本项目权重/数据路径
cd "$REPO"

# ---- AIHC PyTorchJob 注入的 env -> 训练所需 ----
export NUM_GPUS=${NUM_GPUS:-8}                      # 每节点 GPU 数
export NNODES=${NNODES:-${WORLD_SIZE:-2}}           # 节点数(replicas)
export NODE_RANK=${NODE_RANK:-${RANK:-0}}           # 本 pod 节点序
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
NPROC_TOTAL=$((NNODES * NUM_GPUS))

# ---- job pod 无外网(代理只在 dev/login box)→ 强制离线,清掉死代理 ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$REPO"

# ---- RDMA/IB:cosmos 镜像缺 libibverbs,从 PFS 暂存 jammy rdma-core 用户态库进容器(复用 dreamzero 的)----
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
    export NCCL_IB_DISABLE=1                          # 无 IB(如 dev box)→ 退回 TCP/eth0
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}  # bootstrap/带外
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CONFIG=${CONFIG:-world_action_model.configs.visrobot01_fold_16gpu.config}

# stdout/stderr 落盘，方便在 PFS 上直接查错误（类似 fastwam run_train_aihc.sh）
LOG_DIR="$REPO/runs/$(echo "$CONFIG" | python3 -c 'import sys; p=sys.stdin.read().strip().split(".")[-2]; print(p)')/aihc_logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pod_${NODE_RANK:-0}.stdout") 2>&1

echo "[aihc] node $NODE_RANK/$NNODES  gpus/node=$NUM_GPUS  total=$NPROC_TOTAL  master=$MASTER_ADDR:$MASTER_PORT"
echo "[aihc] IB=$([ "$NCCL_IB_DISABLE" = 0 ] && echo on || echo off) ibdev='$HAS_IB' config=$CONFIG"
python -c 'import torch;print("[aihc] torch",torch.__version__,"cuda",torch.cuda.is_available(),"gpus",torch.cuda.device_count())' || true

# ---- 预热 HF datasets arrow cache(单进程),规避多 rank 并发重建竞态 ----
# parquet 改动会让 cache 失效;若 40 个 rank(跨节点、共享 PFS)同时重建 → 半成品 arrow →
# FileNotFoundError(...incomplete...) / check_timestamps_sync IndexError → 整个 job 崩。
# 仅 NODE_RANK 0 单进程 warm(idempotent:cache 在则秒读),其余节点等 sentinel。
WARM_SENTINEL="$LOG_DIR/.cache_ready"
if [ "${NODE_RANK:-0}" = "0" ]; then
    rm -f "$WARM_SENTINEL"
    echo "[aihc] node 0 warming dataset cache (single-process)..."
    python -m scripts.aihc.warm_cache --config "$CONFIG" || echo "[aihc] WARN cache warm failed (non-fatal)"
    touch "$WARM_SENTINEL"; echo "[aihc] cache warm done -> $WARM_SENTINEL"
else
    sleep 10  # 给 node 0 先删旧 sentinel 的窗口
    echo "[aihc] node $NODE_RANK waiting for cache warm (sentinel $WARM_SENTINEL)..."
    for _ in $(seq 1 240); do [ -f "$WARM_SENTINEL" ] && break; sleep 5; done
    echo "[aihc] node $NODE_RANK proceeding (sentinel $([ -f "$WARM_SENTINEL" ] && echo found || echo TIMEOUT))"
fi

# ---- accelerate launch(DeepSpeed ZeRO-2,standard 多机 launcher:每 pod 独立起,不依赖 pdsh)----
exec accelerate launch \
  --use_deepspeed --zero_stage 2 --deepspeed_multinode_launcher standard \
  --gradient_accumulation_steps 1 \
  --num_machines "$NNODES" --num_processes "$NPROC_TOTAL" --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --mixed_precision bf16 --dynamo_backend no \
  third_party/giga-train/giga_train/distributed/run_task.py --config "$CONFIG"
