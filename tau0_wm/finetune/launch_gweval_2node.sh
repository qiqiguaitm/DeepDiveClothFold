#!/usr/bin/env bash
# 2-node x 8-GPU = 16 distributed EVAL of a tau0 joint checkpoint on visrobot01_val.
#   master (this node, 192.168.20.128) = rank0 ; worker (192.168.20.169, ssh -p 429) = rank1
# Usage: bash finetune/launch_eval_2node.sh [eval_gigaworld_dist.py args...]
set -u
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm
VENV=/mnt/pfs/p46h4f/cosmos/.venv
ACC="$VENV/bin/accelerate"
MASTER_IP=192.168.20.128
PORT=29503
B1_SSH="ssh -p 429 -o BatchMode=yes -o StrictHostKeyChecking=no root@120.48.99.93"
COMMON="--num_machines 2 --num_processes 16 --main_process_ip $MASTER_IP --main_process_port $PORT --mixed_precision bf16 --dynamo_backend no"
ENVV="NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
ARGS="$*"

echo "[eval-launch] rank1 on 192.168.20.169 ..."
$B1_SSH "cd $REPO && $ENVV $ACC launch $COMMON --machine_rank 1 finetune/eval_gigaworld_dist.py $ARGS" \
  > /tmp/tau0_eval_rank1.log 2>&1 &
B1_PID=$!
echo "[eval-launch] rank0 on master ..."
cd "$REPO"
env $ENVV $ACC launch $COMMON --machine_rank 0 finetune/eval_gigaworld_dist.py $ARGS
echo "[eval-launch] rank0 done; waiting rank1 ..."
wait $B1_PID
echo "[eval-launch] done."
