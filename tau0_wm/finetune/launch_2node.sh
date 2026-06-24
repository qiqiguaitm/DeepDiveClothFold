#!/usr/bin/env bash
# 2-node x 8-GPU = 16 distributed launch for tau0 joint fine-tune.
#   b2 = this node (192.168.20.128) = rank 0 / master
#   b1 = remote (192.168.20.169, ssh -p 429) = rank 1
# Usage: bash finetune/launch_2node.sh [run_train.py args...]
set -u
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm
VENV=/mnt/pfs/p46h4f/cosmos/.venv
ACC="$VENV/bin/accelerate"
MASTER_IP=192.168.20.128
PORT=29501
B1_SSH="ssh -p 429 -o BatchMode=yes -o StrictHostKeyChecking=no root@120.48.99.93"
COMMON="--num_machines 2 --num_processes 16 --main_process_ip $MASTER_IP --main_process_port $PORT --mixed_precision bf16 --dynamo_backend no"
ENVV="NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
ARGS="$*"

echo "[launch] rank1 on b1 (192.168.20.169) ..."
$B1_SSH "cd $REPO && $ENVV $ACC launch $COMMON --machine_rank 1 finetune/run_train.py $ARGS" \
  > /tmp/tau0_rank1.log 2>&1 &
B1_PID=$!
echo "[launch] rank1 ssh pid=$B1_PID (log: /tmp/tau0_rank1.log on b2)"

echo "[launch] rank0 on b2 (master) ..."
cd "$REPO"
env $ENVV $ACC launch $COMMON --machine_rank 0 finetune/run_train.py $ARGS
RC=$?
echo "[launch] rank0 exited rc=$RC; waiting rank1 ..."
wait $B1_PID
echo "[launch] done."
