#!/usr/bin/env bash
# 2-node (b0 rank0 + b1 rank1) FSDP-16 full fine-tune of wam_fold_nano. Run on b0 (192.168.20.129).
# IMPORTANT: do NOT source the train env here (its conda LD_LIBRARY_PATH breaks ssh's libssl).
# Each rank sources the env inside its own subshell.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY LD_LIBRARY_PATH; export no_proxy='*'
ENVF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/env.sh
TOML=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/recipe_nano.toml
LOGDIR=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports
mkdir -p "$LOGDIR"
MAXITER=${MAXITER:-50000}; SAVEITER=${SAVEITER:-1000}
B1="ssh -p 429 -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@120.48.99.93"

# torchrun command for a given node_rank ($1); sources env inside subshell so LD_LIBRARY_PATH is local
mk() { echo "source $ENVF && exec \"\$VENV/bin/torchrun\" --nnodes=2 --nproc_per_node=8 --node_rank=$1 --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT -m cosmos_framework.scripts.train --sft-toml=$TOML -- trainer.max_iter=$MAXITER checkpoint.save_iter=$SAVEITER"; }

echo "=== 2NODE TRAIN start $(date) | max_iter=$MAXITER save_iter=$SAVEITER ==="
echo "[launch] rank1 -> b1 (192.168.20.169)"
$B1 "nohup bash -c '$(mk 1)' > $LOGDIR/train_2node_rank1.log 2>&1 & echo b1_rank1_pid \$!" 2>&1 | grep -v "OpenSSL" | tail -2

echo "[launch] rank0 -> b0 (192.168.20.129)"
nohup bash -c "$(mk 0)" > "$LOGDIR/train_2node_rank0.log" 2>&1 &
R0=$!
echo "rank0 pid $R0"
wait $R0
echo "=== 2NODE_TRAIN rank0 rc=$? $(date) ==="
echo "=== 2NODE_TRAIN_DONE ==="
