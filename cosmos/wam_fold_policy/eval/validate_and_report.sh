#!/usr/bin/env bash
# Kill in-flight eval/smoke, then: (1) cross-rig smoke validation, (2) 16-GPU eval with 10 viz eps.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY LD_LIBRARY_PATH; export no_proxy='*'
WFP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy   # scripts
D=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports            # outputs/logs
B1="ssh -p 429 -o BatchMode=yes -o StrictHostKeyChecking=no root@120.48.99.93"
# --- kill in-flight ---
for pat in run_16gpu.sh shard.sh eval_report chain_crossrig_smoke smoke_validate cosmos_framework.scripts.train; do
  pkill -9 -f "$pat" 2>/dev/null
done
$B1 "for pat in shard.sh eval_report cosmos_framework.scripts.train; do for p in \$(ps -eo pid,cmd|grep -E \"[c]osmos3/.venv/bin/python.*\$pat|[s]hard.sh\"|awk '{print \$1}'); do kill -9 \$p 2>/dev/null; done; done; echo b1_killed" 2>&1 | grep -v OpenSSL | tail -1
sleep 6
echo "b0 GPUs: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|paste -sd, -)"

# --- (1) cross-rig smoke validation (mixed dataset trains; both domains load) ---
echo "=== [1] cross-rig smoke validation $(date) ==="
SMOKE_ITERS=12 NGPU=8 bash "$WFP/train/smoke_validate.sh" > "$D/smoke_crossrig.log" 2>&1
echo "smoke rc=$?"
grep -E "\[RANK 0\] Iteration.*Loss:|PASS|FAIL|Done with training" "$D/smoke_crossrig.log" 2>/dev/null | tail -8
# free GPUs from smoke
pkill -9 -f cosmos_framework.scripts.train 2>/dev/null; sleep 5

# --- (2) 16-GPU eval, 10 viz episodes -> report.html ---
echo "=== [2] 16-GPU eval (10 viz eps) $(date) ==="
NMETRIC=20 NVIZ=10 bash "$WFP/eval/run_16gpu.sh" > "$D/run_16gpu.log" 2>&1
grep -E "cosmos mae|EVAL_16GPU_DONE" "$D/run_16gpu.log" 2>/dev/null | tail -3
echo "=== P4_DONE $(date) ==="
