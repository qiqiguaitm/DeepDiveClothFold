#!/usr/bin/env bash
OUT=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/train_out_2node
R0=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports/train_2node_rank0.log
for t in $(seq 1 120); do
  ck=$(find "$OUT" -type d -name "iter_*" 2>/dev/null | sort | tail -1)
  if [ -n "$ck" ]; then echo "CHECKPOINT SAVED: $ck $(date +%H:%M:%S)"; fi
  if grep -qE "2NODE_TRAIN_DONE|Done with training" "$R0" /mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports/run_2node.log 2>/dev/null; then echo "TRAIN DONE"; break; fi
  if grep -qE "Error|Traceback|OutOfMemory" "$R0" 2>/dev/null; then echo "TRAIN ERROR"; break; fi
  sleep 20
done
echo "--- final losses ---"; grep -E "\[RANK 0\] Iteration.*Loss:" "$R0" 2>/dev/null | tail -3 | grep -oE "Iteration [0-9]+.*"
echo "--- checkpoints ---"; find "$OUT" -maxdepth 3 -type d -name "iter_*" 2>/dev/null | sort
echo "=== WATCH_10STEP_DONE ==="
