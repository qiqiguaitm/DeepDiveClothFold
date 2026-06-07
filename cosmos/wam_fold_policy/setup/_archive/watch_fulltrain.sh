#!/usr/bin/env bash
OUT=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/train_out_single
LOG=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports/full_train.log
for t in $(seq 1 480); do  # up to ~4h of 30s ticks
  if grep -qE "FULL_TRAIN_DONE|Error|Traceback|OutOfMemory" "$LOG" 2>/dev/null; then echo "TRAIN ENDED/ERRORED"; break; fi
  ckpt=$(find "$OUT" -type d -name "iter_*" 2>/dev/null | sort | tail -1)
  if [ -n "$ckpt" ]; then echo "FIRST CHECKPOINT SAVED: $ckpt $(date +%H:%M:%S)"; break; fi
  sleep 30
done
echo "--- latest loss ---"; grep -E "\[RANK 0\] Iteration" "$LOG" 2>/dev/null | tail -1
echo "--- checkpoints ---"; find "$OUT" -type d -name "iter_*" 2>/dev/null | sort
echo "=== WATCH_FULLTRAIN_DONE ==="
