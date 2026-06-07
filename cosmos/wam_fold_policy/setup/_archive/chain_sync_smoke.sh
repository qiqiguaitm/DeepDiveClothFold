#!/usr/bin/env bash
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/setup/_archive
until grep -qE "SYNCV2 DONE" sync_locallock.log 2>/dev/null; do sleep 30; done
echo "=== ENV SYNC FINISHED $(date +%H:%M:%S) ==="
grep -E "SYNC OK|rc=|GPU OK|torch |SYNCV2 DONE|Failed|Caused by|error:" sync_locallock.log | tail -6
if [ -x /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3/.venv/bin/python ]; then
  echo "=== VENV READY -> SMOKE VALIDATION $(date +%H:%M:%S) ==="
  SMOKE_ITERS=10 bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/smoke_validate.sh > smoke_validation.log 2>&1
  echo "smoke rc=$?"; tail -12 smoke_validation.log
else
  echo "VENV NOT READY"
fi
echo "=== CHAIN_SMOKE_DONE ==="
