#!/usr/bin/env bash
cd /mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports
# wait for the 16-GPU eval to finish (frees GPUs)
until grep -qE "EVAL_16GPU_DONE" run_16gpu.log 2>/dev/null; do sleep 60; done
echo "=== eval done, GPUs free -> cross-rig smoke validation $(date) ==="
sleep 10
# smoke_validate now uses recipe_nano with the mixed ConcatDataset[vis x3, kai]
SMOKE_ITERS=12 NGPU=8 bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/smoke_validate.sh > smoke_crossrig.log 2>&1
echo "smoke rc=$?"
echo "--- losses ---"; grep -E "\[RANK 0\] Iteration.*Loss:|PASS|FAIL|Done with training" smoke_crossrig.log 2>/dev/null | tail -14
echo "--- domain/rig evidence (both 16 & 17 loaded?) ---"; grep -iE "domain|kairobot|visrobot|ConcatDataset|wam_fold" smoke_crossrig.log 2>/dev/null | tail -6
echo "=== CROSSRIG_SMOKE_DONE ==="
