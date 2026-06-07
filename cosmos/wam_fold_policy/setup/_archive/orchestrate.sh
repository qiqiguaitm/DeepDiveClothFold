#!/usr/bin/env bash
# Wait for wheels to be fetched locally, then run the find-links direct sync.
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/setup/_archive
# wait for all 4 wheels present with correct sizes
declare -A SZ=(
 ["natten-0.21.6.dev6+cu128.torch210-cp313-cp313-linux_x86_64.whl"]=230031651
 ["flash_attn-2.7.4.post1+cu128.torch210-cp313-cp313-linux_x86_64.whl"]=406220379
 ["flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl"]=278542748
 ["transformer_engine-2.12+cu128.torch210-cp313-cp313-linux_x86_64.whl"]=222087623
)
L=/mnt/pfs/p46h4f/cosmos/github_wheels
for t in $(seq 1 900); do
  ok=1
  for n in "${!SZ[@]}"; do
    a=$(stat -c%s "$L/$n" 2>/dev/null || echo 0)
    [ "$a" = "${SZ[$n]}" ] || ok=0
  done
  if [ $ok = 1 ]; then echo "all 4 wheels present $(date +%H:%M:%S)"; break; fi
  sleep 60
done
echo "=== launching find-links direct sync $(date +%H:%M:%S) ==="
bash do_sync.sh > sync_finallinks.log 2>&1
echo "=== ENV_SYNC_DONE ==="
tail -6 sync_finallinks.log
# chain: if venv built, attempt the smoke training validation autonomously
if [ -x /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3/.venv/bin/python ]; then
  echo "=== venv ready -> running smoke validation $(date +%H:%M:%S) ==="
  SMOKE_ITERS=10 bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/smoke_validate.sh > smoke_validation.log 2>&1
  echo "smoke validation rc=$?"
  tail -8 smoke_validation.log
else
  echo "venv NOT ready - smoke validation skipped (env build did not finish)"
fi
echo "=== ORCHESTRATE_DONE ==="
