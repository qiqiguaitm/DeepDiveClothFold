#!/usr/bin/env bash
# Poll gf download; when complete, scp wheels to local find-links dir.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
GF="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 55555 tim@14.103.44.161"
LOCAL=/mnt/pfs/p46h4f/cosmos/github_wheels
mkdir -p "$LOCAL"
for t in $(seq 1 240); do   # up to ~4h (60s ticks)
  done=$($GF "grep -c GF_WHEEL_DL_DONE /home/tim/gf_wheel_dl.log 2>/dev/null" 2>/dev/null)
  prog=$($GF "ls -l /home/tim/cosmos_wheels/*.whl 2>/dev/null | wc -l; du -sh /home/tim/cosmos_wheels 2>/dev/null" 2>/dev/null)
  echo "tick $t $(date +%H:%M:%S): done=$done | $prog"
  if [ "$done" = "1" ]; then break; fi
  sleep 60
done
echo "=== transferring wheels gf -> local ==="
scp -o StrictHostKeyChecking=no -o BatchMode=yes -P 55555 \
  tim@14.103.44.161:/home/tim/cosmos_wheels/*.whl "$LOCAL/" 2>&1 | tail -4
echo "=== local wheels ==="; ls -lh "$LOCAL"/*.whl 2>/dev/null
echo "=== FETCH_DONE ==="
