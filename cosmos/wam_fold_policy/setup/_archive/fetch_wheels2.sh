#!/usr/bin/env bash
# Poll gf per-wheel; scp each to local find-links dir as soon as it completes at full size.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
GF="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 55555 tim@14.103.44.161"
L=/mnt/pfs/p46h4f/cosmos/github_wheels; mkdir -p "$L"
declare -A SZ=(
 ["natten-0.21.6.dev6+cu128.torch210-cp313-cp313-linux_x86_64.whl"]=230031651
 ["transformer_engine-2.12+cu128.torch210-cp313-cp313-linux_x86_64.whl"]=222087623
 ["flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl"]=278542748
 ["flash_attn-2.7.4.post1+cu128.torch210-cp313-cp313-linux_x86_64.whl"]=406220379
)
for t in $(seq 1 720); do   # up to ~12h
  alllocal=1
  for n in "${!SZ[@]}"; do
    la=$(stat -c%s "$L/$n" 2>/dev/null || echo 0)
    if [ "$la" = "${SZ[$n]}" ]; then continue; fi
    alllocal=0
    ga=$($GF "stat -c%s /home/tim/cosmos_wheels/'$n' 2>/dev/null || echo 0" 2>/dev/null)
    if [ "$ga" = "${SZ[$n]}" ]; then
      echo "[$(date +%H:%M:%S)] $n complete on gf ($ga) -> scp"
      scp -o StrictHostKeyChecking=no -o BatchMode=yes -P 55555 "tim@14.103.44.161:/home/tim/cosmos_wheels/$n" "$L/" 2>/dev/null
    fi
  done
  if [ $alllocal = 1 ]; then echo "ALL 4 WHEELS LOCAL $(date +%H:%M:%S)"; break; fi
  sleep 60
done
echo "=== local wheels ==="; ls -lh "$L"/*.whl 2>/dev/null
echo "=== FETCH2_DONE ==="
