#!/usr/bin/env bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy="*" NO_PROXY="*"
export UV_HTTP_TIMEOUT=600 GIT_LFS_SKIP_SMUDGE=1
export UV_CONCURRENT_DOWNLOADS=8
export UV=/mnt/pfs/p46h4f/cosmos/uvbin/uv
export UV_CACHE_DIR=/mnt/pfs/p46h4f/cosmos/uv_cache_root
export UV_PROJECT_ENVIRONMENT=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3/.venv
export TMPDIR=/mnt/pfs/p46h4f/cosmos/uv_build_tmp
mkdir -p "$TMPDIR"
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
ok=0
for a in 1 2 3 4 5; do
  echo "=== sync attempt $a $(date +%H:%M:%S) (uv $($UV --version)) ==="
  "$UV" sync --frozen --all-extras --group=cu128-train --find-links /mnt/pfs/p46h4f/cosmos/github_wheels > sync_v2_$a.log 2>&1
  rc=$?
  grep -E "Caused by|error:|fatal|No solution|Built [0-9]|Installed [0-9]|Prepared [0-9]|Resolved [0-9]" sync_v2_$a.log | tail -4
  if [ $rc -eq 0 ]; then echo "=== SYNC OK ==="; ok=1; break; fi
  echo "attempt $a rc=$rc; retry 10s"; sleep 10
done
if [ $ok -eq 1 ]; then
  echo "=== verify torch/cuda ==="
  .venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available());x=torch.randn(8,device='cuda');print('GPU OK',float((x*x).sum()))" 2>&1 | tail -4
fi
echo "=== SYNCV2 DONE ok=$ok ==="
