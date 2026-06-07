#!/usr/bin/env bash
set -uo pipefail
cd /mnt/pfs/p46h4f/cosmos
ENV_DIR=/mnt/pfs/p46h4f/cosmos/cosmos3-venv
# Bypass the flaky proxy; use a fast, directly-reachable CN PyPI mirror
# (same direct-route strategy that makes the ModelScope download reliable).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*" NO_PROXY="*" UV_HTTP_TIMEOUT=900
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
retry() { local n=$1; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "---- attempt $i/$n failed; retry in 20s ----"; sleep 20; done; return 1; }
echo "==== $(date '+%F %T') install torch==2.6.0 (cu124) via Tsinghua mirror, proxy bypassed ===="
retry 8 uv pip install --python "$ENV_DIR" --reinstall \
  --index-url "$MIRROR" \
  torch==2.6.0 torchvision==0.21.0 2>&1 | tail -8
echo "==== verify ===="
"$ENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available(): print("device0:", torch.cuda.get_device_name(0))
from diffusers import Cosmos3OmniPipeline; print("Cosmos3OmniPipeline: OK")
PY
echo "==== TORCH FIX3 FINISHED $(date '+%F %T') ===="
