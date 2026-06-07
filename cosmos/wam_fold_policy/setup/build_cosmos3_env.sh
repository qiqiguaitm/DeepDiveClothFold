#!/usr/bin/env bash
# Build the Cosmos3 development environment (Diffusers generator path) into a
# dedicated venv under /mnt/pfs/p46h4f/cosmos. Uses the local proxy for
# GitHub/PyPI; retries around transient proxy errors.
set -uo pipefail
cd /mnt/pfs/p46h4f/cosmos

ENV_DIR=/mnt/pfs/p46h4f/cosmos/cosmos3-venv
TORCH_INDEX="https://download.pytorch.org/whl/cu124"   # matches A100 + driver 535

# The proxy here negotiates HTTP/2 poorly for large git/wheel transfers
# (curl 16 "HTTP2 framing layer" / "operation timed out"). Force HTTP/1.1 and a
# big post buffer so git fetch (used by uv too) and curl stay stable.
git config --global http.version HTTP/1.1
git config --global http.postBuffer 1048576000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
export UV_HTTP_TIMEOUT=600

retry() { # retry <n> <cmd...>
  local n=$1; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "---- attempt $i/$n failed: $* ; retry in 20s ----"; sleep 20
  done
  return 1
}

echo "==== $(date '+%F %T') uv $(uv --version) ===="

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "==== create venv (python 3.13) at $ENV_DIR ===="
  uv venv --python 3.13 --seed "$ENV_DIR" 2>&1 | tail -3 \
    || uv venv --python 3.11 --seed "$ENV_DIR" 2>&1 | tail -3
fi
echo "venv python: $("$ENV_DIR/bin/python" --version)"

echo "==== install torch/torchvision (cu124) ===="
retry 5 uv pip install --python "$ENV_DIR" --index-url "$TORCH_INDEX" torch torchvision 2>&1 | tail -6

echo "==== fetch diffusers (git main) via manual clone ===="
SRC=/mnt/pfs/p46h4f/cosmos/.src/diffusers
mkdir -p /mnt/pfs/p46h4f/cosmos/.src
if [ ! -d "$SRC/.git" ]; then
  retry 5 git clone --depth 1 https://github.com/huggingface/diffusers.git "$SRC" 2>&1 | tail -4
else
  ( cd "$SRC" && git pull --ff-only 2>&1 | tail -2 )
fi

echo "==== install Cosmos3 Diffusers generator stack ===="
retry 5 uv pip install --python "$ENV_DIR" \
  "$SRC" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  transformers 2>&1 | tail -15

echo "==== verify ===="
"$ENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
import transformers, diffusers, accelerate, av, imageio
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
try:
    from diffusers import Cosmos3OmniPipeline
    print("Cosmos3OmniPipeline: OK")
except Exception as e:
    print("Cosmos3OmniPipeline: FAIL ->", repr(e))
try:
    import cosmos_guardrail
    print("cosmos_guardrail: OK")
except Exception as e:
    print("cosmos_guardrail: WARN ->", repr(e))
PY
echo "==== ENV BUILD FINISHED $(date '+%F %T') ===="
