#!/usr/bin/env bash
# Download Cosmos3 models from ModelScope (nv-community), bypassing the local proxy.
set -uo pipefail
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos

# Bypass proxy for ModelScope (CN endpoint is directly reachable; proxy is not needed)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*" NO_PROXY="*"
export MODELSCOPE_DOMAIN="www.modelscope.cn"

MS=/mnt/pfs/p46h4f/cosmos/.ms-tool/bin/modelscope
DEST=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/models/modelscope

MODELS=(
  "nv-community/Cosmos3-Nano"
  "nv-community/Cosmos3-Super"
  "nv-community/Cosmos3-Super-Image2Video"
)

for m in "${MODELS[@]}"; do
  name="${m#*/}"
  echo "==== $(date '+%F %T') START $m -> $DEST/$name ===="
  for attempt in 1 2 3 4 5; do
    "$MS" download --model "$m" --local_dir "$DEST/$name" && { echo "==== DONE $m ===="; break; }
    echo "---- attempt $attempt failed for $m, retrying in 30s ----"
    sleep 30
  done
done
echo "==== ALL DOWNLOADS FINISHED $(date '+%F %T') ===="
