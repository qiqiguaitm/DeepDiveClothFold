#!/usr/bin/env bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy="*"
DEST=/mnt/pfs/p46h4f/cosmos/github_wheels
mkdir -p "$DEST"
WHEELS=(
  "natten-0.21.6.dev6%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl"
  "flash_attn-2.8.3%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl"
  "flash_attn_3_nv-3.0.0.post1%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl"
  "transformer_engine-2.12%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl"
)
BASE="https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.5.0"
for w in "${WHEELS[@]}"; do
  fname=$(echo "$w" | sed 's/%2B/+/g')
  echo "=== $fname ==="
  for r in 1 2 3 4 5; do
    curl -L --retry 3 --retry-delay 10 --connect-timeout 30 --max-time 300 \
      -o "$DEST/$fname" "$BASE/$w" 2>&1 | tail -1
    if [ -s "$DEST/$fname" ]; then
      ls -lh "$DEST/$fname" | awk '{print "OK "$5}'
      break
    fi
    echo "retry $r failed, wait 10s"; sleep 10
  done
done
echo "=== done ==="; ls -lh "$DEST/"
