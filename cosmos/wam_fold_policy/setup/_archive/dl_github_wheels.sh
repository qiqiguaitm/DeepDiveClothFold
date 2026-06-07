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
  echo "=== [$fname] $(date +%H:%M:%S) ==="
  # single download, 15-min timeout, follow redirects, resume
  wget -c --tries=1 --timeout=30 --read-timeout=900 --progress=dot:giga \
    -O "$DEST/$fname" "$BASE/$w" 2>&1 | tail -3
  if [ -s "$DEST/$fname" ]; then
    actual=$(stat -c%s "$DEST/$fname" 2>/dev/null)
    echo "DONE $(numfmt --to=iec $actual) $(date +%H:%M:%S)"
  else
    echo "FAILED $(date +%H:%M:%S)"
  fi
done
echo "=== SUMMARY ==="; ls -lh "$DEST/"
