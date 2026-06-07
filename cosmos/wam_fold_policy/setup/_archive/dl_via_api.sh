#!/usr/bin/env bash
# Download GitHub release assets via api.github.com (redirects to reachable objects.githubusercontent.com)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
DEST=/mnt/pfs/p46h4f/cosmos/github_wheels
mkdir -p "$DEST"
API="https://api.github.com/repos/nvidia-cosmos/cosmos-dependencies/releases/assets"
# id name
ASSETS=(
  "367904384 flash_attn-2.7.4.post1+cu128.torch210-cp313-cp313-linux_x86_64.whl"
  "370347281 flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl"
  "386399291 natten-0.21.6.dev6+cu128.torch210-cp313-cp313-linux_x86_64.whl"
  "367665227 transformer_engine-2.12+cu128.torch210-cp313-cp313-linux_x86_64.whl"
)
for row in "${ASSETS[@]}"; do
  id="${row%% *}"; name="${row#* }"
  echo "=== [$name] id=$id $(date +%H:%M:%S) ==="
  for r in 1 2 3 4 5; do
    curl -sL -H "Accept: application/octet-stream" \
      --connect-timeout 30 --max-time 1200 --retry 2 \
      -o "$DEST/$name" "$API/$id"
    sz=$(stat -c%s "$DEST/$name" 2>/dev/null || echo 0)
    # wheels are >100MB; reject small/HTML error bodies
    if [ "$sz" -gt 100000000 ]; then
      echo "DONE $(numfmt --to=iec $sz) $(date +%H:%M:%S)"; break
    fi
    echo "  try $r got only ${sz}B, retry in 8s"; sleep 8
  done
done
echo "=== SUMMARY ==="; ls -lh "$DEST/"
echo "=== DL_API_DONE ==="
