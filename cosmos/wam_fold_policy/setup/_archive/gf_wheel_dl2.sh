#!/usr/bin/env bash
# Run ON gf. Resume-until-complete download of 4 cosmos cu128 wheels via local proxy.
# Uses curl -C - so proxy resets RESUME instead of restarting (net forward progress).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
PX=http://localhost:29290
DEST=/home/tim/cosmos_wheels
mkdir -p "$DEST"
B="https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.5.0"
ROWS=(
 "natten-0.21.6.dev6+cu128.torch210-cp313-cp313-linux_x86_64.whl|natten-0.21.6.dev6%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl|230031651"
 "transformer_engine-2.12+cu128.torch210-cp313-cp313-linux_x86_64.whl|transformer_engine-2.12%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl|222087623"
 "flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl|flash_attn_3_nv-1.0.3%2Bcu128.torch210-cp39-abi3-linux_x86_64.whl|278542748"
 "flash_attn-2.7.4.post1+cu128.torch210-cp313-cp313-linux_x86_64.whl|flash_attn-2.7.4.post1%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl|406220379"
)
# launch each wheel as its own resume-loop in parallel (4 streams)
dl_one() {
  local name="$1" enc="$2" size="$3"
  local out="$DEST/$name"
  for it in $(seq 1 400); do
    local cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$size" ]; then echo "[$name] COMPLETE $cur $(date +%H:%M:%S)"; return 0; fi
    # resume from current offset; follow redirect to fresh signed URL each time
    curl -sL -C - -x "$PX" --connect-timeout 25 --max-time 180 -o "$out" "$B/$enc" 2>/dev/null
    sleep 1
  done
  echo "[$name] GAVE UP at $(stat -c%s "$out" 2>/dev/null) / $size"
}
export -f dl_one; export DEST PX B
for row in "${ROWS[@]}"; do
  IFS='|' read -r name enc size <<< "$row"
  ( dl_one "$name" "$enc" "$size" ) &
done
# progress reporter
( for t in $(seq 1 600); do
    sleep 30
    echo "[prog $(date +%H:%M:%S)] $(du -sh $DEST/*.whl 2>/dev/null | tr '\n' ' ')"
  done ) &
PROG=$!
wait %1 %2 %3 %4
kill $PROG 2>/dev/null
echo "=== GF_DL2_SUMMARY ==="; ls -l "$DEST"/*.whl 2>/dev/null
echo "=== GF_WHEEL_DL2_DONE ==="
