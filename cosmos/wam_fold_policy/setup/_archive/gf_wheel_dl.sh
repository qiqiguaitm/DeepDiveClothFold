#!/usr/bin/env bash
# Run ON gf. Parallel-chunked download of 4 cosmos cu128 wheels via local proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
PX=http://localhost:29290
DEST=/home/tim/cosmos_wheels
mkdir -p "$DEST"
B="https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.5.0"
# name|url-encoded-file|exact-size
ROWS=(
 "natten-0.21.6.dev6+cu128.torch210-cp313-cp313-linux_x86_64.whl|natten-0.21.6.dev6%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl|230031651"
 "flash_attn-2.7.4.post1+cu128.torch210-cp313-cp313-linux_x86_64.whl|flash_attn-2.7.4.post1%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl|406220379"
 "flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl|flash_attn_3_nv-1.0.3%2Bcu128.torch210-cp39-abi3-linux_x86_64.whl|278542748"
 "transformer_engine-2.12+cu128.torch210-cp313-cp313-linux_x86_64.whl|transformer_engine-2.12%2Bcu128.torch210-cp313-cp313-linux_x86_64.whl|222087623"
)
N=16   # parallel chunks per wheel
for row in "${ROWS[@]}"; do
  IFS='|' read -r name enc size <<< "$row"
  out="$DEST/$name"
  if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null)" = "$size" ]; then
    echo "[$name] already complete ($size)"; continue
  fi
  echo "=== [$name] size=$size N=$N $(date +%H:%M:%S) ==="
  chunk=$(( (size + N - 1) / N ))
  pdir="$DEST/.parts_$name"; mkdir -p "$pdir"
  for i in $(seq 0 $((N-1))); do
    start=$((i*chunk)); end=$((start+chunk-1)); [ $end -ge $size ] && end=$((size-1))
    [ $start -ge $size ] && break
    part="$pdir/p$(printf %03d $i)"
    if [ -f "$part" ] && [ "$(stat -c%s "$part" 2>/dev/null)" = "$((end-start+1))" ]; then continue; fi
    (
      for r in 1 2 3 4 5 6 7 8; do
        curl -sL -x "$PX" -r ${start}-${end} --connect-timeout 30 --max-time 900 -o "$part" "$B/$enc" 2>/dev/null
        got=$(stat -c%s "$part" 2>/dev/null || echo 0)
        [ "$got" = "$((end-start+1))" ] && exit 0
        sleep 5
      done
    ) &
  done
  wait
  # reassemble
  cat "$pdir"/p* > "$out" 2>/dev/null
  actual=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$actual" = "$size" ]; then
    echo "[$name] OK $(numfmt --to=iec $actual) $(date +%H:%M:%S)"; rm -rf "$pdir"
  else
    echo "[$name] INCOMPLETE got=$actual want=$size (parts kept for resume) $(date +%H:%M:%S)"
  fi
done
echo "=== GF_DL_SUMMARY ==="; ls -l "$DEST"/*.whl 2>/dev/null
echo "=== GF_WHEEL_DL_DONE ==="
