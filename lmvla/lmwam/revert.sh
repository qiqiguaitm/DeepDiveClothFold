#!/bin/bash
# 还原 lawam submodule 到纯净状态(丢弃所有 patch 修改)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAWAM="$(cd "$HERE/../lawam" && pwd)"
while IFS= read -r target; do
  [ -z "$target" ] && continue
  git -C "$LAWAM" checkout -- "$target" 2>/dev/null && echo "  还原 $target"
done < "$HERE/patches/MANIFEST.txt"
echo "[revert.sh] lawam 已还原, 改动数: $(git -C "$LAWAM" status --short | wc -l)"
