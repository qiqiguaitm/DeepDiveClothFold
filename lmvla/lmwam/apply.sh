#!/bin/bash
# 把 lmwam 的修改层应用到干净的 lawam submodule。
#
# 结构(高内聚: 只含我们拥有的东西, 与上游重叠部分全删, 由 submodule 提供):
#   patches/   对上游文件的修改(diff -u, apply 到 ../lawam)
#   adapter/   我们的核心实现(lmwm_adapter / milestone_target), 运行时经 LMWM_ADAPTER_DIR 注入, 无需 patch
#   scripts/   我们的 run_*.sh / robotwin wrapper
#   configs/   我们的训练配置
#   env/       环境搭建/下载脚本
#
# 用法:
#   ./apply.sh          应用所有 patch 到 ../lawam
#   ./apply.sh --check  只校验能否干净应用(不改文件)
#   ./revert.sh         还原(git -C ../lawam checkout 相关文件)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAWAM="$(cd "$HERE/../lawam" && pwd)"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

[ -d "$LAWAM/.git" ] || [ -f "$LAWAM/.git" ] || { echo "FATAL: $LAWAM 不是 lawam submodule" >&2; exit 1; }

fail=0
while IFS= read -r target; do
  [ -z "$target" ] && continue
  pname="$HERE/patches/$(echo "$target" | tr '/' '~').patch"
  [ -f "$pname" ] || { echo "FATAL: 缺 patch $pname" >&2; fail=1; continue; }
  if patch --dry-run -p1 -d "$LAWAM" < "$pname" >/dev/null 2>&1; then
    if [ "$CHECK" = 0 ]; then
      patch -p1 -d "$LAWAM" < "$pname" >/dev/null && echo "  ✅ apply  $target"
    else
      echo "  ✅ 可应用 $target"
    fi
  elif [ "$CHECK" = 0 ] && patch -R --dry-run -p1 -d "$LAWAM" < "$pname" >/dev/null 2>&1; then
    echo "  ⏭  已应用 $target(跳过)"
  else
    echo "  ❌ 无法应用 $target" >&2; fail=1
  fi
done < "$HERE/patches/MANIFEST.txt"

# adapter 目录经环境变量注入, 提示一下
echo ""
echo "[提示] 训练/评测时需设: export LMWM_ADAPTER_DIR=$HERE/adapter"
echo "       (lawam.py 的 LMWM 分支据此 import lmwm_adapter / lmwm_milestone_target)"
[ "$fail" = 0 ] && echo "[apply.sh] 完成" || { echo "[apply.sh] 有失败项" >&2; exit 1; }
