#!/usr/bin/env bash
# Hourly incremental sync: TOS Task_A/base → local vis_base, via tosutil cp -r -u.
#
# 设计 (2026-06-23 改): 直接同步上层路径 base/ 下的**所有 v 版本目录**(自动发现, 整版本一次性 cp -r -u),
#   无需硬编码版本号 → 以后 TOS 出新版本 (v5...) 自动同步。
#   - 自动发现: tosutil ls base/ 取所有 v<N>[.<M>] 子目录。
#   - 整版本同步: cp -r -u base/<ver>/ → vis_base/  → 落到 vis_base/<ver>/<date>/ (cp -u 增量, 从不删本地)。
#   - 排除 depth zarr (top_head_depth): 训练只用 RGB 三路。
#   - **跳过 v3** (SKIP_VERS 默认): vis_base/v3 是本地 front-trim+tail-cap 加工产物, 从 TOS 拉会覆盖。
#     VIS_BASE_SKIP_VERS="" 可关闭跳过 (同步所有版本含 v3)。
#   - README.md + analysis/ 在 base/ 根层 → 同步到 vis_base/ 根。
#   背景: TOS 框架演进 v2(老原始,已清空)→ v3(前裁)→ v4(前裁+尾裁+夹爪取主臂 action≠state, 新标准)。
#
# 运行环境 (host-aware: gf0 / gf3 / uc01-03), 各机需本机 ~/.tosutilconfig (cn-shanghai AK/SK)。
# cron (每小时): 17 * * * * bash <repo>/train_scripts/kai/data/sync_vis_base_from_tos.sh
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done

SRC=tos://transfer-shanghai/KAI0/Task_A/base       # README/analysis 仍在 base/ 根层
SKIP_VERS="${VIS_BASE_SKIP_VERS:-v3}"              # 不从 TOS 拉的版本 (空格分隔); 默认 v3 (本地加工产物)
LOCK=/tmp/vis_base_sync.lock
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ---- host-aware: KAI0_ROOT + TOSUTIL 自动探测 (North-E/data-shared 优先于 /vePFS/tim 空壳) ----
_VB=kai0/data/Task_A/vis_base/v2
if   [ -d /vePFS-North-E/vis_robot/workspace/deepdive_kai0/$_VB ];  then KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
elif [ -d /data/shared/ubuntu/workspace/deepdive_kai0/$_VB ];      then KAI0_ROOT=/data/shared/ubuntu/workspace/deepdive_kai0
elif [ -d /vePFS/tim/workspace/deepdive_kai0/$_VB ];               then KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0
else echo "[$(ts)] ERROR: cannot locate deepdive_kai0/$_VB on this host" >&2; exit 1; fi
if   [ -x "$HOME/tosutil" ];        then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null; then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found (~/tosutil or PATH)" >&2; exit 1; fi
DST_ROOT="$KAI0_ROOT/kai0/data/Task_A/vis_base"
LOG="$KAI0_ROOT/logs/vis_base_sync.log"
mkdir -p "$KAI0_ROOT/logs" 2>/dev/null || true

# 防重叠
exec 9>"$LOCK"
flock -n 9 || { echo "[$(ts)] previous run still active, skip" >>"$LOG"; exit 0; }
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ] && mv "$LOG" "$LOG.old"
[ -x "$TOSUTIL" ] || { echo "[$(ts)] ERROR: tosutil missing at $TOSUTIL" >>"$LOG"; exit 1; }
[ -d "$DST_ROOT" ] || { echo "[$(ts)] ERROR: local DST_ROOT missing $DST_ROOT" >>"$LOG"; exit 1; }

# 护栏: 清理 vis_base 根下不该存在的扁平 <date> 残留 (date 应全在 vN/ 下, 见 2026-06-03 迁移事故)。
for fd in "$DST_ROOT"/2026-*-v*; do
  [ -d "$fd" ] || continue
  echo "[$(ts)] WARN 清理 vis_base 根扁平残留: $(basename "$fd")" >>"$LOG"; rm -rf "$fd"
done

# ---- 自动发现 TOS base/ 下所有版本目录 (v2 v3 v4 ...) ----
versions=$("$TOSUTIL" ls -d "$SRC/" 2>/dev/null | grep -oE '/base/v[0-9]+(\.[0-9]+)?/' | grep -oE 'v[0-9]+(\.[0-9]+)?' | sort -u)
if [ -z "$versions" ]; then echo "[$(ts)] ERROR: TOS base/ 下未发现版本目录 (cred/network?)" >>"$LOG"; exit 1; fi

EXCLUDE='*depth*'   # 排除 depth (旧 top_head_depth + v4 新 observation.depth.top_head 两种命名)
echo "[$(ts)] sync start; TOS 版本=[$(echo $versions)] skip=[$SKIP_VERS]" >>"$LOG"
for ver in $versions; do
  case " $SKIP_VERS " in *" $ver "*) echo "[$(ts)] SKIP $ver (本地加工/受保护)" >>"$LOG"; continue ;; esac
  out=$("$TOSUTIL" cp -r -u "$SRC/$ver/" "$DST_ROOT/" -exclude="$EXCLUDE" -j 32 -p 2 2>&1); rc=$?
  succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
  skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
  fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
  if [ "$rc" -eq 0 ]; then
    echo "[$(ts)] OK $ver succ=${succ:-0} skip=${skip:-0} fail=${fail:-0}" >>"$LOG"
  else
    echo "[$(ts)] FAIL $ver rc=$rc succ=${succ:-?} skip=${skip:-?} fail=${fail:-?}" >>"$LOG"; echo "$out" | tail -3 >>"$LOG"
  fi
done

# ---- README + analysis (base/ 根层 → vis_base/ 根) ----
"$TOSUTIL" cp "$SRC/README.md" "$DST_ROOT/README.md" -f >/dev/null 2>&1 \
  && echo "[$(ts)] README.md synced" >>"$LOG" || echo "[$(ts)] WARN README.md sync failed" >>"$LOG"
"$TOSUTIL" cp -r -u "$SRC/analysis/" "$DST_ROOT/" >/dev/null 2>&1 \
  && echo "[$(ts)] analysis/ synced" >>"$LOG" || echo "[$(ts)] WARN analysis/ sync failed" >>"$LOG"
echo "[$(ts)] sync end" >>"$LOG"
