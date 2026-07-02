#!/usr/bin/env bash
# Incremental sync: TOS KAI0/Task_AV1 → local kai0/data/Task_AV1, via tosutil cp -r -u.
#
# 策略:
#   - 增量: `cp -r -u` 按 size/crc 跳过未变文件, 只拉新/变更对象, 从不删除本地 (安全).
#   - 整目录: 直接同步 Task_AV1/ 前缀 (含 base/<date>-v2/{data,videos,meta}, 含 top_head_depth).
#   - 排除 depth (默认): top_head_depth = 数据集 99.7% 对象 (206k 小文件) 且下游不用 → 默认不拉,
#     省空间+提速。需要 depth 时设 V1_INCLUDE_DEPTH=1。
# 运行环境: host-aware (gf0 / gf3 / uc01-03), 各机需本机 ~/.tosutilconfig (cn-shanghai AK/SK).
# cron 安装 (见本目录 README / 末尾注释): @reboot 开机即同步 + 每小时增量。
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done
ts() { date '+%Y-%m-%d %H:%M:%S'; }

SRC=tos://transfer-shanghai/KAI0/Task_AV1
# 2026-06-19 TOS 重整: Task_AV1/base 下插入 v2/ 层 → cp 整 prefix 镜像成本地 Task_AV1/base/v2/<date>。
# 本地已迁入 base/v2/<date>(2026-06-22)对齐 TOS; cp 整 prefix 镜像 Task_AV1/base/v2/<date>。
LOCK=/tmp/task_av1_sync.lock

# ---- host-aware: KAI0_ROOT (含 kai0/data 的真实根) ----
for cand in \
  /vePFS-North-E/vis_robot/workspace/deepdive_kai0 \
  /data/shared/ubuntu/workspace/deepdive_kai0 \
  /vePFS/tim/workspace/deepdive_kai0; do
  [ -d "$cand/kai0/data" ] && { KAI0_ROOT="$cand"; break; }
done
[ -n "${KAI0_ROOT:-}" ] || { echo "[$(ts)] ERROR: cannot locate deepdive_kai0/kai0/data on this host" >&2; exit 1; }

# tosutil: ~/tosutil 优先, 退回 PATH
if   [ -x "$HOME/tosutil" ];        then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null; then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found" >&2; exit 1; fi

DST="$KAI0_ROOT/kai0/data/Task_AV1"          # 最终数据集路径 (校验/日志用)
DST_PARENT="$KAI0_ROOT/kai0/data"            # ⚠️ tosutil cp 落点 = 父目录: 源末级 Task_AV1 落在此下 → DST (否则双层嵌套 Task_AV1/Task_AV1)
LOG="$KAI0_ROOT/logs/task_av1_sync.log"
mkdir -p "$DST" "$KAI0_ROOT/logs" 2>/dev/null || true

# 防重叠
exec 9>"$LOCK"
flock -n 9 || { echo "[$(ts)] previous run still active, skip" >>"$LOG"; exit 0; }
# 日志轮转 >5MB
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ] && mv "$LOG" "$LOG.old"

# depth 默认排除 (99.7% 对象, 下游不用); V1_INCLUDE_DEPTH=1 才拉
EXC=(-exclude='*depth*')   # 排除 depth (旧 top_head_depth + v4 新 observation.depth.top_head)
[ "${V1_INCLUDE_DEPTH:-0}" = "1" ] && EXC=()

echo "[$(ts)] sync start (tosutil cp -r -u; include_depth=${V1_INCLUDE_DEPTH:-0}) → $DST" >>"$LOG"
out=$("$TOSUTIL" cp -r -u "$SRC/" "$DST_PARENT/" "${EXC[@]}" -j 64 -p 2 2>&1); rc=$?
succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
if [ "$rc" -eq 0 ]; then
  echo "[$(ts)] sync OK succ=${succ:-?} skip=${skip:-?} fail=${fail:-0}" >>"$LOG"
else
  echo "[$(ts)] sync FAIL rc=$rc succ=${succ:-?} skip=${skip:-?} fail=${fail:-?}" >>"$LOG"
  echo "$out" | tail -3 >>"$LOG"
fi
exit $rc
