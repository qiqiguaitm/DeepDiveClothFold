#!/usr/bin/env bash
# Hourly FULL incremental sync: TOS Task_A/dagger → local vis_dagger, via tosutil cp -r -u (逐日期).
#
# 与 sync_vis_base_from_tos.sh 同款策略 (2026-06-02 派生):
#   - 完整: 每次遍历 TOS 上所有 dagger/<date>-v2 (不只新日期) → 接住旧日期追加 episode。
#   - 增量: tosutil `cp -r -u` 按 size/crc 跳过未变文件, 只拉新/变更, 从不删除本地。
#   - 默认不删除: cp 永不删本地多余文件 → 保护指向 vis_dagger 的软链 (如有)。
#   - 排除 depth zarr (top_head_depth): 训练只用 RGB 三路, depth 不被下游消费且对象数巨大。
# 与 base 脚本的差异: SRC=dagger / DST=vis_dagger; dagger 暂无 README/analysis, 故不拉那部分。
# 路径映射: `cp -r .../dagger/<date>/ <DST>/` → tosutil 把末级 <date> 落在 DST 下 = <DST>/<date>/.
#
# 运行环境 (host-aware, 支持 gf0 / gf3 / uc01-03 任一机): 按文件系统探测 KAI0 工作根 + tosutil 路径。
#   各机都需本机 ~/.tosutilconfig 凭据 (cn-shanghai AK/SK) + 可写 tosutil 二进制。
# 安装 cron (每机一次, 每小时, 与 base 错峰):
#   crontab -l 2>/dev/null | grep -q sync_vis_dagger || \
#     (crontab -l 2>/dev/null; echo "37 * * * * bash <repo>/train_scripts/kai/data/sync_vis_dagger_from_tos.sh") | crontab -
#
# --mirror 模式 (手动跑, 不进 cron): cp -u 之后再删本地存在但 TOS 已无的日期目录 / orphan episode。
#   安全护栏: TOS 日期清单为空 (网络/凭据故障) 时直接退出, 绝不批量删。
#     bash sync_vis_dagger_from_tos.sh --mirror
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done

MIRROR=0
[ "${1:-}" = "--mirror" ] && MIRROR=1

SRC=tos://transfer-shanghai/KAI0/Task_A/dagger
LOCK=/tmp/vis_dagger_sync.lock
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ---- host-aware: KAI0_ROOT + TOSUTIL 自动探测 ----
#   探测顺序: North-E / data-shared 优先于 /vePFS/tim (gf3 上 /vePFS/tim 是空壳树, 须用真实数据路径校验)。
_VB=kai0/data/Task_A/vis_base   # 用 vis_base 存在性判定真实 KAI0 根 (各机都有)
if   [ -d /vePFS-North-E/vis_robot/workspace/deepdive_kai0/$_VB ];  then
  KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
elif [ -d /data/shared/ubuntu/workspace/deepdive_kai0/$_VB ];      then
  KAI0_ROOT=/data/shared/ubuntu/workspace/deepdive_kai0
elif [ -d /vePFS/tim/workspace/deepdive_kai0/$_VB ];               then
  KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0
else
  echo "[$(ts)] ERROR: cannot locate deepdive_kai0/$_VB on this host" >&2; exit 1
fi
if   [ -x "$HOME/tosutil" ];        then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null; then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found (~/tosutil or PATH)" >&2; exit 1
fi
DST="$KAI0_ROOT/kai0/data/Task_A/vis_dagger"
LOG="$KAI0_ROOT/logs/vis_dagger_sync.log"
mkdir -p "$KAI0_ROOT/logs" "$DST" 2>/dev/null || true

# 防重叠
exec 9>"$LOCK"
flock -n 9 || { echo "[$(ts)] previous run still active, skip" >>"$LOG"; exit 0; }

# 日志轮转 (>5MB)
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ] && mv "$LOG" "$LOG.old"

[ -x "$TOSUTIL" ] || { echo "[$(ts)] ERROR: tosutil missing at $TOSUTIL" >>"$LOG"; exit 1; }
[ -d "$DST" ]     || { echo "[$(ts)] ERROR: local DST missing $DST" >>"$LOG"; exit 1; }

echo "[$(ts)] sync start (tosutil cp -r -u, full incremental)" >>"$LOG"
dates=$("$TOSUTIL" ls -d "$SRC/" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}-v[0-9]+/?$' | sed 's#/$##' | sort -u)
[ -n "$dates" ] || { echo "[$(ts)] ERROR: no dates from TOS (cred/network?)" >>"$LOG"; exit 1; }

EXCLUDE='*top_head_depth*'
total=$(echo "$dates" | wc -w); ok=0; pulled=0
for d in $dates; do
  out=$("$TOSUTIL" cp -r -u "$SRC/$d/" "$DST/" -exclude="$EXCLUDE" -j 3 -p 8 2>&1); rc=$?
  succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
  skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
  fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
  if [ "$rc" -eq 0 ]; then
    ok=$((ok+1))
    [ "${succ:-0}" -gt "${skip:-0}" ] 2>/dev/null && { pulled=$((pulled+1)); echo "[$(ts)] PULL $d succ=$succ skip=$skip" >>"$LOG"; }
  else
    echo "[$(ts)] FAIL $d rc=$rc succ=$succ skip=$skip fail=$fail" >>"$LOG"
    echo "$out" | tail -3 >>"$LOG"
  fi
done
echo "[$(ts)] sync end: $ok/$total dates ok, $pulled with new/changed objects" >>"$LOG"

# ---- --mirror: 传播 TOS 端删除 (重编号 / 中段删 episode) ----
if [ "$MIRROR" -eq 1 ]; then
  echo "[$(ts)] MIRROR start (propagate TOS deletions)" >>"$LOG"
  tos_set=" $dates "
  for ld in "$DST"/*-v2/; do
    [ -d "$ld" ] || continue
    ldd=$(basename "$ld")
    case "$tos_set" in
      *" $ldd "*) : ;;
      *) echo "[$(ts)] MIRROR rm date-dir not on TOS: $ldd ($(du -sh "$ld" 2>/dev/null|cut -f1))" >>"$LOG"; rm -rf "$ld" ;;
    esac
  done
  for d in $dates; do
    [ -d "$DST/$d/data/chunk-000" ] || continue
    tos_eps=$("$TOSUTIL" ls "$SRC/$d/data/chunk-000/" 2>/dev/null | grep -oE 'episode_[0-9]+\.parquet' | sort -u)
    [ -n "$tos_eps" ] || { echo "[$(ts)] MIRROR skip $d (TOS list empty, guard)" >>"$LOG"; continue; }
    for lp in "$DST/$d/data/chunk-000/"episode_*.parquet; do
      [ -e "$lp" ] || continue
      ep=$(basename "$lp")
      if ! grep -qxF "$ep" <<<"$tos_eps"; then
        epn="${ep%.parquet}"
        rm -f "$lp" "$DST/$d/videos/chunk-000/"*/"$epn.mp4"
        echo "[$(ts)] MIRROR rm orphan $d/$ep (+videos)" >>"$LOG"
      fi
    done
  done
  echo "[$(ts)] MIRROR end" >>"$LOG"
fi
