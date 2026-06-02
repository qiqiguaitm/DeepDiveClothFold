#!/usr/bin/env bash
# Hourly FULL incremental sync: TOS Task_A/base → local vis_base, via tosutil cp -r -u (逐日期).
#
# 策略 (2026-05-28 改为完整同步):
#   - 完整: 每次都遍历 TOS 上所有 <date>-v2 (不只新日期) → 能接住"旧日期后续追加 episode"的更新。
#   - 增量: tosutil `cp -r -u` 按 size/crc 跳过未变文件 (实测早期日期 760/762 skip, 0.24s), 只拉新/变更, 从不删除本地。
#   - 前提: vis_base 已归一化为 TOS 短名视频目录结构 (2026-05-28), 全量比对不会产生重复。
#   - 默认不删除: cp 永不删本地多余文件 → 保护指向 vis_base 的软链 (如有)。
# 路径映射: `cp -r .../base/<date>/ <DST>/` → tosutil 把末级 <date> 落在 DST 下 = <DST>/<date>/ (实测).
#
# 运行环境 (2026-06-02 改为 host-aware, 支持 gf0 / gf3 / uc01-03 任一机直接跑):
#   按 hostname / 文件系统探测自动选 KAI0 工作根 + tosutil 路径, 无需逐机改脚本。
#   各机都需本机 ~/.tosutilconfig 凭据 (cn-shanghai AK/SK) + 可写 tosutil 二进制。
# 安装 cron (每机一次, 每小时):
#   crontab -l 2>/dev/null | grep -q sync_vis_base || \
#     (crontab -l 2>/dev/null; echo "17 * * * * bash <repo>/train_scripts/kai/data/sync_vis_base_from_tos.sh") | crontab -
#
# README/analysis 同步 (2026-06-02 加): 每轮额外拉 base/README.md + base/analysis/ 作为数据描述参考
#   (per-date 场景表 / 质量评估 / Class C 黑名单 / end-snap 清单 等)。
#
# --mirror 模式 (2026-06-02 加, 手动跑, 不进 cron):
#   普通 cp -u 只增不删, 无法传播 TOS 端的"日期重编号 / 中段删 episode"类迁移 (如 2026-06-01 的
#   5-18→5-10 / 6-1→5-18 重编号 + 04-23/04-25 中段删 ep)。--mirror 在 cp -u 之后再:
#     (a) 删本地存在但 TOS 已无的 <date>-v2 整目录;
#     (b) 删各日期内本地存在但 TOS meta 已不引用的 orphan episode (parquet + 3 视频)。
#   安全护栏: TOS 日期清单为空 (网络/凭据故障) 时直接退出, 绝不批量删。建议迁移后手动:
#     bash sync_vis_base_from_tos.sh --mirror
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done

MIRROR=0
[ "${1:-}" = "--mirror" ] && MIRROR=1

SRC=tos://transfer-shanghai/KAI0/Task_A/base
LOCK=/tmp/vis_base_sync.lock
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ---- host-aware: KAI0_ROOT + TOSUTIL 自动探测 ----
#   gf0 (cnsh):  /vePFS/tim/workspace/deepdive_kai0          + /home/tim/tosutil
#   gf3 (cnbj):  /vePFS-North-E/vis_robot/workspace/deepdive_kai0 + tosutil(PATH, /usr/bin)
#   uc01-03:     /data/shared/ubuntu/workspace/deepdive_kai0 + /home/ubuntu/tosutil
if   [ -d /vePFS/tim/workspace/deepdive_kai0 ];                       then
  KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0
elif [ -d /vePFS-North-E/vis_robot/workspace/deepdive_kai0 ];        then
  KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
elif [ -d /data/shared/ubuntu/workspace/deepdive_kai0 ];             then
  KAI0_ROOT=/data/shared/ubuntu/workspace/deepdive_kai0
else
  echo "[$(ts)] ERROR: cannot locate deepdive_kai0 root on this host" >&2; exit 1
fi
# tosutil: 优先 ~/tosutil, 退回 PATH 中的 tosutil
if   [ -x "$HOME/tosutil" ];        then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null; then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found (~/tosutil or PATH)" >&2; exit 1
fi
DST="$KAI0_ROOT/kai0/data/Task_A/vis_base"
LOG="$KAI0_ROOT/logs/vis_base_sync.log"
mkdir -p "$KAI0_ROOT/logs" 2>/dev/null || true

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

# 排除 depth zarr (top_head_depth, 单日期 ~18.5 万小文件 × 13 日期 ≈ 240 万对象):
#   - vis_v2_* 训练只用 RGB (top_head/hand_left/hand_right), depth 当前不被下游消费;
#   - 含 depth 则每轮要比对 240 万对象 (20-30 min), 排除后仅几万 (秒级), 才适合每小时跑;
#   - 本地已有的 depth 文件不会被删 (cp 不删), 只是不再逐轮比对/更新。
#   - 若将来需要 depth, 单独手动 `tosutil cp -r -u .../base/<date>/videos/chunk-*/top_head_depth/ ...` 或改成低频(每日)同步。
EXCLUDE='*top_head_depth*'
total=$(echo "$dates" | wc -w); ok=0; pulled=0
for d in $dates; do
  out=$("$TOSUTIL" cp -r -u "$SRC/$d/" "$DST/" -exclude="$EXCLUDE" -j 3 -p 8 2>&1); rc=$?
  succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
  skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
  fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
  if [ "$rc" -eq 0 ]; then
    ok=$((ok+1))
    # succ>skip 说明确有新/变更对象被拉
    [ "${succ:-0}" -gt "${skip:-0}" ] 2>/dev/null && { pulled=$((pulled+1)); echo "[$(ts)] PULL $d succ=$succ skip=$skip" >>"$LOG"; }
  else
    echo "[$(ts)] FAIL $d rc=$rc succ=$succ skip=$skip fail=$fail" >>"$LOG"
    echo "$out" | tail -3 >>"$LOG"
  fi
done
echo "[$(ts)] sync end: $ok/$total dates ok, $pulled with new/changed objects" >>"$LOG"

# ---- README + analysis (数据描述参考, 2026-06-02) ----
# 注意 cp -r SRC/analysis/ DST/ → DST/analysis/ (末级落 DST 下, 不嵌套); README 单文件直接 -f 覆盖.
"$TOSUTIL" cp "$SRC/README.md" "$DST/README.md" -f >/dev/null 2>&1 \
  && echo "[$(ts)] README.md synced" >>"$LOG" \
  || echo "[$(ts)] WARN README.md sync failed" >>"$LOG"
"$TOSUTIL" cp -r -u "$SRC/analysis/" "$DST/" >/dev/null 2>&1 \
  && echo "[$(ts)] analysis/ synced" >>"$LOG" \
  || echo "[$(ts)] WARN analysis/ sync failed" >>"$LOG"

# ---- --mirror: 传播 TOS 端删除 (重编号 / 中段删 episode) ----
if [ "$MIRROR" -eq 1 ]; then
  echo "[$(ts)] MIRROR start (propagate TOS deletions)" >>"$LOG"
  # (a) 删本地存在但 TOS 已无的日期目录
  tos_set=" $dates "
  for ld in "$DST"/*-v2/; do
    [ -d "$ld" ] || continue
    ldd=$(basename "$ld")
    case "$tos_set" in
      *" $ldd "*) : ;;  # TOS 有, 保留
      *) echo "[$(ts)] MIRROR rm date-dir not on TOS: $ldd ($(du -sh "$ld" 2>/dev/null|cut -f1))" >>"$LOG"; rm -rf "$ld" ;;
    esac
  done
  # (b) 删各日期内 TOS 已不引用的 orphan episode (parquet + 3 视频)
  for d in $dates; do
    [ -d "$DST/$d/data/chunk-000" ] || continue
    tos_eps=$("$TOSUTIL" ls "$SRC/$d/data/chunk-000/" 2>/dev/null | grep -oE 'episode_[0-9]+\.parquet' | sort -u)
    [ -n "$tos_eps" ] || { echo "[$(ts)] MIRROR skip $d (TOS list empty, guard)" >>"$LOG"; continue; }
    for lp in "$DST/$d/data/chunk-000/"episode_*.parquet; do
      [ -e "$lp" ] || continue
      ep=$(basename "$lp")
      if ! grep -qxF "$ep" <<<"$tos_eps"; then
        epn="${ep%.parquet}"   # episode_NNNNNN
        rm -f "$lp" "$DST/$d/videos/chunk-000/"*/"$epn.mp4"
        echo "[$(ts)] MIRROR rm orphan $d/$ep (+videos)" >>"$LOG"
      fi
    done
  done
  echo "[$(ts)] MIRROR end" >>"$LOG"
fi
