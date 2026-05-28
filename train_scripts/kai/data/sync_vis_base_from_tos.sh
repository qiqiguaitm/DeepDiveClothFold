#!/usr/bin/env bash
# Hourly FULL incremental sync: TOS Task_A/base → local vis_base, via tosutil cp -r -u (逐日期).
#
# 策略 (2026-05-28 改为完整同步):
#   - 完整: 每次都遍历 TOS 上所有 <date>-v2 (不只新日期) → 能接住"旧日期后续追加 episode"的更新。
#   - 增量: tosutil `cp -r -u` 按 size/crc 跳过未变文件 (实测早期日期 760/762 skip, 0.24s), 只拉新/变更, 从不删除本地。
#   - 前提: vis_base 已归一化为 TOS 短名视频目录结构 (2026-05-28), 全量比对不会产生重复。
#   - 不删除: cp 永不删本地多余文件 → 保护 vis_v2_*/A_0423_0527 指向 vis_base 的软链。
# 路径映射: `cp -r .../base/<date>/ <DST>/` → tosutil 把末级 <date> 落在 DST 下 = <DST>/<date>/ (实测).
# 运行环境: 仅 gf0 (本机有 ~/tosutil + ~/.tosutilconfig 凭据); cron 每小时.
# 安装: crontab  0 * * * * /vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data/sync_vis_base_from_tos.sh
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done

TOSUTIL=/home/tim/tosutil
SRC=tos://transfer-shanghai/KAI0/Task_A/base
DST=/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base
LOG=/vePFS/tim/workspace/deepdive_kai0/logs/vis_base_sync.log
LOCK=/tmp/vis_base_sync.lock

ts() { date '+%Y-%m-%d %H:%M:%S'; }

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
