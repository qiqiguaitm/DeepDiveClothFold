#!/usr/bin/env bash
# Hourly incremental sync: TOS Task_A/dagger -> local vis_dagger, via tosutil cp -u.
#
# 设计 (2026-07-14 修复 chunk-001)：
#   - 自动发现 TOS dagger/ 下所有版本目录（v2 v3 v4 …）。
#   - **逐日期目录 cp -u**（而非 cp -r -u 整版）：tosutil cp -r -u 在 -p 并行模式
#     下对已存在的目录整目录跳过，不检查内部新增子目录（如后添加的 chunk-001）。
#   - 对每个日期目录的 data/ 和 videos/，**显式发现所有 chunk 子目录**，
#     逐 chunk 计数比较（TOS parcel/mp4 数 vs 本地），缺失或数量不等时 cp -u 拉取。
#   - 排除 depth zarr（*depth*）：训练只用 RGB 三路。
#   - **跳过 v3**（SKIP_VERS 默认）：vis_dagger/v3 是本地加工产物。
#
# 运行环境（host-aware：gf0 / gf3 / uc01-03 / sim01），各机需本机 ~/.tosutilconfig。
# cron（每小时）：47 * * * * bash <repo>/train_scripts/kai/data/sync_vis_dagger_from_tos.sh
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done

SRC=tos://transfer-shanghai/KAI0/Task_A/dagger
SKIP_VERS="${VIS_DAGGER_SKIP_VERS:-v3}"
LOCK=/tmp/vis_dagger_sync.lock
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ---- host-aware: KAI0_ROOT + TOSUTIL 自动探测 ----
_VB=kai0/data/Task_A/vis_base/v2
if   [ -d /vePFS-North-E/vis_robot/workspace/deepdive_kai0/$_VB ];  then KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
elif [ -d /data/shared/ubuntu/workspace/deepdive_kai0/$_VB ];      then KAI0_ROOT=/data/shared/ubuntu/workspace/deepdive_kai0
elif [ -d /vePFS/tim/workspace/deepdive_kai0/$_VB ];               then KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0
else echo "[$(ts)] ERROR: cannot locate deepdive_kai0/$_VB on this host" >&2; exit 1; fi
if   [ -x "$HOME/tosutil" ];        then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null; then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found (~/tosutil or PATH)" >&2; exit 1; fi
DST_ROOT="$KAI0_ROOT/kai0/data/Task_A/vis_dagger"
LOG="$KAI0_ROOT/logs/vis_dagger_sync.log"
mkdir -p "$KAI0_ROOT/logs" "$DST_ROOT" 2>/dev/null || true

# 防重叠
exec 9>"$LOCK"
flock -n 9 || { echo "[$(ts)] previous run still active, skip" >>"$LOG"; exit 0; }
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ] && mv "$LOG" "$LOG.old"
[ -x "$TOSUTIL" ] || { echo "[$(ts)] ERROR: tosutil missing at $TOSUTIL" >>"$LOG"; exit 1; }

# 护栏：清理扁平 <date> 残留 + tosutil cp -r 嵌套自重复
for fd in "$DST_ROOT"/2026-*-v*; do
  [ -d "$fd" ] || continue
  echo "[$(ts)] WARN 清理 vis_dagger 根扁平残留：$(basename "$fd")" >>"$LOG"; rm -rf "$fd"
done
for nd in "$DST_ROOT"/v*/2026-*-v*/2026-*-v*; do
  [ -d "$nd" ] || continue
  echo "[$(ts)] WARN 清理嵌套自重复：${nd#$DST_ROOT/}" >>"$LOG"; rm -rf "$nd"
done

# ---- 自动发现 TOS dagger/ 下所有版本目录 ----
versions=$("$TOSUTIL" ls -d "$SRC/" 2>/dev/null | grep -oE '/dagger/v[0-9]+(\.[0-9]+)?/' | grep -oE 'v[0-9]+(\.[0-9]+)?' | sort -u)
if [ -z "$versions" ]; then echo "[$(ts)] ERROR: TOS dagger/ 下未发现版本目录 (cred/network?)" >>"$LOG"; exit 1; fi

EXCLUDE='*depth*'
echo "[$(ts)] sync start; TOS 版本=[$(echo $versions)] skip=[$SKIP_VERS]" >>"$LOG"
total_succ=0; total_skip=0; total_fail=0; total_miss=0

for ver in $versions; do
  case " $SKIP_VERS " in *" $ver "*) echo "[$(ts)] SKIP $ver (本地加工/受保护)" >>"$LOG"; continue ;; esac

  # 逐日期目录 — 防 tosutil cp -r -u 目录级跳过
  # ⚠️ tosutil ls 不带 -d 递归列出所有文件 → 1000 条截断 → 只看到前几个日期.
  #    -d 只列当前层子目录, 21 个日期全部可见.
  dates=$("$TOSUTIL" ls -d "$SRC/$ver/" 2>/dev/null | grep -oE '/20[0-9]{2}-[0-9]{2}-[0-9]{2}-v[0-9]+/' | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}-v[0-9]+' | sort -u)
  if [ -z "$dates" ]; then echo "[$(ts)] WARN $ver: TOS 下无日期目录" >>"$LOG"; continue; fi

  for dt in $dates; do
    tos_d="$SRC/$ver/$dt"
    loc_d="$DST_ROOT/$ver/$dt"
    mkdir -p "$loc_d/data" "$loc_d/videos" 2>/dev/null || true

    # --- data: 发现所有 chunk 子目录，逐 chunk 比较+同步 ---
    tos_chunks=$("$TOSUTIL" ls -d "$tos_d/data/" 2>/dev/null | grep -oE '/chunk-[0-9]+/' | grep -oE 'chunk-[0-9]+' | sort -u)
    for ck in $tos_chunks; do
      tos_ck="$tos_d/data/$ck/"
      loc_ck="$loc_d/data/$ck/"
      tos_n=$("$TOSUTIL" ls "$tos_ck" 2>/dev/null | grep -c '\.parquet$' || echo 0)
      loc_n=$(ls "$loc_ck"/*.parquet 2>/dev/null | wc -l)
      # 完整 → 跳过
      if [ "$loc_n" -gt 0 ] && [ "${tos_n:-0}" -gt 0 ] && [ "$loc_n" -eq "${tos_n:-0}" ]; then
        total_skip=$((total_skip + tos_n))
        continue
      fi
      # 不完整 → cp -r -u (源不带 / → 复制目录自身到 data/, 防扁平化或嵌套)
      mkdir -p "$loc_d/data"
      out=$("$TOSUTIL" cp -r -u "$tos_d/data/$ck" "$loc_d/data/" -exclude="$EXCLUDE" -j 8 -p 2 2>&1); rc=$?
      succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
      skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
      fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
      total_succ=$((total_succ + ${succ:-0})); total_skip=$((total_skip + ${skip:-0})); total_fail=$((total_fail + ${fail:-0}))
      [ "${succ:-0}" -gt 0 ] && echo "[$(ts)]   $ver/$dt/data/$ck: +${succ:-0} pq ($tos_n TOS / $loc_n local)" >>"$LOG"
      [ "${fail:-0}" -gt 0 ] && echo "[$(ts)] FAIL $ver/$dt/data/$ck rc=$rc fail=$fail" >>"$LOG"
    done

    # --- videos: 同样逐 camera → chunk 比较+同步 ---
    tos_cams=$("$TOSUTIL" ls -d "$tos_d/videos/" 2>/dev/null | grep -oE '/observation\.images\.[a-z_]+/' | grep -oE 'observation\.images\.[a-z_]+' | sort -u)
    for cam in $tos_cams; do
      tos_vchunks=$("$TOSUTIL" ls -d "$tos_d/videos/$cam/" 2>/dev/null | grep -oE '/chunk-[0-9]+/' | grep -oE 'chunk-[0-9]+' | sort -u)
      for ck in $tos_vchunks; do
        tos_vck="$tos_d/videos/$cam/$ck/"
        loc_vck="$loc_d/videos/$cam/$ck/"
        tos_vn=$("$TOSUTIL" ls "$tos_vck" 2>/dev/null | grep -c '\.mp4$' || echo 0)
        loc_vn=$(ls "$loc_vck"/*.mp4 2>/dev/null | wc -l)
        if [ "$loc_vn" -gt 0 ] && [ "${tos_vn:-0}" -gt 0 ] && [ "$loc_vn" -eq "${tos_vn:-0}" ]; then
          total_skip=$((total_skip + tos_vn))
          continue
        fi
        mkdir -p "$loc_d/videos"
        out=$("$TOSUTIL" cp -r -u "$tos_d/videos/$cam/$ck" "$loc_d/videos/" -exclude="$EXCLUDE" -j 8 -p 2 2>&1); rc=$?
        succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
        skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
        fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
        total_succ=$((total_succ + ${succ:-0})); total_skip=$((total_skip + ${skip:-0})); total_fail=$((total_fail + ${fail:-0}))
        [ "${succ:-0}" -gt 0 ] && echo "[$(ts)]   $ver/$dt/videos/$cam/$ck: +${succ:-0} mp4 ($tos_vn TOS / $loc_vn local)" >>"$LOG"
      done
    done

    # 缺失检测：TOS 有 chunk 但本地仍无 → 报警
    loc_data_chunks=$(ls -d "$loc_d/data"/chunk-* 2>/dev/null | xargs -n1 basename 2>/dev/null | sort -u)
    for ck in $tos_chunks; do
      if ! echo "$loc_data_chunks" | grep -qx "$ck"; then
        echo "[$(ts)] MISS $ver/$dt/data/$ck：TOS 有但本地无 (cp -u 失败?)" >>"$LOG"
        total_miss=$((total_miss + 1))
      fi
    done
  done
  echo "[$(ts)] OK $ver dates=$(echo "$dates" | wc -w) succ=$total_succ skip=$total_skip fail=$total_fail miss=$total_miss" >>"$LOG"
done

echo "[$(ts)] sync end; total succ=$total_succ skip=$total_skip fail=$total_fail miss=$total_miss" >>"$LOG"
