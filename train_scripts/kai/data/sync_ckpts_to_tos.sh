#!/usr/bin/env bash
# Incremental sync: local kai0/checkpoints/ckpt_v* → TOS ckpt/KAI0/, via tosutil cp -r -u.
#
# 设计:
#   - 自动发现 checkpoints/ 下所有 ckpt_v<N> 目录 (ckpt_v0 ckpt_v1 ... 以后 ckpt_v2 自动纳入)。
#   - **上传路径正确 + 无重复路径**: cp -r <本地 ckpt_vN 目录, 无尾斜杠> 到**父前缀** ckpt/KAI0/,
#     tosutil 会把末级目录名 ckpt_vN 追加一次 → 落到 ckpt/KAI0/ckpt_vN/<ckpt>/...。
#     ⚠️ 绝不能写成 dest=ckpt/KAI0/ckpt_vN/ (源+目的都含 ckpt_vN) → 会嵌成 ckpt_vN/ckpt_vN (重复路径!)。
#   - **增量**: cp -u 只传"目的不存在或源更新"的对象; ckpt 目录 append-only(新增 ckpt, 旧的不改)→ 每次只传新增。
#   - **排除 train_state**: 优化器态 (~165G), SFT-init 和部署都不需要 (每个 ckpt 都留有独立 params/)。
#     CKPT_SYNC_EXCLUDE="" 可关闭排除 (连 train_state 一并传)。
#   - **不压缩**: bf16 权重高熵, gzip 实测 0% 压缩率 → tar.gz 纯烧 CPU 无收益, 直传 (tosutil 多线程+分片) 最优。
#   - **只增不删** (additive): 本脚本从不删 TOS 对象; 本地删 ckpt 不会镜像到 TOS (ckpt 删除风险高, 需手动 tosutil rm)。
#   - 每轮结束做**重复路径自检**: 若发现 ckpt_vN/ckpt_vN 嵌套键 → WARN (正常永远不该出现)。
#
# 运行环境: sim01 (ckpt 只在 sim01)。需 ~/.tosutilconfig (cn-shanghai AK/SK)。
# cron (每 30 分钟): */30 * * * * bash <repo>/train_scripts/kai/data/sync_ckpts_to_tos.sh
# 手动/预览: DRY_RUN=1 bash sync_ckpts_to_tos.sh   (只打印将执行的 tosutil 命令, 不真传)
set -uo pipefail
# sim01 交互 shell 的 http(s)_proxy=127.0.0.1:29290 SSH 隧道会让 tosutil 卡死 → 必须 unset
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; do unset "$v"; done

REPO=/data1/tim/workspace/deepdive_kai0
SRC_ROOT="$REPO/kai0/checkpoints"                 # 本地 ckpt 根 (含 ckpt_v0 ckpt_v1 ...)
DST_PARENT=tos://transfer-shanghai/ckpt/KAI0      # 上传父前缀 (末级 ckpt_vN 由 tosutil 追加, 不要在此重复!)
EXCLUDE="${CKPT_SYNC_EXCLUDE-*train_state*}"       # 默认排除 train_state; 置空关闭
DRY_RUN="${DRY_RUN:-0}"
LOCK=/tmp/ckpt_tos_sync.lock
LOG="$REPO/logs/ckpt_tos_sync.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p "$REPO/logs" 2>/dev/null || true

TOSUTIL=""
if   [ -x "$HOME/.local/bin/tosutil" ]; then TOSUTIL="$HOME/.local/bin/tosutil"
elif [ -x "$HOME/tosutil" ];            then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null;     then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found" >&2; exit 1; fi

# 防重叠 (上一轮还在传就跳过)
exec 9>"$LOCK"
flock -n 9 || { echo "[$(ts)] previous run still active, skip" >>"$LOG"; exit 0; }
# 日志轮转 (>5MB)
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ] && mv "$LOG" "$LOG.old"
[ -d "$SRC_ROOT" ] || { echo "[$(ts)] ERROR: SRC_ROOT missing $SRC_ROOT" >>"$LOG"; exit 1; }

# 组装 exclude 参数 (空则不加)
EXARG=(); [ -n "$EXCLUDE" ] && EXARG=(-exclude="$EXCLUDE")

echo "[$(ts)] ==== ckpt sync start; exclude=[${EXCLUDE:-none}] dry_run=$DRY_RUN ====" >>"$LOG"
found=0
for d in "$SRC_ROOT"/ckpt_v*/; do
  [ -d "$d" ] || continue          # 无匹配时 glob 原样返回 → -d 过滤掉
  d="${d%/}"                        # 去尾斜杠 → 源为"目录本身"(无尾斜杠), tosutil 只追加末级名一次
  name="$(basename "$d")"           # ckpt_v0 / ckpt_v1
  found=1
  if [ "$DRY_RUN" = "1" ]; then
    echo "[$(ts)] DRY  $TOSUTIL cp -r -u '$d' '$DST_PARENT/' ${EXARG[*]} -j 32 -p 4  → 落到 $DST_PARENT/$name/" >>"$LOG"
    continue
  fi
  echo "[$(ts)] START $name → $DST_PARENT/$name/" >>"$LOG"
  # dest 必须是父前缀 DST_PARENT/ (末级 / 让 tosutil 把 ckpt_vN 作为子目录), 绝不写 $DST_PARENT/$name/
  out=$("$TOSUTIL" cp -r -u "$d" "$DST_PARENT/" "${EXARG[@]}" -j 32 -p 4 2>&1); rc=$?
  succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
  skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
  fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
  if [ "$rc" -eq 0 ]; then
    echo "[$(ts)] OK    $name succ=${succ:-0} skip=${skip:-0} fail=${fail:-0}" >>"$LOG"
  else
    echo "[$(ts)] FAIL  $name rc=$rc succ=${succ:-?} skip=${skip:-?} fail=${fail:-?}" >>"$LOG"; echo "$out" | tail -3 >>"$LOG"
  fi
done
[ "$found" = 1 ] || echo "[$(ts)] WARN 未发现任何 $SRC_ROOT/ckpt_v* 目录" >>"$LOG"

# ---- 重复路径自检: 正常绝不该出现 ckpt_vN/ckpt_vN 嵌套键 ----
if [ "$DRY_RUN" != "1" ]; then
  dup=$("$TOSUTIL" ls "$DST_PARENT/" 2>/dev/null | grep -oE 'ckpt/KAI0/(ckpt_v[0-9]+)/\1/' | sort -u)
  if [ -n "$dup" ]; then
    echo "[$(ts)] 🚨 WARN 检测到重复路径嵌套 (上传路径写错才会有), 请手动核查/清理:" >>"$LOG"
    echo "$dup" | sed 's/^/    /' >>"$LOG"
  else
    echo "[$(ts)] 自检 OK: 无 ckpt_vN/ckpt_vN 重复路径" >>"$LOG"
  fi
fi
echo "[$(ts)] ==== ckpt sync end ====" >>"$LOG"
