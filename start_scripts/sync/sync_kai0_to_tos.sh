#!/bin/bash
# sync_kai0_to_tos.sh — sim01 → TOS 增量同步 KAI0/ 全部内容
#
# 设计:
#   - 同步 3 类对象:
#     (1) KAI0/ 根级文件 (*.py 等)
#     (2) Task_*/<subset>/ 下的 immediate 文件 (README.md 等)
#     (3) Task_*/<subset>/ 下的 immediate 子目录 (含 *-v2, kai0_official_base, analysis,
#         及任意 legacy 旧目录, 递归同步)
#   - tosutil cp -r -flat -u, -u 让已有文件按 size+mtime skip
#   - flock -n 防止两次 timer 触发并行执行 (前一次没跑完直接跳过)
#   - 永不删除 TOS 上文件 (append-only 安全, 单向 sim01→TOS)
#
# 由 systemd user timer 每小时触发: ~/.config/systemd/user/kai0-tos-sync.timer
#
# 手动跑:  bash start_scripts/sync/sync_kai0_to_tos.sh
# 日志:    /data1/tim/.tos_fix/sync_logs/

set -uo pipefail

LOCAL_ROOT="/data1/DATA_IMP/KAI0"
TOS_ROOT="tos://transfer-shanghai/KAI0"
LOG_DIR="/data1/tim/.tos_fix/sync_logs"
LOCK="/tmp/kai0_tos_sync.lock"
TOSUTIL="/home/tim/.local/bin/tosutil"

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/sync_${TS}.log"

# flock: 拿不到锁就立刻退 (前一次还在跑)
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date -Iseconds)] another sync still running, skip this hour" \
        >> "$LOG_DIR/skipped.log"
    exit 0
fi

# 进入正常流程后所有输出落 LOG
exec > "$LOG" 2>&1

echo "============================================"
echo " KAI0 → TOS sync @ $(date -Iseconds)"
echo "============================================"

# 工具自检
[[ -x "$TOSUTIL" ]] || { echo "[FAIL] $TOSUTIL not executable"; exit 1; }
[[ -d "$LOCAL_ROOT" ]] || { echo "[FAIL] $LOCAL_ROOT not found"; exit 1; }

# 滚动日志: 只留最近 50 个
ls -1t "$LOG_DIR"/sync_*.log 2>/dev/null | tail -n +51 | xargs -r rm -f

# 收集 (kind, src, dst) 三元组; kind=DIR 表示递归目录 cp, kind=FILE 表示单文件 cp.
# 三元组用 TAB 分隔, 行间用 NL; 后续用 xargs -0 -n3 -P 4 并行 4 路.

pairs_file=$(mktemp /tmp/kai0_sync_pairs.XXXXXX)
trap 'rm -f "$pairs_file"' EXIT

# (1) KAI0/ 根级文件
for f in "$LOCAL_ROOT"/*; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f")
    printf "FILE\t%s\t%s\n" "$f" "${TOS_ROOT}/${name}" >> "$pairs_file"
done

# (2) Task_*/<subset>/ 下任意 immediate 子目录 + 根级文件
for task_dir in "$LOCAL_ROOT"/Task_*/; do
    [[ -d "$task_dir" ]] || continue
    task=$(basename "$task_dir")
    for subset_dir in "$task_dir"*/; do
        [[ -d "$subset_dir" ]] || continue
        subset=$(basename "$subset_dir")
        for entry in "$subset_dir"*; do
            [[ -e "$entry" ]] || continue
            name=$(basename "$entry")
            # 跳过 hidden 与 .tmp
            [[ "$name" == .* || "$name" == *.tmp ]] && continue
            if [[ -d "$entry" ]]; then
                printf "DIR\t%s\t%s\n" "$entry/" "${TOS_ROOT}/${task}/${subset}/${name}/" >> "$pairs_file"
            elif [[ -f "$entry" ]]; then
                printf "FILE\t%s\t%s\n" "$entry" "${TOS_ROOT}/${task}/${subset}/${name}" >> "$pairs_file"
            fi
        done
    done
done

n_pairs=$(wc -l < "$pairs_file")
n_dir=$(grep -c $'^DIR\t' "$pairs_file" || true)
n_file=$(grep -c $'^FILE\t' "$pairs_file" || true)
echo "to sync: $n_pairs items ($n_dir dirs + $n_file files, 4-way parallel)"

# 单 entry 同步函数 (并发 worker)
sync_pair() {
    local kind="$1" src="$2" dst="$3"
    echo ""
    echo "--- $(date -Iseconds) [$kind] ${src#$LOCAL_ROOT/} ---"
    if [[ "$kind" == "DIR" ]]; then
        "$TOSUTIL" cp -r -flat -u -j 10 -p 50 "$src" "$dst" 2>&1 | tail -3
    else
        "$TOSUTIL" cp -u "$src" "$dst" 2>&1 | tail -2
    fi
}
export -f sync_pair
export LOCAL_ROOT TOSUTIL

# xargs -P 4: 同时跑 4 个 sync_pair. 把 (kind TAB src TAB dst) 转成 NUL 流, -n3 取 3 个.
< "$pairs_file" tr '\t\n' '\0\0' | xargs -0 -n3 -P 4 bash -c 'sync_pair "$1" "$2" "$3"' _

echo ""
echo "============================================"
echo " sync done @ $(date -Iseconds)  ($n_pairs items: $n_dir dirs + $n_file files)"
echo "============================================"
exit 0
