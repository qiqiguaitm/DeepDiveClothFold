#!/bin/bash
# can_health_snap.sh — 数据采集偶发臂中断的现场快照采集
#
# 用法:
#   bash start_scripts/diag/can_health_snap.sh                # 一次性快照到 stdout
#   bash start_scripts/diag/can_health_snap.sh > /tmp/snap.txt # 保存快照
#   bash start_scripts/diag/can_health_snap.sh --loop 30      # 每 30s 写一次到 /tmp/can_diag/
#
# 触发时机: 出现"某条臂突然不能遥操"时立刻跑;
#           长跑 (--loop) 可在数据采集开始时启动, 自动捕捉中断瞬间的状态.
#
# 输出包含:
#   - 4 条 CAN iface 的 ip -s link (errors, bus-off, dropped)
#   - 各 iface 1s 内 candump 帧数 (期望 ~3500)
#   - 各 ROS topic 的 hz (期望 ~200)
#   - /puppet/arm_status 的 err_code / communication_status
#   - ROS2 节点列表
#   - 最近 200 行 arms 服务日志
#
# 判读 (写在快照末尾, 供事后看):
#   - bus-off / errors 暴增  → H2 bus-off
#   - candump 0 帧 + iface UP → H1 piper_sdk 死了 OR H5 dongle 失联
#   - candump 几千 + ros2 hz=0 → H1 piper_sdk 后台线程死
#   - communication_status_joint_* True → 关节单元异常
#   - ctrl_mode = TEACHING_MODE(0x02) → H4 主臂误触 teach button

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IFACES=(can_left_mas can_left_slave can_right_mas can_right_slave)
LOG_DIR="/tmp/can_diag"

# ── 一次快照 ───────────────────────────────────────────────────────────────
snap_once() {
    local out_prefix="${1:-}"   # 空 = stdout, 否则写到文件 + 同时 echo path
    {
    echo "=============================================="
    echo " CAN health snap @ $(date -Iseconds)"
    echo " host: $(hostname)  uptime: $(uptime -p)"
    echo "=============================================="

    echo ""
    echo "## 1. CAN ifaces (ip -s link)"
    for c in "${IFACES[@]}"; do
        echo ""
        echo "--- $c ---"
        ip -s link show "$c" 2>&1 | sed 's/^/  /'
    done

    echo ""
    echo "## 2. CAN frame rate (1s candump count)"
    for c in "${IFACES[@]}"; do
        cnt=$(timeout 1 candump "$c" 2>/dev/null | wc -l)
        # 期望: ~3500 (双向 200Hz × ~9 帧/cycle); 0 = 总线无声
        marker=""
        [[ "$cnt" -lt 100 ]]  && marker=" [DEAD]"
        [[ "$cnt" -ge 100 && "$cnt" -lt 1000 ]] && marker=" [LOW]"
        printf "  %-22s 1s frames = %5d%s\n" "$c" "$cnt" "$marker"
    done

    echo ""
    echo "## 3. ROS2 topic hz (期望 ~200 Hz)"
    if command -v ros2 >/dev/null 2>&1; then
        for t in /puppet/joint_states /master/joint_states /puppet/arm_status; do
            hz=$(timeout 3 ros2 topic hz "$t" --window 30 2>&1 | grep -oE "average rate: [0-9.]+" | tail -1)
            printf "  %-25s %s\n" "$t" "${hz:-<no data>}"
        done
    else
        echo "  (ros2 not in PATH)"
    fi

    echo ""
    echo "## 4. arm_status err_code / communication_status"
    if command -v ros2 >/dev/null 2>&1; then
        for t in /puppet/arm_status; do
            echo "--- $t ---"
            timeout 2 ros2 topic echo "$t" --once 2>&1 | head -30 | sed 's/^/  /'
        done
    fi

    echo ""
    echo "## 5. ROS2 nodes (期望 4 个 arm_* node 在线)"
    if command -v ros2 >/dev/null 2>&1; then
        timeout 3 ros2 node list 2>&1 | grep -E "arm_|piper" | sed 's/^/  /'
    fi

    echo ""
    echo "## 6. piper_sdk 进程"
    pgrep -af "piper|arm_teleop|arm_master" | head -10 | sed 's/^/  /'

    echo ""
    echo "## 7. lsusb gs_usb dongles (期望 4 个)"
    lsusb | grep -E "Geschwister|gs_usb|1d50:606f" | sed 's/^/  /'

    echo ""
    echo "## 8. arms 服务日志 (最近 200 行)"
    arms_log="$PROJECT_ROOT/web/data_manager/logs/arms.log"
    if [[ -f "$arms_log" ]]; then
        echo "  (source: $arms_log)"
        tail -200 "$arms_log" 2>/dev/null | sed 's/^/    /'
    else
        echo "  (no arms.log at $arms_log)"
    fi

    echo ""
    echo "=============================================="
    echo " 判读速查:"
    echo "   - bus-off / RX errors 暴增  → H2 bus-off (ip link down/up 救)"
    echo "   - candump 0 帧 + iface UP   → H1/H5 (重启节点 / dongle 重枚举)"
    echo "   - candump 几千 + hz=0       → H1 piper_sdk 后台线程死"
    echo "   - communication_status True → 关节单元异常 (硬件 / 线缆)"
    echo "   - ctrl_mode = 0x02          → H4 误触 teach (检查主臂按钮)"
    echo "=============================================="
    } > >(if [[ -n "$out_prefix" ]]; then tee "$out_prefix"; else cat; fi)
}

# ── 自检: 在 snap 文件里找故障 marker ──────────────────────────────────────
# 返回 0 = healthy; 1 = incident detected (输出原因到 stdout)
detect_incident() {
    local snap_file="$1"
    [[ -f "$snap_file" ]] || return 0
    # 1. DEAD/LOW iface (1s frames < 100 → DEAD; < 1000 → LOW)
    if grep -q "\[DEAD\]" "$snap_file"; then
        grep "\[DEAD\]" "$snap_file" | head -3
        return 1
    fi
    # 2. iface state 异常 (BUS-OFF 或 ERROR-PASSIVE)
    if grep -qE "BUSOFF|ERROR-PASSIVE|bus-off" "$snap_file"; then
        grep -E "BUSOFF|ERROR-PASSIVE|bus-off" "$snap_file" | head -3
        return 1
    fi
    # 3. arm_status err_code 非零
    if grep -q "communication_status_joint_.: true" "$snap_file"; then
        grep "communication_status_joint" "$snap_file" | grep "true" | head -3
        return 1
    fi
    # 4. (不在此处检测 ROS hz=0, 因为 --once ros2 echo 拿不到也是 "<no data>",
    #    跟真正 hz=0 不易区分; 留给 [health] log 检测)
    return 0
}

# ── 出事自动打包: 把最近 N 个 snap + ROS log + arms.log tail 拷到 INCIDENT 目录 ──
bundle_incident() {
    local trigger_snap="$1" reason="$2"
    local ts=$(date +%Y%m%d-%H%M%S)
    local inc="$LOG_DIR/INCIDENT_${ts}"
    mkdir -p "$inc"

    # 最近 5 个 snap (含 trigger)
    ls -1t "$LOG_DIR"/snap_*.txt 2>/dev/null | head -5 | while read -r f; do
        cp "$f" "$inc/"
    done

    # 触发原因 + 时间
    cat > "$inc/_INCIDENT.txt" <<EOF
incident @ $(date -Iseconds)
trigger snap: $(basename "$trigger_snap")
reason:
$reason

bundled files: $(ls "$inc/" | wc -l)
EOF

    # arms.log 最近 500 行 (含 [health] grep 出来的精华)
    local arms_log="$PROJECT_ROOT/web/data_manager/logs/arms.log"
    if [[ -f "$arms_log" ]]; then
        tail -500 "$arms_log" > "$inc/arms.log.tail500"
        grep "\[health\]" "$arms_log" 2>/dev/null | tail -100 > "$inc/health_lines.log"
    fi

    # ROS native log 最近 30 min 的 [health] grep
    find "$HOME/.ros/log" -name "*.log" -mmin -30 2>/dev/null \
        | xargs -r grep -h "\[health\]" 2>/dev/null \
        | tail -100 > "$inc/ros_health_lines.log"

    # dmesg 最近 (检查 gs_usb / USB disconnect)
    dmesg --since=-5min 2>/dev/null | grep -iE "gs_usb|usb.*disconnect|usb.*reset" \
        > "$inc/dmesg.tail" 2>/dev/null

    echo ""
    echo "███████████████████████████████████████████████████████████████"
    echo "  INCIDENT @ $ts"
    echo "  reason: $reason"
    echo "  bundle: $inc"
    echo "███████████████████████████████████████████████████████████████"
    echo ""

    # 最后 cooldown: 别在 1 分钟内重复打包 (同一次故障可能持续多个 snap 周期)
    COOLDOWN_UNTIL=$(($(date +%s) + 60))
}

# ── loop 模式 ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--loop" ]]; then
    interval="${2:-30}"
    mkdir -p "$LOG_DIR"
    echo "[loop] writing snapshots every ${interval}s to $LOG_DIR/"
    echo "[loop] auto-incident detect: enabled (DEAD iface / bus-off / err_code)"
    echo "[loop] Ctrl-C to stop"
    COOLDOWN_UNTIL=0
    while true; do
        ts=$(date +%Y%m%d-%H%M%S)
        snap_file="$LOG_DIR/snap_${ts}.txt"
        snap_once "$snap_file" > /dev/null
        if [[ $(date +%s) -ge $COOLDOWN_UNTIL ]]; then
            reason=$(detect_incident "$snap_file")
            if [[ -n "$reason" ]]; then
                bundle_incident "$snap_file" "$reason"
            else
                echo "[$ts] snap → $snap_file (healthy)"
            fi
        else
            echo "[$ts] snap → $snap_file (cooldown)"
        fi
        sleep "$interval"
    done
else
    snap_once ""
fi
