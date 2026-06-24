#!/bin/bash
# 按已有映射激活 CAN 接口并重命名为 ROS2 launch 期望的符号名。
#
# 用法:
#   bash piper_tools/activate_can.sh              # 按机器 profile 激活
#   bash piper_tools/activate_can.sh --machine visrobot02
#   bash piper_tools/activate_can.sh --robot visrobot02    # --machine 的兼容别名
#   bash piper_tools/activate_can.sh --two-can    # visrobot02: 左/右各一条共享 CAN
#   bash piper_tools/activate_can.sh --four-can   # visrobot01: 原始 4-CAN
#   bash piper_tools/activate_can.sh --slave-only # 只激活 slave 臂 (纯推理)
#
# 映射由 calibrate_can_mapping.py 校准后自动更新。
#
# visrobot02: 左右各一条共享 CAN; visrobot01: 原始 4-CAN.

set -eo pipefail

BITRATE=1000000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_SH="$PROJECT_ROOT/config/robot_profiles.sh"
if [[ -f "$PROFILE_SH" ]]; then
    # shellcheck disable=SC1090
    source "$PROFILE_SH"
fi

if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    if sudo -n true >/dev/null 2>&1; then
        :
    elif [[ -n "${SUDO_PASSWORD:-}" || -r "$HOME/.sudo_password" ]]; then
        _ASKPASS_SCRIPT="/tmp/.kai0_can_askpass_$$.sh"
        cleanup_askpass() { rm -f "$_ASKPASS_SCRIPT" 2>/dev/null; }
        trap cleanup_askpass EXIT
        if [[ -n "${SUDO_PASSWORD:-}" ]]; then
            _PW="$SUDO_PASSWORD"
        else
            _PW="$(head -n1 "$HOME/.sudo_password")"
        fi
        cat > "$_ASKPASS_SCRIPT" <<EOF
#!/bin/sh
printf '%s\n' '$_PW'
EOF
        chmod 700 "$_ASKPASS_SCRIPT"
        unset _PW
        export SUDO_ASKPASS="$_ASKPASS_SCRIPT"
        sudo -A -v || { echo "[FAIL] 无法获取 sudo 权限"; exit 1; }
        SUDO="sudo -A"
    else
        sudo -v || { echo "[FAIL] 无法获取 sudo 权限"; exit 1; }
    fi
fi

# ── 映射配置: "bus-info:symbolic_name" ───────────────────────────────────────
# calibrate_can_mapping.py 会自动更新这两个数组
SLAVE_MAPPINGS=(
    "${KAI0_LEFT_SLAVE_BUS_INFO:-1-1:1.0}:can_left_slave"
    "${KAI0_RIGHT_SLAVE_BUS_INFO:-}:can_right_slave"
)
MASTER_MAPPINGS=(
    "${KAI0_LEFT_MASTER_BUS_INFO:-1-13:1.0}:can_left_mas"
    "${KAI0_RIGHT_MASTER_BUS_INFO:-1-12:1.0}:can_right_mas"
)
TWO_CAN_MAPPINGS=(
    "${KAI0_LEFT_SHARED_BUS_INFO:-${KAI0_LEFT_MASTER_BUS_INFO:-1-13:1.0}}:can_left_mas"
    "${KAI0_RIGHT_SHARED_BUS_INFO:-${KAI0_RIGHT_MASTER_BUS_INFO:-1-12:1.0}}:can_right_mas"
)

# ── 解析参数 ─────────────────────────────────────────────────────────────────
SLAVE_ONLY=false
TWO_CAN=false
FOUR_CAN=false
MACHINE="${VIS_ROBOT_ID:-${KAI0_ROBOT_ID:-${KAI0_MACHINE_ID:-}}}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slave-only) SLAVE_ONLY=true; shift ;;
        --two-can) TWO_CAN=true; shift ;;
        --four-can) FOUR_CAN=true; shift ;;
        --machine|--robot) MACHINE="${2:-}"; shift 2 ;;
        -h|--help) sed -n '1,13p' "$0"; exit 0 ;;
        *) echo "[FAIL] 未知参数: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$MACHINE" ]]; then
    export KAI0_MACHINE_ID="$MACHINE"
    export VIS_ROBOT_ID="${VIS_ROBOT_ID:-$MACHINE}"
    export KAI0_ROBOT_ID="${KAI0_ROBOT_ID:-$MACHINE}"
fi
if declare -F apply_kai0_robot_profile >/dev/null; then
    apply_kai0_robot_profile teleop
fi
if ! $TWO_CAN && ! $FOUR_CAN && ! $SLAVE_ONLY; then
    case "${KAI0_CAN_TOPOLOGY:-${KAI0_DEFAULT_CAN_TOPOLOGY:-auto}}" in
        2can|two-can) TWO_CAN=true ;;
        4can|four-can) FOUR_CAN=true ;;
        auto)
            if [[ "$(ip -br link show type can 2>/dev/null | wc -l)" -lt 4 ]]; then
                TWO_CAN=true
            else
                FOUR_CAN=true
            fi
            ;;
    esac
fi
if $TWO_CAN && $FOUR_CAN; then
    echo "[FAIL] --two-can 与 --four-can 不能同时使用" >&2
    exit 1
fi

echo "=============================="
echo "  CAN 臂激活"
echo "  机器: ${KAI0_MACHINE_ID:-unknown} (${KAI0_ROBOT_PROFILE:-auto})"
echo "  模式: $(if $TWO_CAN; then echo 'two-can (visrobot02 左/右共享 CAN)'; elif $SLAVE_ONLY; then echo 'slave-only (纯推理)'; else echo 'four-can (visrobot01 原始拓扑)'; fi)"
echo "=============================="

# ── 构建目标列表 ─────────────────────────────────────────────────────────────
declare -a TARGETS=()
if $TWO_CAN; then
    for entry in "${TWO_CAN_MAPPINGS[@]}"; do
        TARGETS+=("$entry")
    done
else
    for entry in "${SLAVE_MAPPINGS[@]}"; do
        TARGETS+=("$entry")
    done
fi
if ! $SLAVE_ONLY && ! $TWO_CAN; then
    for entry in "${MASTER_MAPPINGS[@]}"; do
        TARGETS+=("$entry")
    done
fi

# ── Step 1: 全部 down + 重命名为临时名 (避免名称冲突) ────────────────────────
echo ""
echo "--- Step 1: 重命名为临时名 ---"

for iface in $(ip -br link show type can | awk '{print $1}'); do
    $SUDO ip link set "$iface" down 2>/dev/null || true
done

# 建立 bus-info → 当前接口名
declare -A BUS_TO_IFACE=()
for iface in $(ip -br link show type can | awk '{print $1}'); do
    bus=$($SUDO ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//')
    if [[ -n "$bus" ]]; then
        BUS_TO_IFACE["$bus"]="$iface"
    fi
done

# 重命名为临时名
TMP_IDX=0
declare -A BUS_TO_TMP=()
for entry in "${TARGETS[@]}"; do
    symbolic="${entry##*:}"
    bus_info="${entry%:*}"

    if [[ -z "$bus_info" ]]; then
        echo "  [SKIP] $symbolic 未配置 bus-info"
        continue
    fi

    current="${BUS_TO_IFACE[$bus_info]:-}"
    if [[ -z "$current" ]]; then
        echo "  [SKIP] 未找到 bus-info=$bus_info ($symbolic)"
        continue
    fi

    tmp_name="can_tmp_${TMP_IDX}"
    if [[ "$current" != "$tmp_name" ]]; then
        $SUDO ip link set "$current" name "$tmp_name" 2>/dev/null || true
        echo "  $current → $tmp_name (bus: $bus_info)"
    fi
    BUS_TO_TMP["$bus_info"]="$tmp_name"
    TMP_IDX=$((TMP_IDX + 1))
done

# ── Step 2: 临时名 → 符号名 + 激活 ──────────────────────────────────────────
echo ""
echo "--- Step 2: 重命名为符号名并激活 ---"

for entry in "${TARGETS[@]}"; do
    symbolic="${entry##*:}"
    bus_info="${entry%:*}"

    if [[ -z "$bus_info" ]]; then
        continue
    fi

    tmp_name="${BUS_TO_TMP[$bus_info]:-}"
    if [[ -z "$tmp_name" ]]; then
        continue
    fi

    $SUDO ip link set "$tmp_name" type can bitrate "$BITRATE" 2>/dev/null || true
    if [[ "$tmp_name" != "$symbolic" ]]; then
        $SUDO ip link set "$tmp_name" name "$symbolic"
    fi
    $SUDO ip link set "$symbolic" up
    echo "  [OK] $tmp_name → $symbolic (bus: $bus_info)"
done

# ── 结果 ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== 激活结果 ==="
ip -br link show type can
echo ""

echo "=== 数据流校验 ==="
for iface in $(ip -br link show type can | awk '{print $1}'); do
    link_flags=$(ip link show "$iface" | head -1)
    if echo "$link_flags" | grep -q ",UP"; then
        result=$(timeout 1 candump "$iface" -n 1 2>&1 || true)
        if [[ -n "$result" ]]; then
            echo "  $iface: OK (有数据)"
        else
            echo "  $iface: UP 但无数据 (臂未上电?)"
        fi
    fi
done
