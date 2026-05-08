#!/bin/bash
# 按已有映射激活 CAN 接口并重命名为 ROS2 launch 期望的符号名。
#
# 用法:
#   bash can_tools/activate_can.sh              # 激活全部 4 臂 (遥操/DAgger)
#   bash can_tools/activate_can.sh --slave-only # 只激活 2 个 slave 臂 (纯推理)
#
# 映射由 calibrate_can_mapping.py 校准后自动更新。
#
# sim01 bus-info (2026-05-08 calibrate_can_mapping.py 校准):
#   3-2.2.2:1.0 → can_left_mas  (左 master (示教左臂))
#   3-2.2.1:1.0 → can_left_slave  (左 slave (执行左臂))
#   3-2.2.3:1.0 → can_right_mas  (右 master (示教右臂))
#   3-2.2.4:1.0 → can_right_slave  (右 slave (执行右臂))

set -eo pipefail

BITRATE=1000000

if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    sudo -v || { echo "[FAIL] 无法获取 sudo 权限"; exit 1; }
fi

# ── 映射配置: "bus-info:symbolic_name" ───────────────────────────────────────
# calibrate_can_mapping.py 会自动更新这两个数组
SLAVE_MAPPINGS=(
    "3-2.2.1:1.0:can_left_slave"
    "3-2.2.4:1.0:can_right_slave"
)
MASTER_MAPPINGS=(
    "3-2.2.2:1.0:can_left_mas"
    "3-2.2.3:1.0:can_right_mas"
)

# ── 解析参数 ─────────────────────────────────────────────────────────────────
SLAVE_ONLY=false
if [[ "${1:-}" == "--slave-only" ]]; then
    SLAVE_ONLY=true
fi

echo "=============================="
echo "  CAN 臂激活"
echo "  模式: $(if $SLAVE_ONLY; then echo 'slave-only (纯推理)'; else echo '全部 (遥操/DAgger)'; fi)"
echo "=============================="

# ── 构建目标列表 ─────────────────────────────────────────────────────────────
declare -a TARGETS=()
for entry in "${SLAVE_MAPPINGS[@]}"; do
    TARGETS+=("$entry")
done
if ! $SLAVE_ONLY; then
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
