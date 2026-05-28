#!/bin/bash
# activate_can_v2.sh — 用 USB iSerial 把当前 can 接口重命名为符号名 + bitrate up.
# 不修改 (dongle serial ↔ 臂) 映射, 仅按 YAML 激活. 没改物理链接时, 直接跑这个.
#
# 跟 activate_can.sh (v1) 的区别:
#   v1: 按 USB bus-info 映射重命名, 换 USB 口就错位
#   v2: 按 USB iSerial 映射重命名, USB 口任意插换都对
#
# 前置: config/dongle_serials.yml 已通过 setup_can_v2.sh 写入.
#
# 用法:
#   bash piper_tools/activate_can_v2.sh                 # 标准模式
#   bash piper_tools/activate_can_v2.sh --dry-run       # 只打印计划, 不动接口
#   bash piper_tools/activate_can_v2.sh --bitrate 1000000
#
# 行为:
#   1. 读 config/dongle_serials.yml 拿到 4 个 (iface → serial) 映射
#   2. 扫描所有 'type can' 接口, 沿 sysfs 父链找 USB iSerial
#   3. 按 serial 匹配, 把对应当前接口重命名为符号名
#   4. 全部 set up + bitrate
#
# 物理 USB 口随便插换, 只要 dongle 没物理换臂, 符号名永远一致.

set -uo pipefail   # 注意: 故意不开 -e, rename 单步失败要继续诊断

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
YAML="$PROJECT_ROOT/config/dongle_serials.yml"
BITRATE=1000000
DRY_RUN=false

while (( $# )); do
    case "$1" in
        --bitrate) shift; BITRATE="$1" ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help) sed -n '1,21p' "$0"; exit 0 ;;
        *) echo "[FAIL] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

if [[ $(id -u) -eq 0 ]]; then SUDO=""; else SUDO="sudo"; sudo -v; fi

[[ -f "$YAML" ]] || { echo "[FAIL] missing $YAML — run setup_can_v2.sh first"; exit 1; }

# ── 解析 YAML (单层 'key: value' / "key: 'value'") ─────────────────────────
declare -A WANT_SERIAL
while IFS= read -r line; do
    line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    key="${line%%:*}"
    val="${line#*:}"
    key="${key// }"
    val="$(echo "$val" | sed -E "s/^[[:space:]]*'?//; s/'?[[:space:]]*$//")"
    [[ -n "$key" && -n "$val" ]] && WANT_SERIAL["$key"]="$val"
done < "$YAML"

if [[ "${#WANT_SERIAL[@]}" -ne 4 ]]; then
    echo "[FAIL] expected 4 entries in $YAML, got ${#WANT_SERIAL[@]}"
    exit 1
fi

# ── sysfs walk: 给定 iface, 返回 USB iSerial ─────────────────────────────
iface_serial() {
    local iface="$1" cur
    cur="$(readlink -f "/sys/class/net/$iface/device" 2>/dev/null)"
    while [[ -n "$cur" && "$cur" != "/" ]]; do
        [[ -f "$cur/serial" ]] && { cat "$cur/serial" 2>/dev/null; return; }
        cur="$(dirname "$cur")"
    done
}

# ── 探测当前所有 can 接口 → serial map ────────────────────────────────────
declare -A CUR_BY_SERIAL    # serial → current_iface
declare -a ALL_CAN=()
while read -r iface; do
    [[ -z "$iface" ]] && continue
    ALL_CAN+=("$iface")
    sn="$(iface_serial "$iface")"
    if [[ -n "$sn" ]]; then
        CUR_BY_SERIAL["$sn"]="$iface"
    fi
done < <(ip -br link show type can 2>/dev/null | awk '{print $1}')

echo "=== Discovered ${#ALL_CAN[@]} CAN ifaces ==="
for iface in "${ALL_CAN[@]}"; do
    sn="$(iface_serial "$iface")"
    role="?"
    for want_iface in "${!WANT_SERIAL[@]}"; do
        [[ "${WANT_SERIAL[$want_iface]}" == "$sn" ]] && { role="$want_iface"; break; }
    done
    printf "  %-22s serial=%s  → role=%s\n" "$iface" "${sn:-<none>}" "$role"
done
echo ""

# ── 校验: 4 个期望 serial 是否都在当前总线 ────────────────────────────────
echo "=== Plan ==="
missing=0
declare -a PLAN_FROM=() PLAN_TO=()
for want_iface in "${!WANT_SERIAL[@]}"; do
    want_sn="${WANT_SERIAL[$want_iface]}"
    cur="${CUR_BY_SERIAL[$want_sn]:-}"
    if [[ -z "$cur" ]]; then
        echo "  [MISS] $want_iface (serial=$want_sn) not present on bus"
        missing=$((missing + 1))
        continue
    fi
    if [[ "$cur" == "$want_iface" ]]; then
        echo "  [SKIP] $want_iface already named correctly"
    else
        echo "  [RENAME] $cur → $want_iface"
    fi
    PLAN_FROM+=("$cur")
    PLAN_TO+=("$want_iface")
done
echo ""

if (( missing > 0 )); then
    echo "[WARN] $missing dongle(s) missing — check power / USB connections"
    exit 1
fi

if $DRY_RUN; then
    echo "[dry-run] not modifying any interface"
    exit 0
fi

# ── Step 1: down ALL discovered can ifaces ────────────────────────────────
echo "=== Step 1: down all ==="
for iface in "${ALL_CAN[@]}"; do
    $SUDO ip link set "$iface" down 2>/dev/null
done

# ── Step 2: rename all to unique tmp names (avoid name collisions) ────────
echo "=== Step 2: rename → tmp ==="
declare -A TMP_OF_SERIAL
i=0
for sn in "${!CUR_BY_SERIAL[@]}"; do
    cur="${CUR_BY_SERIAL[$sn]}"
    tmp="can_serialtmp_${i}"
    if ! $SUDO ip link set "$cur" name "$tmp"; then
        echo "  [FAIL] rename $cur → $tmp"
        exit 2
    fi
    TMP_OF_SERIAL["$sn"]="$tmp"
    echo "  $cur → $tmp"
    i=$((i + 1))
done

# ── Step 3: tmp → target symbolic name + bitrate + up ─────────────────────
echo "=== Step 3: tmp → symbolic + bitrate + up ==="
for want_iface in "${!WANT_SERIAL[@]}"; do
    want_sn="${WANT_SERIAL[$want_iface]}"
    tmp="${TMP_OF_SERIAL[$want_sn]:-}"
    if [[ -z "$tmp" ]]; then
        echo "  [FAIL] no tmp iface for serial=$want_sn (want=$want_iface)"
        exit 3
    fi
    if ! $SUDO ip link set "$tmp" name "$want_iface"; then
        echo "  [FAIL] rename $tmp → $want_iface"
        exit 4
    fi
    $SUDO ip link set "$want_iface" type can bitrate "$BITRATE" 2>/dev/null
    $SUDO ip link set "$want_iface" up
    echo "  ✓ $tmp → $want_iface (serial=$want_sn, bitrate=$BITRATE, up)"
done

# ── Final state ───────────────────────────────────────────────────────────
echo ""
echo "=== Final ==="
ip -br link show type can | awk '/^can_/{print "  " $0}'
