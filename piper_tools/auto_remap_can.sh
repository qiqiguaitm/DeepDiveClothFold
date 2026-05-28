#!/bin/bash
# auto_remap_can.sh — 统一 CAN 接口激活入口, 自动处理 4 种物理变化场景:
#
#   C1 USB 口换位置 (dongle serial 集合不变, 顺序变):
#       → 全自动: activate_can_v2.sh
#
#   C2 dongle 物理换臂绑定 (serial 集合不变, 但实际接的臂换了):
#       → 不可纯软件检测; 只能晃臂查 (verify 流程)
#       本脚本结尾会建议跑 verify_can_mapping.py
#
#   C3 新增/换掉/拔走 dongle (serial 集合变了):
#       → 自动检测 + 拉起 setup_can_v2.sh HITL 校准 (写 dongle_serials.yml) + 激活
#
#   C0 一切不变 (常规开机):
#       → 跟 C1 同, 一行 activate_can_v2.sh 即可
#
# 用法:
#   bash piper_tools/auto_remap_can.sh                # 检测 + 激活
#   bash piper_tools/auto_remap_can.sh --force-hitl   # 跳过 diff, 强制 HITL 重校准
#   bash piper_tools/auto_remap_can.sh --no-hitl      # 检测到 serial 变更也不交互, 直接报错

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
YAML="$PROJECT_ROOT/config/dongle_serials.yml"
BITRATE=1000000

FORCE_HITL=false
NO_HITL=false
while (( $# )); do
    case "$1" in
        --force-hitl) FORCE_HITL=true ;;
        --no-hitl)    NO_HITL=true ;;
        -h|--help)    sed -n '1,26p' "$0"; exit 0 ;;
        *) echo "[FAIL] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

if [[ $(id -u) -eq 0 ]]; then SUDO=""; else SUDO="sudo"; sudo -v; fi

# ── helpers ────────────────────────────────────────────────────────────────
iface_serial() {
    local iface="$1" cur
    cur="$(readlink -f "/sys/class/net/$iface/device" 2>/dev/null)"
    while [[ -n "$cur" && "$cur" != "/" ]]; do
        [[ -f "$cur/serial" ]] && { cat "$cur/serial" 2>/dev/null; return; }
        cur="$(dirname "$cur")"
    done
}

current_serial_set() {
    # Print all serials on current CAN bus, one per line, sorted
    for iface in $(ip -br link show type can 2>/dev/null | awk '{print $1}'); do
        s="$(iface_serial "$iface")"
        [[ -n "$s" ]] && echo "$s"
    done | sort -u
}

yaml_serial_set() {
    [[ -f "$YAML" ]] || return 0
    awk -F: '/^[a-z_]+:[[:space:]]+/{
        gsub(/[[:space:]]|'\''|"/, "", $2); print $2
    }' "$YAML" | sort -u
}

# ── 1. probe current vs YAML ───────────────────────────────────────────────
CUR_SERIALS="$(current_serial_set)"
YAML_SERIALS="$(yaml_serial_set)"
CUR_COUNT="$(echo "$CUR_SERIALS" | grep -c .)"
YAML_COUNT="$(echo "$YAML_SERIALS" | grep -c .)"

echo "=== Probe ==="
echo "  current bus: $CUR_COUNT serial(s)"
echo "  YAML cache:  $YAML_COUNT serial(s)"

if [[ "$CUR_COUNT" -lt 4 ]]; then
    echo "[FAIL] only $CUR_COUNT CAN ifaces on bus; need 4 dongles powered/connected"
    exit 2
fi

if [[ ! -f "$YAML" ]]; then
    echo "  $YAML 不存在 → 需要首次校准"
    NEED_HITL=true
elif $FORCE_HITL; then
    echo "  --force-hitl: 强制重校准"
    NEED_HITL=true
elif [[ "$CUR_SERIALS" == "$YAML_SERIALS" ]]; then
    echo "  ✓ serial set 与 YAML 一致 (C0/C1: 仅 USB 口位置可能变)"
    NEED_HITL=false
else
    echo "  ✗ serial set 变化 (C3: 新增/移除/换 dongle)"
    NEW="$(comm -23 <(echo "$CUR_SERIALS") <(echo "$YAML_SERIALS"))"
    GONE="$(comm -13 <(echo "$CUR_SERIALS") <(echo "$YAML_SERIALS"))"
    [[ -n "$NEW" ]]  && echo "    新出现: " && echo "$NEW"  | sed 's/^/      /'
    [[ -n "$GONE" ]] && echo "    消失:   " && echo "$GONE" | sed 's/^/      /'
    NEED_HITL=true
fi
echo ""

# ── 2. 若需要 HITL: 拉起 setup_can_v2.sh ──────────────────────────────────
if $NEED_HITL; then
    if $NO_HITL; then
        echo "[FAIL] 需要 HITL 校准但 --no-hitl, 退出"
        exit 3
    fi
    if [[ ! -t 0 ]]; then
        echo "[FAIL] 需要 HITL 校准但当前不是交互终端 (stdin 不是 tty)"
        echo "       请在带 tty 的会话中跑此脚本, 或先手动跑 setup_can_v2.sh"
        exit 3
    fi
    echo "=== 进入 HITL 校准 (setup_can_v2) ==="
    echo "将依次提示晃 4 个角色 (左 master / 右 master / 左 slave / 右 slave),"
    echo "脚本检测哪个 CAN iface 在动 → iface→serial → 直接写 YAML."
    echo ""
    read -r -p "按 Enter 开始, Ctrl+C 取消: "

    # setup_can_v2.sh --no-activate: HITL → 写 YAML; auto_remap 后面会统一激活
    if ! bash "$SCRIPT_DIR/setup_can_v2.sh" --no-activate; then
        echo "[FAIL] setup_can_v2.sh 失败"
        exit 4
    fi
    echo ""
fi

# ── 3. 按 (可能已更新的) YAML 激活 ────────────────────────────────────────
echo "=== 按 serial 激活 ==="
if ! bash "$SCRIPT_DIR/activate_can_v2.sh"; then
    echo "[FAIL] activate_can_v2.sh 失败"
    exit 6
fi

# ── 4. 收尾建议 ────────────────────────────────────────────────────────────
echo ""
echo "✓ CAN 激活完成"
if ! $NEED_HITL; then
    echo ""
    echo "提示: 若你刚换过 dongle ↔ 臂的绑定 (C2), 当前激活的符号名可能与实际臂不符."
    echo "      建议跑一次 HITL 校验:  python3 piper_tools/verify_can_mapping.py"
fi
