#!/bin/bash
# setup_can_v2.sh — HITL 重新识别 4 个 dongle ↔ 臂的映射关系.
#
# 跟 setup_can.sh (v1) 的区别:
#   v1: USB bus-info 映射, 换 USB 口就错位 → 写 activate_can.sh 静态表
#   v2: USB iSerial 映射, dongle 在哪个 USB 口都对 → 写 config/dongle_serials.yml
#
# 适用场景 (任一就跑此脚本):
#   - 首次部署
#   - 换/加/减 dongle
#   - 调换 dongle ↔ 臂 的物理绑定
#   - 怀疑映射错位 (verify_can_mapping.py 抽查不过)
#
# 用法:
#   bash piper_tools/setup_can_v2.sh                  # HITL → 写 YAML → 激活
#   bash piper_tools/setup_can_v2.sh --no-activate    # 只写 YAML, 不激活
#
# 校准后日常使用 activate_can_v2.sh 不再需要 HITL.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ACTIVATE_FLAG="--activate"
for arg in "$@"; do
    case "$arg" in
        --no-activate) ACTIVATE_FLAG="" ;;
        -h|--help) sed -n '1,21p' "$0"; exit 0 ;;
        *) echo "[FAIL] 未知参数: $arg"; exit 1 ;;
    esac
done

echo "=============================="
echo "  setup_can_v2: HITL 重新识别"
echo "=============================="
echo ""
echo "流程: 依次晃 4 个角色 → 检测 iface → iface→serial → 写 YAML${ACTIVATE_FLAG:+ → 激活}"
echo ""

exec python3 "$SCRIPT_DIR/calibrate_serial_hitl.py" $ACTIVATE_FLAG
