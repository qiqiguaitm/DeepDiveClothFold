#!/bin/bash
# 纯遥操启动 (不录制 / 不开相机) — 给手眼标定用
#
# 与 start_data_collect.sh 的区别:
#   - 只起 4 个 arm 节点 (master_left/right + slave_left/right)
#   - 不起 launch_3cam (相机由 capture_handeye.py 自己 pyrealsense2 直连占用)
#   - 不起 web backend/frontend, 不起 pedal 监听
#
# 用法:
#   bash start_scripts/start_teleop_only.sh
#
# 流程:
#   1. 检查并清理可能在跑的 RealSense node (避免它们抢相机)
#   2. 激活 CAN (需 sudo, 若已 UP 则跳过)
#   3. source ROS2 jazzy + ros2_ws
#   4. 前台启动 teleop_launch.py (Ctrl+C 退出)
#
# 退出后相机会被释放给 calib/capture_handeye.py 使用.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
CAN_ACTIVATE="$PROJECT_ROOT/piper_tools/activate_can.sh"
ROS_DISTRO="jazzy"

echo "============================================"
echo "  kai0 纯遥操 (不占相机, 为手眼标定准备)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── 0. 清理可能占用相机的 ros2 节点 ──────────────────────────────────────
# 历史背景: start_data_collect.sh 启动后会留 realsense2_camera_node 孤儿,
# 即使 stop 了有时也清不干净。本脚本启动前主动清一次,
# 保证 capture_handeye.py 退出本脚本后能拿到相机。
echo "[0/3] 检查相机占用..."
REALSENSE_PIDS=$(pgrep -f realsense2_camera_node || true)
if [[ -n "$REALSENSE_PIDS" ]]; then
    echo "  发现 realsense2_camera_node 运行中 (PID: $REALSENSE_PIDS)"
    echo "  这会让 capture_handeye.py 拿不到相机。是否杀掉这些进程?"
    read -r -p "  [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        pkill -TERM -f realsense2_camera_node 2>/dev/null || true
        sleep 1
        pkill -KILL -f realsense2_camera_node 2>/dev/null || true
        echo "  已清理"
    else
        echo "  保留 — 后续标定脚本可能会报 'failed to open device'"
    fi
else
    echo "  相机未被占用"
fi
# 同时清掉可能的 ros2 launch 孤儿 (它们持有 arm 节点会跟我们的 launch 冲突)
pkill -f "ros2 launch piper teleop_launch" 2>/dev/null || true
sleep 0.3
echo ""

# ── 1. 激活 CAN ──────────────────────────────────────────────────────────
echo "[1/3] 激活 CAN 接口..."
EXPECTED_IFACES="can_left_mas can_left_slave can_right_mas can_right_slave"
ALL_UP=true
for iface in $EXPECTED_IFACES; do
    if ! ip link show "$iface" 2>/dev/null | grep -q ",UP"; then
        ALL_UP=false
        break
    fi
done

if $ALL_UP; then
    echo "  CAN 接口已就绪, 跳过激活"
else
    if [[ $(id -u) -eq 0 ]]; then
        bash "$CAN_ACTIVATE"
    else
        echo "  需要 root 权限"
        sudo bash "$CAN_ACTIVATE"
    fi
fi
echo ""

# ── 2. Source ROS2 ───────────────────────────────────────────────────────
echo "[2/3] Source ROS2 环境..."

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "  退出 conda 环境 ($CONDA_DEFAULT_ENV)..."
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda deactivate 2>/dev/null || true
fi

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    echo "  OK: /opt/ros/${ROS_DISTRO}"
else
    echo "  [FAIL] /opt/ros/${ROS_DISTRO}/setup.bash 不存在"
    exit 1
fi

if [[ -f "$ROS2_WS/install/setup.bash" ]]; then
    source "$ROS2_WS/install/setup.bash"
    echo "  OK: $ROS2_WS"
else
    echo "  [FAIL] $ROS2_WS/install/setup.bash 不存在"
    echo "  请先编译: cd $ROS2_WS && colcon build"
    exit 1
fi
echo ""

# ── 3. 启动 Master-Slave (前台) ──────────────────────────────────────────
echo "[3/3] 启动 Master-Slave 遥操..."
echo "  master: can_left_mas / can_right_mas — 拖拽示教"
echo "  slave:  can_left_slave / can_right_slave — 跟随执行"
echo ""
echo "  相机: 未启动 (留给 capture_handeye.py)"
echo "  Ctrl+C 停止"
echo "============================================"
echo ""

ros2 launch piper teleop_launch.py
