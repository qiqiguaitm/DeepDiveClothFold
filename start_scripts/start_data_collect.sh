#!/bin/bash
###############################################################################
# kai0 数据采集一键启动
#
# 组合: USB 相机 reset + CAN 激活 + 主从遥操 + 3 相机 + Web 数据管理后端/前端
# 底层复用 web/data_manager/run.sh 的进程管理 (setsid/pidfile/start/stop/status/logs)
#
# 用法:
#   ./start_scripts/start_data_collect.sh               # 启动全部
#   ./start_scripts/start_data_collect.sh stop           # 停止全部
#   ./start_scripts/start_data_collect.sh restart        # 重启
#   ./start_scripts/start_data_collect.sh status         # 查看各服务状态
#   ./start_scripts/start_data_collect.sh logs [svc]     # 追踪日志 (arms|cameras|backend|frontend)
#
# 环境变量 (传递给 run.sh):
#   SKIP_ARMS=1        跳过机械臂
#   SKIP_CAMERAS=1     跳过相机
#   SKIP_PEDAL=1       跳过 USB 踏板监听 (默认启用; 无踏板硬件时会被重试循环自动退避)
#   SKIP_DEPS=1        跳过后端 pip 依赖同步
#   SKIP_CAN_DIAG=1    跳过 CAN 健康监控后台任务 (默认启用, 30s 间隔, 出事自动打包到 /tmp/can_diag/)
#   CAN_DIAG_INTERVAL=N (默认 30) — can_diag 快照间隔秒数
#   KAI0_DATA_ROOT=... 采集落盘根目录 (默认 /data1/DATA_IMP/KAI0)
#                      磁盘布局: $KAI0_DATA_ROOT/<Task>/<subset>/<YYYY-MM-DD>/{data,meta,videos}/
#                      (subset=base|dagger|...; 同一 subset 的多日数据聚在一棵子树, 方便整 subset
#                       做训练/同步; 路径生成在 web/data_manager/backend/app/layout.py)
#   PEDAL_VID/PEDAL_PID/PEDAL_KEY/PEDAL_EDGE/PEDAL_DEBOUNCE_MS
#                      踏板硬件参数覆盖, 详见 web/data_manager/backend/tools/pedal_listener.py
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
RUN_SH="$PROJECT_ROOT/web/data_manager/run.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

ACTION="${1:-start}"

# Replay support has migrated to start_autonomy.sh (`--replay [--sim]`).
# data_collect intentionally does NOT touch /tmp/kai0_deployment_mode: when only
# data_collect is running the marker is absent, so backend preflight rejects
# replay attempts (correct — teleop publishes /master/joint_* and would conflict).

# ── Pre-flight: only on start/restart ──
if [[ "$ACTION" == "start" || "$ACTION" == "restart" ]]; then
    echo "============================================================"
    echo " kai0 Data Collection"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # 1. ROS2 daemon restart (clean DDS state)
    eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
    source /opt/ros/jazzy/setup.bash 2>/dev/null || true
    ros2 daemon stop  2>/dev/null || true
    ros2 daemon start 2>/dev/null || true

    # 2. USB camera reset
    #
    # 优先走 /etc/sudoers.d/kai0-autonomy 的 NOPASSWD 规则 (sudo -n 一次成功).
    # 如果未配置或规则被改, fall back 到 SUDO_ASKPASS helper, 自动喂密码.
    # 密码来源优先级:
    #   1. 环境变量 $SUDO_PASSWORD (不写盘, 最安全)
    #   2. ~/.sudo_password 文件 (chmod 600, 不进 git)
    #   3. 字面量 "tim" (兜底, 仅为本机便利; **不要把这脚本连同字面量一起 push 出去**)
    #
    # ⚠️ 安全提醒: 硬编码弱口令 "tim" 是本机调试便利, 不应出现在远端/公共环境。
    #    长期建议: 扩展 /etc/sudoers.d/kai0-autonomy, 让 tim ALL=(ALL) NOPASSWD:
    #    /usr/bin/tee /sys/bus/usb/devices/*/authorized, 然后本 askpass 路径可以删除.
    echo "--- USB camera reset ---"
    _ASKPASS_SCRIPT="/tmp/.kai0_askpass_$$.sh"
    cleanup_askpass() { rm -f "$_ASKPASS_SCRIPT" 2>/dev/null; }
    trap cleanup_askpass EXIT
    if [ -n "${SUDO_PASSWORD:-}" ]; then
        _PW="$SUDO_PASSWORD"
    elif [ -r "$HOME/.sudo_password" ]; then
        _PW="$(head -n1 "$HOME/.sudo_password")"
    else
        # 无硬编码口令 (勿入库): 设 $SUDO_PASSWORD 或写 ~/.sudo_password (chmod 600);
        # 二者皆无则 askpass 给空, sudo -n NOPASSWD 路径仍可工作。
        _PW=""
    fi
    cat > "$_ASKPASS_SCRIPT" <<EOF
#!/bin/sh
echo '$_PW'
EOF
    chmod 700 "$_ASKPASS_SCRIPT"
    unset _PW
    export SUDO_ASKPASS="$_ASKPASS_SCRIPT"
    for dev in 2-1 2-2 4-2.2; do
        auth="/sys/bus/usb/devices/$dev/authorized"
        if [ -e "$auth" ]; then
            # Try NOPASSWD first (fast path), fall back to askpass if sudo -n fails
            echo 0 | sudo -n tee "$auth" >/dev/null 2>&1 \
                || echo 0 | sudo -A tee "$auth" >/dev/null 2>&1 \
                || true
            sleep 0.5
            echo 1 | sudo -n tee "$auth" >/dev/null 2>&1 \
                || echo 1 | sudo -A tee "$auth" >/dev/null 2>&1 \
                || true
        fi
    done
    unset SUDO_ASKPASS
    cleanup_askpass
    trap - EXIT
    sleep 3

    CAM_COUNT=$(lsusb | grep -c "Intel.*RealSense" 2>/dev/null || echo 0)
    if [ "$CAM_COUNT" -ge 3 ]; then
        ok "3 RealSense cameras detected"
    elif [ "$CAM_COUNT" -ge 2 ]; then
        warn "only $CAM_COUNT cameras (need 3)"
    else
        fail "only $CAM_COUNT cameras, check USB"
    fi

    # 3. Rebuild ros2_ws if source changed
    echo ""
    echo "--- Build check ---"
    INSTALL_MARKER="$ROS2_WS/install/piper/.colcon_install_layout"
    NEEDS_BUILD=false
    if [ ! -f "$INSTALL_MARKER" ]; then
        NEEDS_BUILD=true
    else
        NEWEST_SRC=$(find "$ROS2_WS/src/piper" -name '*.py' -newer "$INSTALL_MARKER" 2>/dev/null | head -1)
        if [ -n "$NEWEST_SRC" ]; then
            NEEDS_BUILD=true
        fi
    fi

    if [ "$NEEDS_BUILD" = true ]; then
        info "source changed, rebuilding..."
        (cd "$ROS2_WS" && source /opt/ros/jazzy/setup.bash && colcon build --packages-select piper 2>&1 | tail -3)
        ok "rebuild done"
    else
        ok "install up to date"
    fi
    echo ""
fi

# ── Delegate to run.sh ──
# SETUP_CAN=1: let run.sh activate CAN (start_teleop.sh handles it too,
#              but run.sh's activate_can path is the explicit toggle)
export SETUP_CAN=1
# Data collection needs 30 fps to match training data (launch_3cam.py defaults
# to 15 fps to ease USB bandwidth; override here for full-rate recording).
export CAM_FPS=30
# Recording convention: action = state (puppet/slave joint state) — matches
# official KAI0 upstream so this run's data mixes cleanly with kai0_base /
# kai0_dagger / advantage. Set KAI0_ACTION_EQ_STATE=0 to revert to legacy
# bilateral capture (master command goes into action). Consumed by
# web/data_manager/backend/app/ros_bridge.py::get_state_action.
export KAI0_ACTION_EQ_STATE="${KAI0_ACTION_EQ_STATE:-1}"
# V3 collection (2026-06-15): generate V3 datasets directly at record time.
#   KAI0_FRONT_TRIM=1         online leading-idle trim (EpisodeWriter rolling
#                             buffer; same semantics as build_no_release, keeps
#                             MARGIN=15 lead-in — NOT a full delete).
#   KAI0_GRIPPER_FROM_MASTER=1 action gripper dims (6,13) follow the master
#                             (teleop leader) grasp command; 12 arm dims stay = state.
# Set either to 0 to opt back out (e.g. legacy V2 capture).
export KAI0_FRONT_TRIM="${KAI0_FRONT_TRIM:-1}"
export KAI0_GRIPPER_FROM_MASTER="${KAI0_GRIPPER_FROM_MASTER:-1}"
# Dataset CONTENT version → auto-creates a version folder and tags the date leaf.
# Layout: KAI0/<Task>/<subset>/<vN>/<date>-<vN>/  (e.g. Task_A/base/v3/2026-06-15-v3).
# The recorder mkdir's the full path, so the v2/v3 folder is created on the fly
# and each episode lands under its version's subtree (train v2/v3 separately).
#   V3 = online front-trim + gripper-action-from-master; else legacy v2.
#   Override the version explicitly with KAI0_DATASET_VERSION=vN.
if [[ "$KAI0_FRONT_TRIM" == "1" && "$KAI0_GRIPPER_FROM_MASTER" != "0" ]]; then
    export KAI0_DATASET_VERSION="${KAI0_DATASET_VERSION:-v3}"
else
    export KAI0_DATASET_VERSION="${KAI0_DATASET_VERSION:-v2}"
fi
export KAI0_DATE_SUFFIX="-${KAI0_DATASET_VERSION}"   # date leaf suffix (layout.py)
# GPU video encode (2026-06-15): the teleop recorder runs under
# web/data_manager/backend/.venv (PyAV 17, NVENC-capable), so offload the 3×
# mp4 encode to the GPU hardware encoder — frees CPU and uses an otherwise-idle
# 5090. Output is still standard H.264/mp4 (decodes everywhere incl. the LeRobot
# loader). KAI0_NVENC_GPU picks the encode card; KAI0_VIDEO_CODEC=h264 reverts
# to CPU libx264. (The dagger recorder uses kai0/.venv PyAV 13 which lacks NVENC
# → it auto-falls back to libx264; that path is isolated via CPU affinity.)
export KAI0_VIDEO_CODEC="${KAI0_VIDEO_CODEC:-nvenc}"
export KAI0_NVENC_GPU="${KAI0_NVENC_GPU:-0}"
if [[ "${ACTION:-start}" == "start" || "${ACTION:-start}" == "restart" ]]; then
    if [[ "$KAI0_ACTION_EQ_STATE" == "1" ]]; then
        info "data convention: action == state (KAI0 official); gripper-from-master=$KAI0_GRIPPER_FROM_MASTER (dims 6,13 ← teleop leader)"
    else
        info "data convention: action = master (legacy bilateral; falls back to state if master topic missing)"
    fi
    info "V3 front-trim: $([ "$KAI0_FRONT_TRIM" = "1" ] && echo 'ON (leading-idle trimmed at record time, keep 15-frame lead-in)' || echo 'OFF (raw V2 capture)')"
    info "dataset version: $KAI0_DATASET_VERSION → auto folder <task>/<subset>/$KAI0_DATASET_VERSION/$(date +%Y-%m-%d)$KAI0_DATE_SUFFIX/"
    info "video codec: $KAI0_VIDEO_CODEC$([ "$KAI0_VIDEO_CODEC" = "nvenc" ] && echo " (GPU h264_nvenc @ GPU $KAI0_NVENC_GPU; auto-falls back to libx264 if unavailable)")"
fi

exec bash "$RUN_SH" "$ACTION" "${@:2}"
