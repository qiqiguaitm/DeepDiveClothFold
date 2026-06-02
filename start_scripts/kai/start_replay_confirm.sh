#!/bin/bash
###############################################################################
# kai0 Replay — 单脚本 · 手动确认版 (路径 A)
#
# 把"起 replay 栈 → 等就绪 → 交互确认回放 → 拆栈"四步合成一条命令。等价于:
#   终端1: ./start_autonomy.sh --replay --episode <ep> auto_execute:=false
#   终端2: ./start_replay_test.sh <ep> [rate]
# 但全程一个脚本、一个终端，退出/Ctrl-C 时自动拆栈。
#
# 与 `start_autonomy.sh --replay --episode <ep>` 的区别:
#   后者 auto_execute=true，落地即自动下发 (仅节点内 S4 对齐预检，无人工确认)。
#   本脚本 auto_execute=false 起 idle 栈，再走 start_replay_test.sh 的完整安全栏
#   (S1-S6) + 交互式 [y/N] 确认 + rate 慢放 + /replay_progress 实时进度。
#
# 用法:
#   ./start_replay_confirm.sh <task>/<subset>/<date>/<ep_id> [rate]
#
# 示例 (注意日期目录带 -v2 后缀):
#   ./start_replay_confirm.sh Task_A/base/2026-05-19-v2/0
#   ./start_replay_confirm.sh Task_A/base/2026-05-19-v2/0 0.8     # 0.8x 慢放
#
# 中途停: Ctrl+C — 先 abort 回放 (execute=false)，再拆栈。
###############################################################################
set -eo pipefail

EP="${1:-}"
RATE="${2:-1.0}"
if [[ -z "$EP" ]]; then
    echo "用法: $0 <task>/<subset>/<date>/<ep_id> [rate]" >&2
    echo "示例: $0 Task_A/base/2026-05-19-v2/0 0.8" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STACK_LOG=/tmp/replay_stack.log

# ── teardown: 无论正常结束 / 非零退出 / Ctrl-C 都拆栈 (幂等) ──
# playback_launch.py 起的节点: replay + arm_reader ×2 + video_publisher +
# joint_state_loopback。全部 pkill，并清掉自己写的 replay marker。
teardown() {
    echo ""
    echo "[replay] 拆栈中..."
    for pat in playback_launch replay_node arm_reader_node video_publisher_node joint_state_loopback; do
        pkill -9 -f "$pat" 2>/dev/null || true
    done
    if [[ "$(cat /tmp/kai0_deployment_mode 2>/dev/null)" == "replay" ]]; then
        rm -f /tmp/kai0_deployment_mode
    fi
    echo "[replay] 栈已停"
}
trap teardown EXIT INT TERM

# ── 1. 后台起 idle 栈 (auto_execute=false → 落地不自动下发) ──
echo "[replay] 起 idle replay 栈 (auto_execute=false)，日志 → $STACK_LOG"
nohup "$SCRIPT_DIR/start_autonomy.sh" --replay --episode "$EP" auto_execute:=false \
    > "$STACK_LOG" 2>&1 &

# ── 2. source ROS2，轮询等 /replay 就绪 (栈早退即报错并打日志) ──
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
[[ -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]] && \
    source "$PROJECT_ROOT/ros2_ws/install/setup.bash" 2>/dev/null || true
command -v ros2 >/dev/null 2>&1 || { echo "[FAIL] ros2 不在 PATH (source /opt/ros/jazzy/setup.bash 失败)"; exit 5; }

# daemon 刷新一次，清掉上一轮死栈残留的 stale 拓扑。必须放在"等节点 + param set
# + pub execute"之前——若放在它们之后 (紧挨 pub --once)，会重置 DDS 发现，导致
# start_replay_test.sh 的 `pub --once /policy/execute` 在发现未完成时发出即丢，
# 节点收不到 execute → 停在 frame 0 不动 (本脚本初版的真实 bug)。
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

echo -n "[replay] 等 /replay 就绪 "
READY=false
for i in $(seq 1 60); do
    if ros2 node list 2>/dev/null | grep -qx '/replay'; then READY=true; break; fi
    # 起栈失败 (CAN 没起来 / parquet 解析失败等) → 进程会早退，别干等满 60s
    if [[ $i -gt 5 ]] && ! pgrep -f playback_launch >/dev/null 2>&1; then
        echo ""; echo "[FAIL] 起栈进程已退出，启动失败。日志末尾:"; tail -25 "$STACK_LOG"; exit 1
    fi
    echo -n "."; sleep 1
done
echo ""
[[ "$READY" == "true" ]] || { echo "[FAIL] 60s 内 /replay 未就绪。日志末尾:"; tail -25 "$STACK_LOG"; exit 1; }
echo "[replay] /replay 就绪"
# 给发现一点稳定时间 (节点刚 up，warm 一下，让随后 pub --once execute 必达)
sleep 2

# ── 3. 交互确认回放 (preflight S1-S6 + [y/N] + 进度)；退出后由 trap 拆栈 ──
echo "[replay] 进入交互确认流程 (start_replay_test.sh)"
echo ""
"$SCRIPT_DIR/start_replay_test.sh" "$EP" "$RATE"
