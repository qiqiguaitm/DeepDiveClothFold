#!/usr/bin/env bash
# 用 GigaWorld-Policy 世界-动作模型 (gwp_ans / gwp_ori) 跑真机 autonomy。
#
# 架构: gwp 推理在独立 venv 的 server 进程 (serve_gwp_opt.py, ZeroMQ :PORT, fp8+T_a3 ~87ms);
#        现有 kai0 栈起相机+机械臂 (autonomy_launch.py enable_policy:=false, 不起 kai0 policy 节点);
#        gwp_bridge_node.py 做 ROS I/O + 客户端 + 平滑, 发 /master/joint_*。
#
# 用法:
#   ./start_scripts/kai/start_autonomy_gwp.sh            # 默认 gwp_ans, observe-only
#   ./start_scripts/kai/start_autonomy_gwp.sh --model ori --server-gpu 1
#   起来后真机执行: ros2 topic pub /policy/execute std_msgs/Bool "data: true"   (急停: data: false)
#
# 安全: 默认 observe-only (bridge /policy/execute 初值 false), 手臂不动; 显式发 /policy/execute=true 才动。
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- defaults ----
MODEL="ans"                 # ans | ori
SERVER_GPU="${KAI0_GWP_SERVER_GPU:-0}"
SERVER_PORT=8093
OPT_TIER="fp8"              # eager | exact | fp8
STEPS_ACT=3                 # T_a; ans 默认 3 (部署王牌档)
GWP_VENV_PY="${GWP_VENV_PY:-/home/tim/gwp_eval_env/venv/bin/python}"
GWP_REPO="${GWP_REPO:-/data2/gwp_eval/repo/giga_world_policy}"
CKPT_ROOT="${GWP_CKPT_ROOT:-/data2/gwp_eval/checkpoints}"
T5_PKL="${GWP_T5_PKL:-/data2/gwp_eval/data/visrobot01_val/t5_embedding/episode_000000.pt}"
EXEC_DEFAULT=false          # --execute -> true: 桥启动即驱动机械臂 (与 kai0 --execute 对称)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --server-gpu)  SERVER_GPU="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --opt-tier)    OPT_TIER="$2"; shift 2 ;;
    --steps-act)   STEPS_ACT="$2"; shift 2 ;;
    --execute)     EXEC_DEFAULT=true; shift ;;
    --no-execute)  EXEC_DEFAULT=false; shift ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$MODEL" in
  ans) TRANSFORMER="$CKPT_ROOT/gwp_ans/transformer" ;;
  ori) TRANSFORMER="$CKPT_ROOT/gwp_ori/transformer" ;;
  *)   echo "ERROR: --model must be ans|ori"; exit 2 ;;
esac
MODEL_ID="$CKPT_ROOT/Wan2.2-TI2V-5B-Diffusers"
STATS="$GWP_REPO/assets_visrobot01/norm_stats_vis_abs.json"

echo "=========================================================="
echo " gwp autonomy: model=gwp_${MODEL}  tier=${OPT_TIER} T_a=${STEPS_ACT}"
echo " server: GPU${SERVER_GPU} :${SERVER_PORT}   ckpt=$TRANSFORMER"
echo "=========================================================="

# ---- preflight: paths + bridge python deps ----
for p in "$GWP_VENV_PY" "$TRANSFORMER" "$MODEL_ID" "$STATS" "$T5_PKL"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }
done

SERVER_PID=""; BRIDGE_PID=""
cleanup() {
  echo; echo "[gwp] cleanup..."
  [[ -n "$BRIDGE_PID" ]] && kill "$BRIDGE_PID" 2>/dev/null
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  pkill -f "serve_gwp_opt.py --port ${SERVER_PORT}" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---- 1. start gwp inference server (gwp venv) ----
echo "[gwp] starting inference server (compile/warmup may take ~1-2min)..."
mkdir -p log
( cd "$GWP_REPO" && CUDA_VISIBLE_DEVICES="$SERVER_GPU" PYTHONPATH=. \
    TORCHINDUCTOR_CACHE_DIR=/data2/gwp_eval/.inductor \
    "$GWP_VENV_PY" scripts/serve_gwp_opt.py \
      --transformer_path "$TRANSFORMER" --model_id "$MODEL_ID" --stats_path "$STATS" \
      --t5_embedding_pkl "$T5_PKL" --opt_tier "$OPT_TIER" --steps_act "$STEPS_ACT" \
      --port "$SERVER_PORT" --warmup 1 \
) > log/gwp_server.log 2>&1 &
SERVER_PID=$!
echo "[gwp] server pid=$SERVER_PID, log: log/gwp_server.log"

# ---- 2. wait for server ready (look for 'ready, listening' in log) ----
echo -n "[gwp] waiting for server ready"
for i in $(seq 1 180); do
  grep -q "ready, listening" log/gwp_server.log 2>/dev/null && { echo " OK"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "ERROR: server died, see log/gwp_server.log"; tail -20 log/gwp_server.log; exit 4; }
  echo -n "."; sleep 2
  [[ $i -eq 180 ]] && { echo; echo "ERROR: server not ready after 360s"; exit 4; }
done

# ---- 3. resolve a python for the bridge (needs rclpy + zmq + torch) ----
#  ROS2 节点用系统/ROS 环境的 python; 这里假设已 source ros2_ws/install/setup.bash。
BRIDGE_PY="${GWP_BRIDGE_PY:-python3}"
if ! "$BRIDGE_PY" -c "import rclpy, zmq, torch" 2>/dev/null; then
  echo "WARN: bridge python ($BRIDGE_PY) missing rclpy/zmq/torch."
  echo "      Source ROS2 (source ros2_ws/install/setup.bash) and ensure pyzmq+torch(cpu) installed,"
  echo "      or set GWP_BRIDGE_PY=/path/to/python. Continuing — bridge may fail."
fi

# ---- 4. start the bridge (idles until camera/joint topics appear) ----
echo "[gwp] starting gwp_bridge_node (execute_default=$EXEC_DEFAULT)..."
"$BRIDGE_PY" ros2_ws/src/piper/scripts/gwp_bridge_node.py \
    --ros-args -p server_port:="$SERVER_PORT" -p execute_default:="$EXEC_DEFAULT" \
    > log/gwp_bridge.log 2>&1 &
BRIDGE_PID=$!
echo "[gwp] bridge pid=$BRIDGE_PID, log: log/gwp_bridge.log"

# ---- 5. bring up cameras + arms (NO kai0 policy node) in foreground ----
echo "[gwp] bringing up cameras + arms (enable_policy:=false)..."
exec ./start_scripts/kai/start_autonomy.sh enable_policy:=false "${EXTRA_ARGS[@]}"
