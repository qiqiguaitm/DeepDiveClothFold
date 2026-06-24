#!/usr/bin/env bash
# 用 GigaWorld-Policy 世界-动作模型 (gwp_ans / gwp_ori) 跑真机 autonomy —— 与 kai0 同栈同参。
#
# 架构 (与 kai0 一致, 仅 ws-port 后面的模型不同):
#   gwp 推理 = 独立 venv 的 openpi-WebSocket server (serve_gwp_ws.py, fp8+T_a3 ~87ms);
#   控制 = **现有 kai0 policy_inference_node + start_autonomy.sh**(--mode websocket 连该 port),
#          继承 inference_rate / latency_k / rtc_execute_horizon / publish_rate / StreamActionBuffer /
#          min-jerk / proprio 反馈 / jump-protect / rerun / recorder 全套 —— 在线对比 apples-to-apples。
#   (取代旧的 ZeroMQ 桥 gwp_bridge_node.py + serve_gwp_opt.py)
#
# 用法:
#   ./start_scripts/kai/start_autonomy_gwp.sh --model ans            # observe-only
#   ./start_scripts/kai/start_autonomy_gwp.sh --model ans --execute  # 真机执行 (手臂会动!)
#   控制旋钮 (gwp 默认 vs kai0): --inference-rate 10(kai0 20) --exec-horizon 8(kai0 12) --publish-rate 30(kai0 40)
#   急停: ros2 topic pub /policy/execute std_msgs/Bool "data: false"
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- defaults ----
MODEL="ans"
SERVER_GPU="${KAI0_GWP_SERVER_GPU:-0}"
WS_PORT="${KAI0_GWP_WS_PORT:-8003}"      # 8000 常被占; 8001=FLASH 8002=V1, gwp 用 8003
OPT_TIER="fp8"; STEPS_ACT=3; STEPS_INF=10   # gwp_ans: fp8/T_a3/T_O10; gwp_ori: exact/steps_inf=5/steps_act=5(NFE5)
DEBUG_DUMP=""            # --debug-dump DIR: 落盘头 15 次在线 ref图+state+action 做诊断
# gwp 默认控制节奏 (与 kai0 的 12/40/20 不同 —— gwp 重规划更勤补长程弱、发布贴 30fps 数据、推理封顶 ~10Hz)
INFER_RATE=10            # 推理(重规划)Hz 上限; gwp_ans ~90ms/次 → 实际天花板 ~10-11Hz
EXEC_HORIZON=8           # 每块执行步数 (-> rtc_execute_horizon); gwp 用 8 (kai0=12)
PUBLISH_RATE=30          # 发布 Hz; gwp 用 30 (贴数据 30fps; kai0=40)
EXECUTE_FLAG=""
GWP_VENV_PY="${GWP_VENV_PY:-/home/tim/gwp_eval_env/venv/bin/python}"
GWP_REPO="${GWP_REPO:-/data2/gwp_eval/repo/giga_world_policy}"
CKPT_ROOT="${GWP_CKPT_ROOT:-/data2/gwp_eval/checkpoints}"
T5_PKL="${GWP_T5_PKL:-/data2/gwp_eval/data/visrobot01_val/t5_embedding/episode_000000.pt}"
# 控制参数 (与 kai0 V1 一致, 便于公平对比); enable_rtc=false 因 gwp 不消费 RTC 引导。
# inference_rate / rtc_execute_horizon / publish_rate 由 --flag 注入 (见下); 其余固定。
CTRL_ARGS=( "latency_k:=6" "min_smooth_steps:=8" "publish_smooth_alpha:=0.7" "enable_rtc:=false"
            "cam_fps:=30" "fast_obs_pipeline:=true"
            "obs_image_h:=480" "obs_image_w:=640" )   # gwp 要近原生帧 (server 拼 768x192)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --server-gpu)  SERVER_GPU="$2"; shift 2 ;;
    --ws-port)     WS_PORT="$2"; shift 2 ;;
    --opt-tier)    OPT_TIER="$2"; shift 2 ;;
    --steps-act)   STEPS_ACT="$2"; shift 2 ;;
    --steps-inf)   STEPS_INF="$2"; shift 2 ;;
    --inference-rate) INFER_RATE="$2"; shift 2 ;;
    --exec-horizon)   EXEC_HORIZON="$2"; shift 2 ;;
    --publish-rate)   PUBLISH_RATE="$2"; shift 2 ;;
    --debug-dump)     DEBUG_DUMP="$2"; shift 2 ;;
    --execute)     EXECUTE_FLAG="--execute"; shift ;;
    --no-execute)  EXECUTE_FLAG=""; shift ;;
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
# inference_rate 节点声明为 DOUBLE → 必须带小数 (否则 InvalidParameterTypeException 崩节点)。
case "$INFER_RATE" in *.*) ;; *) INFER_RATE="${INFER_RATE}.0" ;; esac
CTRL_ARGS+=( "inference_rate:=${INFER_RATE}" "rtc_execute_horizon:=${EXEC_HORIZON}" "publish_rate:=${PUBLISH_RATE}" )

echo "=========================================================="
echo " gwp autonomy (同 kai0 栈): model=gwp_${MODEL} tier=${OPT_TIER} T_a=${STEPS_ACT} T_O/steps_inf=${STEPS_INF}"
echo " 控制: infer_rate=${INFER_RATE}Hz exec_horizon=${EXEC_HORIZON} publish_rate=${PUBLISH_RATE}Hz"
echo " ws server: GPU${SERVER_GPU} :${WS_PORT}   execute=${EXECUTE_FLAG:-observe-only}"
echo "=========================================================="
for p in "$GWP_VENV_PY" "$TRANSFORMER" "$MODEL_ID" "$STATS" "$T5_PKL"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }
done

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; pkill -f "serve_gwp_ws.py --.*${WS_PORT}" 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---- 1. start gwp openpi-WebSocket server (gwp venv) ----
mkdir -p log
echo "[gwp] starting ws server (compile/warmup ~1-2min)..."
( cd "$GWP_REPO" && CUDA_VISIBLE_DEVICES="$SERVER_GPU" PYTHONPATH=. \
    TORCHINDUCTOR_CACHE_DIR=/data2/gwp_eval/.inductor \
    "$GWP_VENV_PY" scripts/serve_gwp_ws.py \
      --transformer_path "$TRANSFORMER" --model_id "$MODEL_ID" --stats_path "$STATS" \
      --t5_embedding_pkl "$T5_PKL" --opt_tier "$OPT_TIER" --steps_act "$STEPS_ACT" --steps_inf "$STEPS_INF" \
      --port "$WS_PORT" --warmup 2 \
      ${DEBUG_DUMP:+--debug_dump_dir "$DEBUG_DUMP" --debug_dump_n 15} \
) > log/gwp_server.log 2>&1 &
SERVER_PID=$!
echo "[gwp] server pid=$SERVER_PID, log: log/gwp_server.log"

echo -n "[gwp] waiting for server ready"
for i in $(seq 1 180); do
  grep -q "ready, listening" log/gwp_server.log 2>/dev/null && { echo " OK"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "ERROR: server died"; tail -20 log/gwp_server.log; exit 4; }
  echo -n "."; sleep 2
  [[ $i -eq 180 ]] && { echo; echo "ERROR: server not ready after 360s"; exit 4; }
done

# ---- 2. bring up cameras + arms + kai0 policy_inference_node (websocket -> gwp server) ----
echo "[gwp] launching kai0 autonomy stack (websocket :$WS_PORT)..."
# 不用 exec: 保留本脚本的 cleanup trap, 这样 Ctrl-C 时能杀掉 gwp server (否则它被孤儿化, 残留占 8003 → 下次启动撞死)
./start_scripts/kai/start_autonomy.sh \
    --mode websocket --ws-port "$WS_PORT" --execution-mode joint $EXECUTE_FLAG \
    "${CTRL_ARGS[@]}" "${EXTRA_ARGS[@]}"
