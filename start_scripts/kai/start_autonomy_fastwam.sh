#!/usr/bin/env bash
# 用 FastWAM(无 test-time 视频想象的双专家 WAM)跑真机 autonomy —— 与 gwp/kai0 同栈同参。
#
# 架构(与 gwp 一致,仅 ws-port 后面的模型不同):
#   FastWAM 推理 = 独立 venv(gwp_eval_env)的 openpi-WebSocket server(serve_fastwam_ws.py, opt infer_action ~90ms);
#   控制 = 现有 kai0 policy_inference_node + start_autonomy.sh(--mode websocket 连该 port),继承全套控制参数。
#   FastWAM 的 action expert 只读首帧 KV(不 rollout 视频)→ 天然回避 gwp_ans 的闭环视频塌缩。
#
# 用法:
#   ./start_scripts/kai/start_autonomy_fastwam.sh --server-gpu 2            # observe-only
#   ./start_scripts/kai/start_autonomy_fastwam.sh --server-gpu 2 --execute  # 真机执行(手臂会动!)
#   急停: ros2 topic pub /policy/execute std_msgs/Bool "data: false"
#   旋钮: --nfe 4(去噪步) --opt-tier exact|fp8 --inference-rate 10 --debug-dump DIR
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

SERVER_GPU="${KAI0_FASTWAM_SERVER_GPU:-2}"
WS_PORT="${KAI0_FASTWAM_WS_PORT:-8004}"
NFE=4; OPT_TIER="exact"
INFER_RATE=10; EXEC_HORIZON=8; PUBLISH_RATE=30
EXECUTE_FLAG=""; DEBUG_DUMP=""
FW_VENV_PY="${FW_VENV_PY:-/home/tim/gwp_eval_env/venv/bin/python}"
FW_REPO="${FW_REPO:-$PWD/fastwam}"
WEIGHTS="${FASTWAM_WEIGHTS:-$FW_REPO/runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v3/checkpoints/weights/step_025510.pt}"
STATS="${FASTWAM_STATS:-$FW_REPO/data/visrobot01_fold/dataset_stats.json}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-gpu)     SERVER_GPU="$2"; shift 2 ;;
    --ws-port)        WS_PORT="$2"; shift 2 ;;
    --nfe)            NFE="$2"; shift 2 ;;
    --opt-tier)       OPT_TIER="$2"; shift 2 ;;
    --inference-rate) INFER_RATE="$2"; shift 2 ;;
    --exec-horizon)   EXEC_HORIZON="$2"; shift 2 ;;
    --publish-rate)   PUBLISH_RATE="$2"; shift 2 ;;
    --weights)        WEIGHTS="$2"; shift 2 ;;
    --debug-dump)     DEBUG_DUMP="$2"; shift 2 ;;
    --execute)        EXECUTE_FLAG="--execute"; shift ;;
    --no-execute)     EXECUTE_FLAG=""; shift ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done
case "$INFER_RATE" in *.*) ;; *) INFER_RATE="${INFER_RATE}.0" ;; esac   # 节点声明 DOUBLE, 必须带小数

# 控制参数(与 gwp/kai0 一致);enable_rtc=false(FastWAM 不消费 RTC);obs_image 近原生(server 侧拼 384x320)
CTRL_ARGS=( "latency_k:=6" "min_smooth_steps:=8" "publish_smooth_alpha:=0.7" "enable_rtc:=false"
            "cam_fps:=30" "fast_obs_pipeline:=true" "obs_image_h:=480" "obs_image_w:=640"
            "inference_rate:=${INFER_RATE}" "rtc_execute_horizon:=${EXEC_HORIZON}" "publish_rate:=${PUBLISH_RATE}" )

echo "=========================================================="
echo " FastWAM autonomy (同 kai0 栈): nfe=${NFE} tier=${OPT_TIER}"
echo " 控制: infer_rate=${INFER_RATE}Hz exec_horizon=${EXEC_HORIZON} publish_rate=${PUBLISH_RATE}Hz"
echo " ws server: GPU${SERVER_GPU} :${WS_PORT}  execute=${EXECUTE_FLAG:-observe-only}"
echo "=========================================================="
for p in "$FW_VENV_PY" "$WEIGHTS" "$STATS"; do [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }; done

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; pkill -f "serve_fastwam_ws.py --.*${WS_PORT}" 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---- 1. FastWAM openpi-WebSocket server ----
mkdir -p log
echo "[fastwam] starting ws server (compile/warmup ~1-2min)..."
( cd "$FW_REPO" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
    PYTHONPATH=src:scripts TORCHINDUCTOR_CACHE_DIR=/data2/gwp_eval/.inductor_fastwam \
    "$FW_VENV_PY" scripts/serve_fastwam_ws.py \
      --weights "$WEIGHTS" --stats "$STATS" --nfe "$NFE" --opt_tier "$OPT_TIER" \
      --port "$WS_PORT" --warmup 2 \
      ${DEBUG_DUMP:+--debug_dump_dir "$DEBUG_DUMP"} \
) > log/fastwam_server.log 2>&1 &
SERVER_PID=$!
echo "[fastwam] server pid=$SERVER_PID, log: log/fastwam_server.log"

echo -n "[fastwam] waiting for server ready"
for i in $(seq 1 180); do
  grep -q "ready, listening" log/fastwam_server.log 2>/dev/null && { echo " OK"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "ERROR: server died"; tail -20 log/fastwam_server.log; exit 4; }
  echo -n "."; sleep 2
  [[ $i -eq 180 ]] && { echo; echo "ERROR: server not ready after 360s"; exit 4; }
done

# ---- 2. cameras + arms + kai0 policy_inference_node (websocket -> fastwam server) ----
echo "[fastwam] launching kai0 autonomy stack (websocket :$WS_PORT)..."
# 不 exec: 保留 cleanup trap, Ctrl-C 时杀 fastwam server 防孤儿残留
./start_scripts/kai/start_autonomy.sh \
    --mode websocket --ws-port "$WS_PORT" --execution-mode joint $EXECUTE_FLAG \
    "${CTRL_ARGS[@]}" "${EXTRA_ARGS[@]}"
