#!/usr/bin/env bash
# 在线 shadow 对比的"gwp 影子端"一键起:起 gwp ws server(等就绪)→ 起对比器。
# kai0 你单独跑(另一个终端): start_autonomy_from_ckpt_v1.sh <ckpt> --execute
#
# 数据流: kai0 节点执行叠衣 → /policy/action_chunk(本对比器只读); 本脚本的 gwp server 当影子,
#         对比器逐帧 query gwp + 读 kai0 块 → MAE@{1,8,16} + 两者 motion → csv。
#
# 用法:
#   # 终端A: ./start_scripts/kai/start_autonomy_from_ckpt_v1.sh <kai0_ckpt> --execute
#   # 终端B:
#   ./start_scripts/kai/start_shadow_compare.sh --model ans
#   #   可选: --server-gpu N --ws-port 8003 --rate 4 --debug-dump /data2/gwp_eval/out/online_dump
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="ans"
SERVER_GPU="${KAI0_GWP_SERVER_GPU:-0}"
WS_PORT="${KAI0_GWP_WS_PORT:-8003}"
OPT_TIER="fp8"; STEPS_ACT=3
RATE=4
OUT="/data2/gwp_eval/out/online_shadow_compare.csv"
DEBUG_DUMP=""
GWP_VENV_PY="${GWP_VENV_PY:-/home/tim/gwp_eval_env/venv/bin/python}"
GWP_REPO="${GWP_REPO:-/data2/gwp_eval/repo/giga_world_policy}"
CKPT_ROOT="${GWP_CKPT_ROOT:-/data2/gwp_eval/checkpoints}"
T5_PKL="${GWP_T5_PKL:-/data2/gwp_eval/data/visrobot01_val/t5_embedding/episode_000000.pt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --server-gpu)  SERVER_GPU="$2"; shift 2 ;;
    --ws-port)     WS_PORT="$2"; shift 2 ;;
    --opt-tier)    OPT_TIER="$2"; shift 2 ;;
    --steps-act)   STEPS_ACT="$2"; shift 2 ;;
    --rate)        RATE="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --debug-dump)  DEBUG_DUMP="$2"; shift 2 ;;
    *)             echo "unknown arg: $1"; exit 2 ;;
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
echo " shadow 影子端: gwp_${MODEL} @ GPU${SERVER_GPU}:${WS_PORT}  rate=${RATE}Hz  out=${OUT}"
echo " (kai0 请在另一终端 --execute 单独跑)"
echo "=========================================================="
for p in "$GWP_VENV_PY" "$TRANSFORMER" "$MODEL_ID" "$STATS" "$T5_PKL" "kai0/.venv/bin/python"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }
done

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; pkill -f "serve_gwp_ws.py --.*${WS_PORT}" 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---- 1. gwp ws server (影子) ----
mkdir -p log
echo "[shadow] starting gwp ws server (compile/warmup ~1-2min)..."
( cd "$GWP_REPO" && CUDA_VISIBLE_DEVICES="$SERVER_GPU" PYTHONPATH=. \
    TORCHINDUCTOR_CACHE_DIR=/data2/gwp_eval/.inductor \
    "$GWP_VENV_PY" scripts/serve_gwp_ws.py \
      --transformer_path "$TRANSFORMER" --model_id "$MODEL_ID" --stats_path "$STATS" \
      --t5_embedding_pkl "$T5_PKL" --opt_tier "$OPT_TIER" --steps_act "$STEPS_ACT" \
      --port "$WS_PORT" --warmup 2 \
      ${DEBUG_DUMP:+--debug_dump_dir "$DEBUG_DUMP" --debug_dump_n 15} \
) > log/gwp_shadow_server.log 2>&1 &
SERVER_PID=$!
echo "[shadow] server pid=$SERVER_PID, log: log/gwp_shadow_server.log"

echo -n "[shadow] waiting for server ready"
for i in $(seq 1 180); do
  grep -q "ready, listening" log/gwp_shadow_server.log 2>/dev/null && { echo " OK"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "ERROR: server died"; tail -20 log/gwp_shadow_server.log; exit 4; }
  echo -n "."; sleep 2
  [[ $i -eq 180 ]] && { echo; echo "ERROR: server not ready after 360s"; exit 4; }
done

# ---- 2. comparator (ROS sourced + kai0/.venv python) ----
echo "[shadow] starting comparator (reads /policy/action_chunk from kai0, queries gwp)..."
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source ros2_ws/install/setup.bash 2>/dev/null || true
exec kai0/.venv/bin/python train_scripts/kai/eval/compare_online_kai0_gwp.py \
    --gwp-port "$WS_PORT" --rate "$RATE" --out "$OUT"
