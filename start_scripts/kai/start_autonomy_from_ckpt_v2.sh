#!/bin/bash
###############################################################################
# V2 (FLASH 投机推理) autonomy 启动脚本.
#
# 从 PyTorch pi05 ckpt 直起 FLASH 投机 server (serve_policy_flash.py) + 标准
# autonomy 栈 (--mode websocket). 与 v0/v1 同款 server-only 思路: policy_inference_node
# 走现成的 websocket 客户端, **不改任何 ROS2 / 旧代码**.
#
# FLASH = draft head 一次前向给出整条 chunk → Action Expert 一步 verify → radius 接受
# 最长前缀; 低接受帧 full_fallback 退回标准 denoise (输出≈baseline). 见
# docs/deployment/inference/realtime_vla/flash_future_research.md + flash_impl_log.md.
#
# 使用:
#   ./start_scripts/kai/start_autonomy_from_ckpt_v2.sh <ckpt_dir> [--draft <draft.pt>] \
#        [--port 8001] [--gpu N] [--tau 0.3] [--no-execute] [--trace] [--no-spec] [其他 autonomy args...]
#
#   --trace    : 每帧写一行 JSONL (accept/fallback/radius + 臂/夹爪 chunk 运动幅度) 到
#                /tmp/flash_trace_<port>_{spec,nospec}_<ts>.jsonl, 供 flash_trace_phase_report.py 分析。
#   --no-spec  : A/B 基线 — 跳过 draft+verify, 纯 eager 全量去噪 (= 你之前 --no-spec 真机跑的)。
#                配 --trace 录基线臂运动, 与带 spec 的 trace 直接比幅值 (诊断真机 OOD 欠驱动)。
#
# 要求:
#   <ckpt_dir>/train_config.json   ({"base_config_name":..., "override_asset_id":...})
#   <ckpt_dir>/model.safetensors   (PyTorch ckpt; JAX-only ckpt 不支持 FLASH)
#   <ckpt_dir>/assets/<asset_id>/norm_stats.json
#   一个**为该 ckpt 自蒸馏的** draft head:
#       --draft <path>  或  <ckpt_dir>/draft_head.pt
#     若缺失, 用 train_scripts/kai/eval/spec_draft_r1d.py 蒸馏一个再来.
#
# draft head 是逐 ckpt 产物, 与 ckpt 不配对会被 radius 持续拒绝 → 一直 fallback
# (能跑但无加速). 上机务必看 server 日志的 mean_accept / fallback%.
###############################################################################

set -eo pipefail

CKPT_DIR="${1:?Usage: $0 <ckpt_dir> [--draft <draft.pt>] [--port N] [--gpu N] [--tau F] [--no-execute] [extra args...]}"
shift || true

# ── 参数解析 (V2 专属 flag 截下, 其余透传给 start_autonomy.sh) ──
DRAFT=""
WS_PORT="8001"            # flash 默认 8001 (8000=JAX,8002=V1,8003=XVLA)
FLASH_GPU="${KAI0_FLASH_GPU_ID:-}"
TAU="0.3"
EXECUTE="--execute"      # 默认直接执行; --no-execute 改成观察模式
SEED_ARG=()
TRACE_FLAG=0
TRACE_ARG=()
NOSPEC_ARG=()
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --draft)       DRAFT="$2"; shift 2 ;;
        --port)        WS_PORT="$2"; shift 2 ;;
        --gpu)         FLASH_GPU="$2"; shift 2 ;;
        --tau)         TAU="$2"; shift 2 ;;
        --seed)        SEED_ARG=(--seed "$2"); shift 2 ;;
        --trace)       TRACE_FLAG=1; shift ;;
        --no-spec)     NOSPEC_ARG=(--no-spec); shift ;;
        --no-execute)  EXECUTE=""; shift ;;
        *)             PASSTHRU+=("$1"); shift ;;
    esac
done

# trace 路径在解析完才建 (此时 WS_PORT 已定); no-spec 时文件名带 _nospec 便于 A/B 区分
if [ "$TRACE_FLAG" = "1" ]; then
    _TRACE_TAG=$([ ${#NOSPEC_ARG[@]} -gt 0 ] && echo "nospec" || echo "spec")
    TRACE_ARG=(--trace-out "/tmp/flash_trace_${WS_PORT}_${_TRACE_TAG}_$(date +%Y%m%d_%H%M%S).jsonl")
    echo "    trace →      ${TRACE_ARG[1]}"
fi

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/data1/miniconda3/bin/python
VENV_PY="$REPO/kai0/.venv_5090/bin/python"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1" >&2; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

# ── 1. 校验 ckpt ──
[ -d "$CKPT_DIR" ] || fail "ckpt_dir not found: $CKPT_DIR"
[ -f "$CKPT_DIR/train_config.json" ] || fail "$CKPT_DIR/train_config.json missing — 不是 pack_inference_ckpt.py 产物?"
[ -f "$CKPT_DIR/model.safetensors" ] || fail "$CKPT_DIR/model.safetensors 缺失 — FLASH 只支持 PyTorch ckpt (JAX ckpt 请用 v0/v1)"
[ -f "$VENV_PY" ] || fail "PyTorch venv 缺失: $VENV_PY (FLASH server 需要 .venv_5090)"

CONFIG_NAME=$($PY -c "import json; print(json.load(open('$CKPT_DIR/train_config.json'))['base_config_name'])")
ASSET_ID=$($PY -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('override_asset_id',''))")
DATASET_ID=$($PY -c "import json; v=json.load(open('$CKPT_DIR/train_config.json')).get('deploy_dataset_id'); print('' if v is None else int(v))")

if [ -n "$ASSET_ID" ] && [ ! -f "$CKPT_DIR/assets/$ASSET_ID/norm_stats.json" ]; then
    fail "$CKPT_DIR/assets/$ASSET_ID/norm_stats.json missing (override_asset_id mismatch)"
fi

# ── 2. 定位 draft head ──
if [ -z "$DRAFT" ]; then
    if [ -f "$CKPT_DIR/draft_head.pt" ]; then
        DRAFT="$CKPT_DIR/draft_head.pt"
    else
        fail "未提供 draft head. 给 --draft <path> 或把它放到 $CKPT_DIR/draft_head.pt
       该 draft 必须是为本 ckpt 自蒸馏的产物, 用:
         CUDA_VISIBLE_DEVICES=N $VENV_PY \\
           $REPO/train_scripts/kai/eval/spec_draft_r1d.py \\
           --config $CONFIG_NAME --ckpt $CKPT_DIR --asset-id $ASSET_ID \\
           --out /tmp/draft_${ASSET_ID}.pt   # 训完拷到 $CKPT_DIR/draft_head.pt"
    fi
fi
[ -f "$DRAFT" ] || fail "draft head not found: $DRAFT"

# ── 3. 端口空闲检查 ──
if ss -tlnp 2>/dev/null | grep -q ":${WS_PORT} "; then
    fail "端口 :${WS_PORT} 已被占用 (可能有残留 server). 换 --port 或先杀掉占用进程."
fi

# ── 4. 选 GPU (flash server 用; 默认挑最空闲的一张, 与 ROS/JAX 侧错开) ──
if [ -z "$FLASH_GPU" ]; then
    FLASH_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    FLASH_GPU="${FLASH_GPU:-0}"
fi
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$FLASH_GPU" 2>/dev/null || echo '?')

# Deploy-time gripper frame remap (old 100mm-range ckpt → real 0–70mm robot).
# 默认关。部署官方夹爪标定(0–70mm)前训练的旧 ckpt 时设 =1。serve_policy_flash.py
# (经 create_trained_policy) 读取。见 docs/deployment/data_collection/gripper_calibration.md
export KAI0_GRIPPER_DEPLOY_REMAP="${KAI0_GRIPPER_DEPLOY_REMAP:-0}"
export KAI0_GRIPPER_REAL_RANGE="${KAI0_GRIPPER_REAL_RANGE:-0.0,0.07}"
[ "$KAI0_GRIPPER_DEPLOY_REMAP" = "1" ] && echo "[gripper-remap] ON: 夹爪 norm_stats [q01,q99]→真机[$KAI0_GRIPPER_REAL_RANGE]m (dims 6,13)"

echo "============================================================"
echo "  start_autonomy_from_ckpt_v2.sh  (FLASH speculative)"
echo "    ckpt_dir:    $CKPT_DIR"
echo "    base_config: $CONFIG_NAME"
echo "    asset_id:    $ASSET_ID"
echo "    dataset_id:  ${DATASET_ID:-<none>}"
echo "    draft head:  $DRAFT"
echo "    flash port:  $WS_PORT   (action_kind=joint, 14D)"
echo "    flash GPU:   $FLASH_GPU  (free ${FREE_MB}MB)"
echo "    tau_radius:  $TAU"
echo "    execute:     ${EXECUTE:-<observe-only>}"
echo "============================================================"

# ── 5. 启动 FLASH server (后台) ──
SERVE_LOG="/tmp/serve_policy_flash_${WS_PORT}.log"
info "starting FLASH server → $SERVE_LOG"

DRAFT_ARG=(--draft "$DRAFT")
ASSET_ARG=(); [ -n "$ASSET_ID" ] && ASSET_ARG=(--asset-id "$ASSET_ID")

CUDA_VISIBLE_DEVICES="$FLASH_GPU" JAX_PLATFORMS="" \
    "$VENV_PY" "$REPO/kai0/scripts/serve_policy_flash.py" \
    --config "$CONFIG_NAME" \
    --dir "$CKPT_DIR" \
    "${ASSET_ARG[@]}" \
    "${DRAFT_ARG[@]}" \
    --port "$WS_PORT" \
    --tau "$TAU" \
    "${SEED_ARG[@]}" \
    "${TRACE_ARG[@]}" \
    "${NOSPEC_ARG[@]}" \
    >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!

# 退出时清理 server (Ctrl+C autonomy 栈也会触发)
cleanup() {
    if kill -0 "$SERVE_PID" 2>/dev/null; then
        info "stopping FLASH server (pid $SERVE_PID)…"
        kill "$SERVE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── 6. 等 server 就绪 (端口监听) ──
info "waiting for FLASH server on :${WS_PORT} (model load 可能 ~30-60s)…"
for i in $(seq 1 120); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "----- last 30 lines of $SERVE_LOG -----" >&2
        tail -30 "$SERVE_LOG" >&2 || true
        fail "FLASH server 进程提前退出 (见上). 常见: draft 与 ckpt 不配对 / norm_stats 缺失 / GPU OOM."
    fi
    if ss -tlnp 2>/dev/null | grep -q ":${WS_PORT} "; then
        ok "FLASH server ready on :${WS_PORT} ($(($i*1)) checks)"
        break
    fi
    sleep 1
    [ "$i" -eq 120 ] && fail "FLASH server 120s 内未监听 :${WS_PORT} — 见 $SERVE_LOG"
done

# ── 7. 拉起 autonomy 栈 (websocket 模式连本地 FLASH server) ──
# 注意: websocket 模式下 config_name/checkpoint_dir 在 node 侧是 inert 的 (node 不本地
# 加载模型, 走 ws 客户端), 仍透传以保持日志一致 + 满足 launch 参数.
EXTRA=()
[ -n "$DATASET_ID" ] && EXTRA+=("dataset_id:=$DATASET_ID")

info "launching autonomy stack (--mode websocket --ws-port $WS_PORT)…"
"$REPO/start_scripts/kai/start_autonomy.sh" \
    --mode websocket --ws-port "$WS_PORT" $EXECUTE \
    "config_name:=$CONFIG_NAME" \
    "checkpoint_dir:=$CKPT_DIR" \
    "${EXTRA[@]}" \
    "${PASSTHRU[@]}"
