#!/bin/bash
###############################################################################
# kai0 DAgger session launcher — policy half only (infra already up).
#
# Companion to start_dagger_collect.sh (which brings up infra-only because
# dagger_launch passes enable_policy:=false). Forks just the policy, safe to
# call from web/dagger_manager backend on every "Start session" click.
#
# Two ckpt variants (auto-detected from the path, or forced with --variant):
#
#   v0  (ckpt_v0/*)  JAX orbax ckpt, in-process inference.
#                    → session_launch.py mode:=ros2 (loads JAX, ~22s).
#                    Mirrors start_autonomy_from_ckpt.sh.
#
#   v1  (ckpt_v1/*)  V1 Triton path. Needs a v1_p200.pkl (self-contained at
#                    <ckpt>/v1_p200.pkl, or optimize/results/<name>_v1_p200.pkl).
#                    → start_serve_v1.sh (:8002) + session_launch.py
#                      mode:=websocket client, V1 RTC tuning (20Hz/k=6/shm).
#                    Mirrors start_autonomy_from_ckpt_v1.sh (minus cameras/arms,
#                    which the dagger infra already owns).
#
# Idempotency: assumes ROS2 / CAN / cameras / arms / dagger_recorder are
# already up. Without infra, policy_inference spins on "Waiting for sensor
# data…" forever.
#
# Usage:
#   ./start_dagger_session.sh --ckpt <path>                 # auto v0/v1
#   ./start_dagger_session.sh --ckpt <path> --variant v1    # force v1
#   ./start_dagger_session.sh --ckpt <path> --gpu 1         # CUDA_VISIBLE_DEVICES=1
###############################################################################
set -eo pipefail

CHECKPOINT_DIR=""
GPU_ID=""             # empty = auto-select GPU with most free memory (see below)
VARIANT="auto"        # v0 | v1 | auto
SERVE_PORT="8002"     # V1 serve port (v1 only)
CONFIG_OVERRIDE=""
PROMPT_OVERRIDE=""
EXTRA=()

usage() {
    cat <<EOF
Usage: $0 --ckpt <checkpoint_dir> [options]
Options:
  --ckpt <path>          Packed ckpt dir (train_config.json + assets/norm_stats.json)
  --gpu <id>             CUDA_VISIBLE_DEVICES (default: auto-pick GPU with most free VRAM)
  --variant <v0|v1|auto> Inference path (default auto: ckpt_v1/* → v1, else v0)
  --serve-port <port>    V1 serve_policy_v1.py port (v1 only, default 8002)
  --config-name <name>   Override base_config_name (v0 only; default: from sidecar)
  --prompt <text>        Override prompt (default: from sidecar / config default)
All remaining args forwarded to ros2 launch session_launch.py.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt|--checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --gpu|--gpu-id)          GPU_ID="$2"; shift 2 ;;
        --variant)               VARIANT="$2"; shift 2 ;;
        --serve-port)            SERVE_PORT="$2"; shift 2 ;;
        --config-name)           CONFIG_OVERRIDE="$2"; shift 2 ;;
        --prompt)                PROMPT_OVERRIDE="$2"; shift 2 ;;
        -h|--help)               usage ;;
        *)                        EXTRA+=("$1"); shift ;;
    esac
done

[[ -z "$CHECKPOINT_DIR" ]] && { echo "[FAIL] --ckpt required" >&2; exit 1; }
[[ ! -d "$CHECKPOINT_DIR" ]] && { echo "[FAIL] ckpt dir not found: $CHECKPOINT_DIR" >&2; exit 1; }

# ── GPU auto-select (default when --gpu omitted) ──────────────────────────
# Web "Start session" forks us without --gpu, so historically GPU_ID hardcoded
# to 0 — the busiest card on sim01 (shared with other users' jobs). Combined
# with the old MEM_FRACTION=0.9 preallocation that made the load hang. Now pick
# the GPU with the MOST free VRAM (tiebreak: higher index, per request — leaves
# the lower-numbered cards for others). KAI0_GPU_ID=<n> or --gpu <n> override.
if [[ -z "$GPU_ID" ]]; then
    if [[ -n "${KAI0_GPU_ID:-}" ]]; then
        GPU_ID="$KAI0_GPU_ID"
        echo "[gpu] KAI0_GPU_ID override → GPU $GPU_ID" >&2
    else
        GPU_ID=$(nvidia-smi --query-gpu=index,memory.free \
                     --format=csv,noheader,nounits 2>/dev/null \
                 | awk -F',' '{i=$1+0; f=$2+0; if (f>bf || (f==bf && i>bi)) {bf=f; bi=i}} END{print bi}')
        GPU_ID=${GPU_ID:-0}
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null || echo "?")
        echo "[gpu] auto-selected GPU $GPU_ID (most free VRAM: ${FREE_MB}MB)" >&2
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY=/data1/miniconda3/bin/python

# ── Common sidecar validation (both variants need the orbax meta + sidecar) ──
SIDECAR="$CHECKPOINT_DIR/train_config.json"
[[ ! -f "$SIDECAR" ]] && { echo "[FAIL] $SIDECAR missing (pack_inference_ckpt.py?)" >&2; exit 1; }
[[ ! -f "$CHECKPOINT_DIR/_CHECKPOINT_METADATA" ]] && {
    echo "[FAIL] $CHECKPOINT_DIR/_CHECKPOINT_METADATA missing" >&2; exit 1;
}

CONFIG_NAME=$($PY -c "import json; print(json.load(open('$SIDECAR'))['base_config_name'])")
ASSET_ID=$($PY -c "import json; print(json.load(open('$SIDECAR')).get('override_asset_id', ''))")
SIDECAR_PROMPT=$($PY -c \
    "import json; c=json.load(open('$SIDECAR')); print(c.get('prompt') or c.get('default_prompt') or '')" \
    2>/dev/null || echo "")
# CLI override wins; otherwise whatever the sidecar carries (may be empty →
# downstream defaults apply). NOTE: do NOT lowercase — PaligemmaTokenizer
# training keeps capitals; a lowercased prompt silently breaks narrow ckpts.
EFFECTIVE_PROMPT="${PROMPT_OVERRIDE:-$SIDECAR_PROMPT}"

NORM_STATS="$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json"
if [[ -n "$ASSET_ID" ]] && [[ ! -f "$NORM_STATS" ]]; then
    echo "[FAIL] $NORM_STATS missing (override_asset_id mismatch)" >&2
    exit 1
fi

# ── Resolve variant ──────────────────────────────────────────────────────
if [[ "$VARIANT" == "auto" ]]; then
    if [[ "$CHECKPOINT_DIR" == */ckpt_v1/* ]]; then VARIANT="v1"; else VARIANT="v0"; fi
fi
if [[ "$VARIANT" != "v0" && "$VARIANT" != "v1" ]]; then
    echo "[FAIL] --variant must be v0|v1|auto (got '$VARIANT')" >&2; exit 1
fi

# ROS2 source + workspace (idempotent — already done by infra, redo for safety).
source /opt/ros/jazzy/setup.bash
source "$PROJECT_ROOT/ros2_ws/install/setup.bash"

###############################################################################
# v0 — JAX in-process (mode=ros2). OPENPI_EXTRA_CONFIG is the contract: the
# JAX loader reads asset_id + norm_stats overrides from it.
###############################################################################
if [[ "$VARIANT" == "v0" ]]; then
    [[ -n "$CONFIG_OVERRIDE" ]] && CONFIG_NAME="$CONFIG_OVERRIDE"
    export OPENPI_EXTRA_CONFIG="$SIDECAR"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    echo "============================================================"
    echo " kai0 DAgger Session (v0 / JAX in-process)"
    echo "  ckpt:    $CHECKPOINT_DIR"
    echo "  config:  $CONFIG_NAME"
    echo "  asset:   ${ASSET_ID:-<none>}"
    echo "  prompt:  ${EFFECTIVE_PROMPT:-<config default>}"
    echo "  gpu:     $GPU_ID"
    echo "============================================================"

    LAUNCH_ARGS=(
        "checkpoint_dir:=$CHECKPOINT_DIR"
        "config_name:=$CONFIG_NAME"
        "gpu_id:=$GPU_ID"
        "mode:=ros2"
        "execute_mode:=true"
    )
    [[ -n "$EFFECTIVE_PROMPT" ]] && LAUNCH_ARGS+=("prompt:=$EFFECTIVE_PROMPT")
    exec ros2 launch piper session_launch.py "${LAUNCH_ARGS[@]}" "${EXTRA[@]}"
fi

###############################################################################
# v1 — V1 Triton serve (:SERVE_PORT) + websocket client. Mirrors
# start_autonomy_from_ckpt_v1.sh, minus cameras/arms (dagger infra owns them).
###############################################################################

# Resolve v1 pickle: self-contained layout first, then legacy optimize/results.
CKPT_BASENAME=$(basename "$CHECKPOINT_DIR")
V1_PKL="$CHECKPOINT_DIR/v1_p200.pkl"
if [[ ! -f "$V1_PKL" ]]; then
    V1_PKL="$PROJECT_ROOT/optimize/results/${CKPT_BASENAME}_v1_p200.pkl"
fi
if [[ ! -f "$V1_PKL" ]]; then
    echo "[FAIL] V1 pickle not found. Tried:" >&2
    echo "       - $CHECKPOINT_DIR/v1_p200.pkl  (self-contained layout)" >&2
    echo "       - $PROJECT_ROOT/optimize/results/${CKPT_BASENAME}_v1_p200.pkl" >&2
    echo "       Convert first with optimize/v1_triton/convert_kai0_to_v1.py +" >&2
    echo "       expand_v1_pkl_for_phase2.py (see start_autonomy_from_ckpt_v1.sh)." >&2
    exit 1
fi
[[ -z "$ASSET_ID" || ! -f "$NORM_STATS" ]] && {
    echo "[FAIL] v1 needs norm_stats: $NORM_STATS (override_asset_id=$ASSET_ID)" >&2; exit 1;
}

# Delta auto-detection: base_config name contains 'delta', or config.py marks
# use_delta_joint_actions=True.
DELTA_FLAG=""
if [[ "$CONFIG_NAME" == *delta* ]]; then
    DELTA_FLAG="--delta-joint-actions"
else
    IS_DELTA=$($PY -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT/kai0/src')
try:
    from openpi.training import config as _cfg
    print(getattr(_cfg.get_config('$CONFIG_NAME').data, 'use_delta_joint_actions', False))
except Exception:
    print('False')
" 2>/dev/null || echo "False")
    [[ "$IS_DELTA" == "True" ]] && DELTA_FLAG="--delta-joint-actions"
fi

SERVE_SH="$PROJECT_ROOT/start_scripts/kai/start_serve_v1.sh"
[[ ! -x "$SERVE_SH" && ! -f "$SERVE_SH" ]] && { echo "[FAIL] $SERVE_SH missing" >&2; exit 1; }
SERVE_LOG="/tmp/dagger_v1_serve.log"
SERVE_PID=""

kill_serve_port() {
    local pids
    # `|| true` is REQUIRED: under `set -eo pipefail`, an empty port means
    # grep matches nothing → exits 1 → pipefail propagates → the command
    # substitution returns 1 → the assignment fails → set -e kills the whole
    # script. On a fresh start the port is always empty, so without this the
    # session dies silently right after printing the header.
    pids=$(ss -lntp 2>/dev/null | awk -v p=":$SERVE_PORT\$" '$4 ~ p' \
           | grep -oP 'pid=\K[0-9]+' | sort -u || true)
    [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null || true
}

cleanup_v1() {
    [[ -n "$SERVE_PID" ]] && kill -INT "$SERVE_PID" 2>/dev/null || true
    sleep 1
    [[ -n "$SERVE_PID" ]] && kill -KILL "$SERVE_PID" 2>/dev/null || true
    # Port-level backup in case the python re-parented away from $SERVE_PID.
    kill_serve_port
}
trap cleanup_v1 EXIT INT TERM

echo "============================================================"
echo " kai0 DAgger Session (v1 / Triton serve + websocket client)"
echo "  ckpt:        $CHECKPOINT_DIR"
echo "  config:      $CONFIG_NAME"
echo "  v1_pkl:      $V1_PKL"
echo "  norm_stats:  $NORM_STATS"
echo "  delta:       ${DELTA_FLAG:-(no, absolute action mode)}"
echo "  serve_port:  $SERVE_PORT"
echo "  prompt:      ${EFFECTIVE_PROMPT:-<serve default>}"
echo "  gpu:         $GPU_ID"
echo "============================================================"

# Clear any stale serve squatting on the port (e.g. a crashed prior session).
kill_serve_port

# ── Step 1: V1 serve ──────────────────────────────────────────────────────
echo "[1/2] launching V1 serve on :$SERVE_PORT (log: $SERVE_LOG) ..."
SERVE_ARGS=(--port "$SERVE_PORT" --pkl "$V1_PKL" --norm "$NORM_STATS")
[[ -n "$EFFECTIVE_PROMPT" ]] && SERVE_ARGS+=(--prompt "$EFFECTIVE_PROMPT")
[[ -n "$DELTA_FLAG" ]] && SERVE_ARGS+=("$DELTA_FLAG")
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$SERVE_SH" "${SERVE_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

SERVE_READY=false
for i in $(seq 1 90); do
    if curl -s --max-time 1 "http://localhost:${SERVE_PORT}/healthz" 2>/dev/null | grep -q "OK"; then
        echo "[1/2] V1 serve healthy after ${i}s"
        SERVE_READY=true
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[1/2] [FAIL] V1 serve process died — last 20 lines of $SERVE_LOG:" >&2
        tail -20 "$SERVE_LOG" >&2 2>&1 || true
        exit 1
    fi
    sleep 1
done
if [[ "$SERVE_READY" != "true" ]]; then
    echo "[1/2] [FAIL] V1 serve not ready within 90s; see $SERVE_LOG" >&2
    exit 1
fi

# ── Step 2: policy_inference_node (websocket client → :SERVE_PORT) ──────────
# V1 production RTC tuning (from start_autonomy_v1.sh, 2026-05-25 retune).
# Camera knobs (cam_fps/depth) are NOT set here — the dagger infra already
# owns the cameras; this launch only spawns policy_inference_node.
echo "[2/2] launching policy_inference (websocket client) ..."
export CUDA_VISIBLE_DEVICES="$GPU_ID"
LAUNCH_ARGS=(
    "mode:=websocket"
    "host:=localhost"
    "port:=$SERVE_PORT"
    "gpu_id:=$GPU_ID"
    "config_name:=pi05_flatten_fold_normal"
    "execute_mode:=true"
    "inference_rate:=20.0"
    "latency_k:=6"
    "min_smooth_steps:=8"
    "rtc_execute_horizon:=12"
    "publish_rate:=80"
    "transport:=shm"
    "fast_obs_pipeline:=true"
    "pipelined_obs:=true"
)
[[ -n "$EFFECTIVE_PROMPT" ]] && LAUNCH_ARGS+=("prompt:=$EFFECTIVE_PROMPT")

# Foreground (NOT exec) so the EXIT trap can tear down the serve. When the
# dagger_manager SessionManager kills our process group (SIGINT), ros2 launch
# shuts down the policy node, this returns, and cleanup_v1 reaps the serve.
ros2 launch piper session_launch.py "${LAUNCH_ARGS[@]}" "${EXTRA[@]}"
