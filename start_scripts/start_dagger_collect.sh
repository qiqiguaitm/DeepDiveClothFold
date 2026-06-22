#!/bin/bash
###############################################################################
# kai0 DAgger 数据采集 (thin wrapper)
#
# 复用 start_autonomy_from_ckpt.sh 的 sidecar 校验 + start_autonomy.sh 的全部
# 基础设施 (CAN/相机/GPU/env)。只额外提供 dagger 特有参数 (subset/task/prompt)
# + --dagger 标志让 start_autonomy.sh 切换到 dagger_launch.py (额外加 master_servo
# + dagger_recorder)。
#
# 用法:
#   ./scripts/start_dagger_collect.sh --ckpt <ckpt_dir> [options]
#   ./scripts/start_dagger_collect.sh --ckpt <ckpt_dir> --task Task_A
#   ./scripts/start_dagger_collect.sh --ckpt <ckpt_dir> --no-rerun
###############################################################################

set -eo pipefail

# ── 参数解析 ──
# Default ckpt: dagger 采集默认基线模型 (Task_A pure 200 ep pi0.5 base @ step 49999).
# 通过 --ckpt 显式覆盖.
DEFAULT_CHECKPOINT_DIR="/data1/DATA_IMP/checkpoints/ckpt_v0/task_a_pure200_base_pi05_step49999"
CHECKPOINT_DIR=""
TASK_NAME=""
PROMPT=""
SUBSET="dagger"
# Form C inference-rollout recording (-> <task>/inference/<vN>/<date-vN>/, intervention=0).
# Default ON (2026-06-16): policy rollouts are recorded alongside dagger/ so the
# inference dataset is captured by default; --no-inference records dagger/ only.
RECORD_INFERENCE="true"
CONFIG_NAME=""         # auto from sidecar
EXTRA_ARGS=()          # passed through to start_autonomy.sh
# DAgger sessions are driven via web/dagger_manager (or via the freedrive
# switches + pedal); rerun viz adds GPU + latency contention with no
# operator benefit. Default OFF — re-enable with --rerun.
USE_RERUN="false"

usage() {
    cat <<EOF
Usage: $0 [options]
       $0 --ckpt <checkpoint_dir> [options]

Default checkpoint (used when --ckpt is omitted):
  $DEFAULT_CHECKPOINT_DIR

Options:
  --ckpt <path>      Override default ckpt (must contain train_config.json + _CHECKPOINT_METADATA)
  --task <name>      Task name (Task_A/B/C); empty = infer from --ckpt
  --prompt <str>     Override prompt for tasks.jsonl
  --subset <str>     Subset under <task>/ (default: dagger)
  --config-name <s>  Override base_config_name (default: from sidecar)
  --no-rerun         Disable Rerun viz (DEFAULT — dagger session uses web/dagger_manager)
  --rerun            Force Rerun viz on (override default-off)
  --record-inference Also record policy rollouts to <task>/inference/<vN>/<date-vN>/ (Form C) (DEFAULT)
  --no-inference     Record dagger/ only — no inference dataset
  --mode <ros2|websocket|both>  Inference channel (forwarded)

All other flags are forwarded to start_autonomy.sh.
EOF
    exit 1
}

need_value() {
    if [[ $# -lt 2 ]]; then echo "[FAIL] $1 requires a value" >&2; exit 1; fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt|--checkpoint-dir)  need_value "$@"; CHECKPOINT_DIR="$2"; shift 2 ;;
        --task|--task-name)       need_value "$@"; TASK_NAME="$2";      shift 2 ;;
        --prompt)                 need_value "$@"; PROMPT="$2";          shift 2 ;;
        --config-name)            need_value "$@"; CONFIG_NAME="$2";    shift 2 ;;
        --subset)                 need_value "$@"; SUBSET="$2";          shift 2 ;;
        --rerun)                  USE_RERUN="true"; shift ;;
        --no-rerun)               USE_RERUN="false"; shift ;;
        --record-inference)       RECORD_INFERENCE="true"; shift ;;
        --no-inference)           RECORD_INFERENCE="false"; shift ;;
        -h|--help)                usage ;;
        *)                        EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
    CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
    echo "[info] --ckpt omitted, using default: $CHECKPOINT_DIR" >&2
fi
[[ ! -d "$CHECKPOINT_DIR" ]] && { echo "[FAIL] ckpt dir not found: $CHECKPOINT_DIR" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Sidecar validation + env setup (mirrors start_autonomy_from_ckpt.sh exactly) ──
SIDECAR="$CHECKPOINT_DIR/train_config.json"
[[ ! -f "$SIDECAR" ]] && {
    echo "[FAIL] $SIDECAR not found." >&2
    echo "       Use train_scripts/data/pack_inference_ckpt.py to produce it." >&2
    exit 1
}
[[ ! -f "$CHECKPOINT_DIR/_CHECKPOINT_METADATA" ]] && {
    echo "[FAIL] $CHECKPOINT_DIR/_CHECKPOINT_METADATA missing — invalid ckpt dir." >&2
    exit 1
}

# CRITICAL: openpi reads OPENPI_EXTRA_CONFIG at policy load to pick up
# override_asset_id + per-experiment overrides (norm_stats path, etc.).
# Without this env var, policy uses base-config defaults → different actions.
export OPENPI_EXTRA_CONFIG="$SIDECAR"

# V3 collection (2026-06-15): inherited by dagger_recorder_node via env.
#   KAI0_FRONT_TRIM=1          online leading-idle trim in EpisodeWriter (keeps
#                              15-frame lead-in; same semantics as build_no_release).
#   KAI0_TAIL_TRIM=1           online trailing-idle cap in EpisodeWriter (holds the
#                              post-task idle run, keeps TAIL_CAP=15 terminal settle;
#                              arm AND gripper must be static → final release kept).
#                              Defaults to follow KAI0_FRONT_TRIM.
#   KAI0_GRIPPER_FROM_MASTER=1 action gripper dims (6,13) ← master grasp command;
#                              12 arm dims stay = slave state. Set 0 to opt out.
export KAI0_FRONT_TRIM="${KAI0_FRONT_TRIM:-1}"
export KAI0_TAIL_TRIM="${KAI0_TAIL_TRIM:-$KAI0_FRONT_TRIM}"
export KAI0_GRIPPER_FROM_MASTER="${KAI0_GRIPPER_FROM_MASTER:-1}"
# Async writer: capture thread enqueues, bg thread encodes → no startup stall +
# holds rate. With the cheap demux validate, the writer keeps up → fast finalize.
# KAI0_ASYNC_WRITER=0 reverts to inline encode.
export KAI0_ASYNC_WRITER="${KAI0_ASYNC_WRITER:-1}"
# Per-episode finalize self-check (first-pts==0, frames==rows, first frame
# decodes / not black). Cheap (~0.2s); default ON. KAI0_VALIDATE_TRIM=0 disables.
export KAI0_VALIDATE_TRIM="${KAI0_VALIDATE_TRIM:-1}"
# Dataset CONTENT version → auto-creates a version folder and tags the date leaf.
# Layout: KAI0/<Task>/<subset>/<vN>/<date>-<vN>/ (e.g. Task_A/dagger/v3/2026-06-15-v3).
# dagger_recorder_node inherits this env; the recorder mkdir's the full path, so
# the v2/v3 folder is created on the fly and data lands in its version's subtree.
#   V3 = online front-trim + gripper-action-from-master; else legacy v2.
#   Override the version explicitly with KAI0_DATASET_VERSION=vN.
if [[ "$KAI0_FRONT_TRIM" == "1" && "$KAI0_GRIPPER_FROM_MASTER" != "0" ]]; then
    export KAI0_DATASET_VERSION="${KAI0_DATASET_VERSION:-v3}"
else
    export KAI0_DATASET_VERSION="${KAI0_DATASET_VERSION:-v2}"
fi
export KAI0_DATE_SUFFIX="-${KAI0_DATASET_VERSION}"   # date leaf suffix (layout.py)

# ── V1/v0 inference efficiency (2026-06-15) ──────────────────────────────────
# The dagger infra adds dagger_recorder (PyAV mp4×3 + zarr) + 2× master_servo on
# top of the V1 20 Hz (50 ms) inference loop; on a shared box that contention
# dropped the achieved rate below 20 Hz → RTC ran off-design → different real-
# machine behavior than standalone V1. Two mitigations, both opt-out:
#   1. Head (D435) depth OFF — V1/v0 don't consume depth; the depth grab added
#      USB3 bandwidth + color-frame jitter. KAI0_HEAD_DEPTH=1 re-enables it.
#   2. CPU affinity — pin recorder/servo/cameras to dedicated physical cores
#      (64-thread EPYC; SMT sibling = core+32) so they can't steal the inference
#      cores (those are pinned by start_dagger_session.sh). KAI0_CPU_PIN=0 off.
# These reach dagger_launch.py as launch args (forwarded via start_autonomy.sh
# EXTRA_ARGS). See docs/deployment/inference/dagger_v1_inference_tuning.md.
# Export so the dagger_recorder node (inherits this env) drops top_head from its
# recorded depth set in lockstep with the camera — otherwise the writer would
# keep allocating a top_head depth zarr and fill it with zeros (dataset_writer
# _load_depth_flags honors KAI0_HEAD_DEPTH=0).
export KAI0_HEAD_DEPTH="${KAI0_HEAD_DEPTH:-0}"
HEAD_DEPTH="$([ "$KAI0_HEAD_DEPTH" = "1" ] && echo true || echo false)"
TUNE_ARGS=( "enable_head_depth:=$HEAD_DEPTH" )
if [[ "${KAI0_CPU_PIN:-1}" == "1" ]]; then
    AFF_CAM="${KAI0_AFFINITY_CAMERAS:-12-15,44-47}"
    AFF_REC="${KAI0_AFFINITY_RECORDER:-16-23,48-55}"
    AFF_SRV="${KAI0_AFFINITY_SERVO:-24-27,56-59}"
    REC_NICE="${KAI0_RECORDER_NICE:-10}"
    TUNE_ARGS+=(
        "camera_cpu_prefix:=taskset -c $AFF_CAM"
        "recorder_cpu_prefix:=nice -n $REC_NICE ionice -c2 -n7 taskset -c $AFF_REC"
        "servo_cpu_prefix:=taskset -c $AFF_SRV"
    )
fi

if [[ -z "$CONFIG_NAME" ]]; then
    CONFIG_NAME=$(/data1/miniconda3/bin/python -c \
        "import json; print(json.load(open('$SIDECAR'))['base_config_name'])")
fi
ASSET_ID=$(/data1/miniconda3/bin/python -c \
    "import json; print(json.load(open('$SIDECAR')).get('override_asset_id', ''))")

if [[ -n "$ASSET_ID" ]] && [[ ! -f "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" ]]; then
    echo "[FAIL] $CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json missing (override_asset_id mismatch)" >&2
    exit 1
fi

echo "============================================================"
echo " kai0 DAgger Collection (delegates to start_autonomy.sh --dagger)"
echo " checkpoint : $CHECKPOINT_DIR"
echo " task       : ${TASK_NAME:-<infer-from-ckpt>}"
echo " subset     : $SUBSET"
echo " leaf suffix: $KAI0_DATE_SUFFIX (<task>/<subset>/<date>$KAI0_DATE_SUFFIX; head_depth=$KAI0_HEAD_DEPTH)"
echo " trim       : front=$KAI0_FRONT_TRIM tail=$KAI0_TAIL_TRIM (record-time leading + trailing idle cap, keep 15-frame settle)"
echo " depth fmt  : packed 1 file/episode (.zarr.zip) — EpisodeWriter.finalize auto-packs the zarr dir"
echo " inference  : $([ "$RECORD_INFERENCE" = "true" ] && echo 'ON (Form C: dagger/ + inference/)' || echo 'OFF (dagger/ only)')"
echo " prompt     : ${PROMPT:-<infer-from-ckpt>}"
echo " config     : $CONFIG_NAME"
echo " asset_id   : ${ASSET_ID:-<none>}"
echo " OPENPI_EXTRA_CONFIG : $SIDECAR"
echo "============================================================"
echo ""

# Build dagger-specific launch args for dagger_launch.py
DAGGER_ARGS=("record_subset:=$SUBSET" "record_inference:=$RECORD_INFERENCE")
[[ -n "$TASK_NAME" ]] && DAGGER_ARGS+=("record_task:=$TASK_NAME")
[[ -n "$PROMPT" ]] && DAGGER_ARGS+=("record_prompt:=$PROMPT" "prompt:=$PROMPT")

# Delegate to start_autonomy.sh with --dagger flag.
# start_autonomy.sh handles: CAN activation, USB camera reset, GPU selection,
# venv/PATH/PYTHONPATH setup, deployment marker. With --dagger, it uses
# dagger_launch.py which IncludeLaunchDescription's autonomy_launch.py + adds
# 2× master_servo + dagger_recorder.
RERUN_FLAG=()
if [[ "$USE_RERUN" != "true" ]]; then
    # Important: this MUST land before EXTRA_ARGS so that the user can still
    # override with --rerun without us silently re-disabling it. start_autonomy.sh
    # passes enable_rerun:="$ENABLE_RERUN" to dagger_launch.py; dagger_launch.py's
    # IncludeLaunchDescription override is masked by that CLI arg, so we have to
    # set ENABLE_RERUN false at the start_autonomy.sh layer.
    RERUN_FLAG=(--no-rerun)
fi

# ── dagger_manager web (lifecycle bundled with this script) ──
# Web UI launches alongside the ROS2 infra and dies when this script exits.
# Avoids the failure mode the user observed: ./run.sh start leaving uvicorn +
# vite running for hours after a dagger session was Ctrl-C'd, holding ports
# 8788/5174 and serving a stale snapshot.
#
# Skip with SKIP_WEB=1 if you want to manage the web manually (e.g. dev mode
# with hot-reload). The web still works fine when launched separately.
WEB_DIR="$PROJECT_ROOT/web/dagger_manager"
WEB_RUN="$WEB_DIR/run.sh"

stop_web() {
    [[ "${SKIP_WEB:-0}" == "1" ]] && return
    echo "[dagger] stopping web..." >&2
    if [[ -x "$WEB_RUN" ]]; then
        # </dev/null = don't share our stdin; >/dev/null 2>&1 = silent;
        # || true = don't propagate exit code (we're shutting down anyway).
        bash "$WEB_RUN" stop </dev/null >/dev/null 2>&1 || true
    fi
    # Belt-and-suspenders: port-level kill in case run.sh missed (pidfile
    # race, SIGKILL'd parent leaving orphans). ss is reliable even when
    # bash is mid-signal-handler.
    local port pids
    for port in 8788 5174; do
        pids=$(ss -lntp 2>/dev/null \
            | awk -v p=":$port\$" '$4 ~ p' \
            | grep -oP 'pid=\K[0-9]+' \
            | sort -u)
        [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null || true
    done
}

if [[ "${SKIP_WEB:-0}" != "1" ]] && [[ -x "$WEB_RUN" ]]; then
    echo "[dagger] starting web (background)..."
    bash "$WEB_RUN" start
fi

# Trap covers every exit path: EXIT (normal end), INT (Ctrl-C), TERM
# (kill from another script), HUP (terminal closed). Without HUP, closing
# the SSH session would orphan the web.
trap stop_web EXIT INT TERM HUP

# Foreground call — NOT `exec` so our trap fires when start_autonomy.sh
# returns. `|| true` neutralizes set -e for the ros2-launch-via-SIGINT exit
# code (130); without it, bash could short-circuit out before the trap chain
# completes when the user runs us inside a pipeline like `| tee log`.
# start_autonomy.sh was relocated to start_scripts/kai/. Resolve it there,
# falling back to the old flat location for older checkouts.
AUTONOMY_SH="$SCRIPT_DIR/kai/start_autonomy.sh"
[[ -x "$AUTONOMY_SH" ]] || AUTONOMY_SH="$SCRIPT_DIR/start_autonomy.sh"

# TUNE_ARGS (head-depth + cpu-affinity) land before EXTRA_ARGS so an explicit
# user override on the CLI still wins (ros2 launch: last key:=value wins).
"$AUTONOMY_SH" --dagger \
    "${RERUN_FLAG[@]}" \
    "config_name:=$CONFIG_NAME" \
    "checkpoint_dir:=$CHECKPOINT_DIR" \
    "${DAGGER_ARGS[@]}" \
    "${TUNE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    || true

# Defensive: explicit cleanup in case neither EXIT nor INT trap fires
# (observed on bash 5 under `| tee` with set -eo pipefail). Idempotent
# with the trap so calling twice is fine.
stop_web
