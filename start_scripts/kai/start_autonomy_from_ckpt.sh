#!/bin/bash
# 通用 autonomy 启动脚本: 直接从 ckpt 内的 train_config.json 加载配置, 不需修改 src/openpi/training/config.py.
#
# 使用:
#   ./start_scripts/kai/start_autonomy_from_ckpt.sh <ckpt_dir> [其他 ROS args...]
#
# 要求 <ckpt_dir>/train_config.json 存在, 形如:
#   { "base_config_name": "...", "override_asset_id": "..." }
#
# 且 ckpt 自带 norm_stats: <ckpt_dir>/assets/<override_asset_id>/norm_stats.json

set -euo pipefail

CKPT_DIR="${1:?Usage: $0 <ckpt_dir> [extra ROS args...]}"
shift || true

if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: ckpt_dir does not exist: $CKPT_DIR" >&2
  exit 1
fi
if [ ! -f "$CKPT_DIR/train_config.json" ]; then
  echo "ERROR: $CKPT_DIR/train_config.json not found." >&2
  echo "       Use train_scripts/kai/data/pack_inference_ckpt.py to produce it." >&2
  exit 1
fi
# JAX (orbax) ckpt 自带 _CHECKPOINT_METADATA; PyTorch (safetensors) ckpt 自带 metadata.pt。
# 二者之一存在即认为是有效 ckpt 目录 (向后兼容: 旧 JAX ckpt 仍走 _CHECKPOINT_METADATA)。
if [ ! -f "$CKPT_DIR/_CHECKPOINT_METADATA" ] && [ ! -f "$CKPT_DIR/metadata.pt" ]; then
  echo "ERROR: neither $CKPT_DIR/_CHECKPOINT_METADATA (JAX) nor $CKPT_DIR/metadata.pt (PyTorch) found — invalid ckpt dir." >&2
  exit 1
fi

export OPENPI_EXTRA_CONFIG="$CKPT_DIR/train_config.json"
CONFIG_NAME=$(/data1/miniconda3/bin/python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json'))['base_config_name'])")
ASSET_ID=$(/data1/miniconda3/bin/python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('override_asset_id', ''))")

if [ -n "$ASSET_ID" ] && [ ! -f "$CKPT_DIR/assets/$ASSET_ID/norm_stats.json" ]; then
  echo "ERROR: $CKPT_DIR/assets/$ASSET_ID/norm_stats.json missing (override_asset_id mismatch)" >&2
  exit 1
fi

# Optional sidecar key for domain-conditioned models (action_head_cond_num_domains>0):
# "deploy_dataset_id": <int> (e.g. 1=vis) → forces the per-domain token at inference.
# Absent (old sidecars / plain pi05) → no dataset_id:= arg, launch default -1 (disabled). Backward-compatible.
DATASET_ID=$(/data1/miniconda3/bin/python -c "import json; v=json.load(open('$CKPT_DIR/train_config.json')).get('deploy_dataset_id'); print('' if v is None else int(v))")

# Optional sidecar key for prompt-conditioned models (e.g. AWBC: advantage carried in the
# text prompt, not a domain token). "deploy_prompt": "<str>" → overrides the launch prompt
# default so the model's expected prompt (e.g. "...Advantage: positive") travels with the ckpt.
# Absent (plain pi05) → no prompt:= arg, launch default "Flatten and fold the cloth.". Backward-compatible.
DEPLOY_PROMPT=$(/data1/miniconda3/bin/python -c "import json; v=json.load(open('$CKPT_DIR/train_config.json')).get('deploy_prompt'); print('' if v is None else v)")

# Publish-time EMA smoothing (Layer 1.1E): α∈(0,1], cmd[t]=α·cmd+(1-α)·last_pub.
# The node default is 1.0 (OFF) — without this arg the cmd timeline is unsmoothed
# (this is why setting it elsewhere "had no effect": it was never reaching here,
# so EMA stayed off). Default 0.5 = moderate low-pass attenuating cmd noise above
# the Piper PD low-pass bandwidth. Grippers (j6,j13) are auto-excluded in-node.
# Override: KAI0_PUBLISH_SMOOTH_ALPHA=1.0 disables; any (0,1] sets the factor.
PUBLISH_SMOOTH_ALPHA="${KAI0_PUBLISH_SMOOTH_ALPHA:-0.5}"

# Deploy-time gripper frame remap (old 100mm-range ckpt → real 0–70mm robot).
# 默认开(本机已做官方 0–70mm 夹爪标定, 现测的多是旧 frame ckpt)。把夹爪 norm_stats
# 按其训练范围 [q01,q99] 重映射到真机范围(默认 0.0,0.07 m)。部署新 frame ckpt 时设 =0 关。
# 见 docs/deployment/data_collection/gripper_calibration.md
export KAI0_GRIPPER_DEPLOY_REMAP="${KAI0_GRIPPER_DEPLOY_REMAP:-1}"
export KAI0_GRIPPER_REAL_RANGE="${KAI0_GRIPPER_REAL_RANGE:-0.0,0.07}"
[ "$KAI0_GRIPPER_DEPLOY_REMAP" = "1" ] && echo "[gripper-remap] ON: 夹爪 norm_stats [q01,q99]→真机[$KAI0_GRIPPER_REAL_RANGE]m (dims 6,13)"

echo "[start_autonomy_from_ckpt] config_name=$CONFIG_NAME asset_id=$ASSET_ID dataset_id=${DATASET_ID:-<none>} prompt=${DEPLOY_PROMPT:-<default>} publish_smooth_alpha=$PUBLISH_SMOOTH_ALPHA ckpt_dir=$CKPT_DIR"

cd "$(dirname "$0")/../.."
EXTRA_ARGS=()
[ -n "$DATASET_ID" ] && EXTRA_ARGS+=("dataset_id:=$DATASET_ID")
[ -n "$DEPLOY_PROMPT" ] && EXTRA_ARGS+=("prompt:=$DEPLOY_PROMPT")
# Before "$@" so an explicit CLI publish_smooth_alpha:=… still wins (ros2: last wins).
EXTRA_ARGS+=("publish_smooth_alpha:=$PUBLISH_SMOOTH_ALPHA")
exec ./start_scripts/kai/start_autonomy.sh --execute \
  "config_name:=$CONFIG_NAME" \
  "checkpoint_dir:=$CKPT_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
