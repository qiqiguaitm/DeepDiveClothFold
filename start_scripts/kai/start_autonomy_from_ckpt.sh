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

echo "[start_autonomy_from_ckpt] config_name=$CONFIG_NAME asset_id=$ASSET_ID dataset_id=${DATASET_ID:-<none>} ckpt_dir=$CKPT_DIR"

cd "$(dirname "$0")/../.."
EXTRA_ARGS=()
[ -n "$DATASET_ID" ] && EXTRA_ARGS+=("dataset_id:=$DATASET_ID")
exec ./start_scripts/kai/start_autonomy.sh --execute \
  "config_name:=$CONFIG_NAME" \
  "checkpoint_dir:=$CKPT_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
