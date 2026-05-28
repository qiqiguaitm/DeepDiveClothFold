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
if [ ! -f "$CKPT_DIR/_CHECKPOINT_METADATA" ]; then
  echo "ERROR: $CKPT_DIR/_CHECKPOINT_METADATA missing — invalid ckpt dir." >&2
  exit 1
fi

export OPENPI_EXTRA_CONFIG="$CKPT_DIR/train_config.json"
CONFIG_NAME=$(/data1/miniconda3/bin/python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json'))['base_config_name'])")
ASSET_ID=$(/data1/miniconda3/bin/python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('override_asset_id', ''))")

if [ -n "$ASSET_ID" ] && [ ! -f "$CKPT_DIR/assets/$ASSET_ID/norm_stats.json" ]; then
  echo "ERROR: $CKPT_DIR/assets/$ASSET_ID/norm_stats.json missing (override_asset_id mismatch)" >&2
  exit 1
fi

echo "[start_autonomy_from_ckpt] config_name=$CONFIG_NAME asset_id=$ASSET_ID ckpt_dir=$CKPT_DIR"

cd "$(dirname "$0")/../.."
exec ./start_scripts/kai/start_autonomy.sh --execute \
  "config_name:=$CONFIG_NAME" \
  "checkpoint_dir:=$CKPT_DIR" \
  "$@"
