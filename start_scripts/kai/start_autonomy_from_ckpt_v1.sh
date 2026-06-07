#!/bin/bash
# V1 Triton 版本: 从原始 JAX orbax ckpt dir 推断对应 v1_p200.pkl + delta 模式 +
# norm_stats, 然后启动 V1 stack (SHM transport, 20Hz, A.2 流水线).
#
# 使用:
#   ./start_scripts/kai/start_autonomy_from_ckpt_v1.sh <ckpt_dir> [其他 ROS args...]
#
# 要求:
#   <ckpt_dir>/train_config.json 存在 ({"base_config_name": "...", "override_asset_id": "..."})
#   <ckpt_dir>/_CHECKPOINT_METADATA 存在 (orbax meta)
#   <ckpt_dir>/assets/<override_asset_id>/norm_stats.json 存在
#   optimize/results/<ckpt_dir 名>_v1_p200.pkl 存在 (convert_kai0_to_v1 + expand 产物)
#     若缺失, 脚本会提示如何生成.
#
# Delta 模式自动检测: train_config.json 的 base_config_name 含 "delta" 字样,
# 或 kai0/src/openpi/training/config.py 中该 config 标记 use_delta_joint_actions=True.

set -euo pipefail

CKPT_DIR="${1:?Usage: $0 <ckpt_dir> [extra autonomy_v1 args...]}"
shift || true

if [ ! -d "$CKPT_DIR" ]; then
  echo "[FAIL] ckpt_dir not found: $CKPT_DIR" >&2
  exit 1
fi
if [ ! -f "$CKPT_DIR/train_config.json" ]; then
  echo "[FAIL] $CKPT_DIR/train_config.json missing — 不是 pack_inference_ckpt.py 产物?" >&2
  exit 1
fi
# JAX-sourced v1 ckpt 带 _CHECKPOINT_METADATA (orbax); PyTorch-sourced v1 ckpt
# (convert_pytorch_safetensors_to_v1.py 产物) 带 metadata.pt。二者之一即有效。
if [ ! -f "$CKPT_DIR/_CHECKPOINT_METADATA" ] && [ ! -f "$CKPT_DIR/metadata.pt" ]; then
  echo "[FAIL] neither $CKPT_DIR/_CHECKPOINT_METADATA (JAX) nor $CKPT_DIR/metadata.pt (PyTorch) found — invalid ckpt dir" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/data1/miniconda3/bin/python

# 解析 train_config.json
BASE_CONFIG=$($PY -c "import json; print(json.load(open('$CKPT_DIR/train_config.json'))['base_config_name'])")
ASSET_ID=$($PY -c "import json; d=json.load(open('$CKPT_DIR/train_config.json')); print(d.get('override_asset_id', ''))")
NORM_STATS="$CKPT_DIR/assets/$ASSET_ID/norm_stats.json"

if [ ! -f "$NORM_STATS" ]; then
  echo "[FAIL] norm_stats not found: $NORM_STATS" >&2
  echo "       check override_asset_id=$ASSET_ID matches ckpt assets/" >&2
  exit 1
fi

# Delta 自动检测: base_config 名字含 'delta' 或显式查 config.py
DELTA_FLAG=""
if [[ "$BASE_CONFIG" == *delta* ]]; then
  DELTA_FLAG="--delta"
fi
# 另: 兜底查 use_delta_joint_actions=True (config.py 里的标记)
if [ -z "$DELTA_FLAG" ]; then
  IS_DELTA=$($PY -c "
import sys; sys.path.insert(0, '$REPO/kai0/src')
try:
    from openpi.training import config as _cfg
    cfg = _cfg.get_config('$BASE_CONFIG')
    print(getattr(cfg.data, 'use_delta_joint_actions', False))
except Exception:
    print('False')
" 2>/dev/null || echo "False")
  if [ "$IS_DELTA" = "True" ]; then
    DELTA_FLAG="--delta"
  fi
fi

# 自动找 v1 pickle. 优先级:
#   1) 新版 layout: <ckpt_dir>/v1_p200.pkl  (ckpt_v1 自包含, vis_5day_recent 起的约定)
#   2) 旧版 layout: optimize/results/<basename>_v1_p200.pkl  (vis_v2_full / mixed_1 等)
CKPT_BASENAME=$(basename "$CKPT_DIR")
V1_PKL="$CKPT_DIR/v1_p200.pkl"
if [ ! -f "$V1_PKL" ]; then
  V1_PKL="$REPO/optimize/results/${CKPT_BASENAME}_v1_p200.pkl"
fi

if [ ! -f "$V1_PKL" ]; then
  echo "[FAIL] V1 pickle not found. 试过两个位置:" >&2
  echo "       - $CKPT_DIR/v1_p200.pkl  (新版自包含 layout)" >&2
  echo "       - $REPO/optimize/results/${CKPT_BASENAME}_v1_p200.pkl  (旧版 layout)" >&2
  echo "       需要先转换:" >&2
  echo "       $REPO/kai0/.venv_5090_trt/bin/python $REPO/optimize/v1_triton/convert_kai0_to_v1.py \\" >&2
  echo "           --jax_path <ckpt_v0_dir> \\" >&2
  echo "           --output $REPO/optimize/results/${CKPT_BASENAME}_v1.pkl \\" >&2
  echo "           --prompt \"\${PROMPT_FROM_TRAIN_CONFIG}\" \\" >&2
  echo "           --tokenizer_model $REPO/openpi_cache/big_vision/paligemma_tokenizer.model" >&2
  echo "       $REPO/kai0/.venv_5090_trt/bin/python $REPO/optimize/v1_triton/expand_v1_pkl_for_phase2.py \\" >&2
  echo "           --in $REPO/optimize/results/${CKPT_BASENAME}_v1.pkl \\" >&2
  echo "           --out $CKPT_DIR/v1_p200.pkl" >&2
  exit 1
fi

echo "============================================================"
echo "  start_autonomy_from_ckpt_v1.sh"
echo "    ckpt_dir:    $CKPT_DIR"
echo "    base_config: $BASE_CONFIG"
echo "    asset_id:    $ASSET_ID"
echo "    norm_stats:  $NORM_STATS"
echo "    v1_pkl:      $V1_PKL"
echo "    delta:       ${DELTA_FLAG:-(no, absolute action mode)}"
echo "============================================================"

# 透传 args: --execute / --rerun / --no-rerun / 等其他 autonomy_v1 args
exec "$REPO/start_scripts/kai/start_autonomy_v1.sh" \
  --pkl "$V1_PKL" \
  --norm "$NORM_STATS" \
  $DELTA_FLAG \
  "$@"
