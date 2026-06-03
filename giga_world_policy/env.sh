#!/usr/bin/env bash
# GigaWorld-Policy 训练环境便捷加载脚本(deepdive_kai0 项目内副本)
# 用法:  source /mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy/env.sh
# 作用:  激活共享 venv + 缓存/临时目录指向大盘 + HF 镜像 + 导出本项目权重/数据路径
#
# 与 /mnt/pfs/p46h4f/cosmos/giga-world-policy/env.sh 的区别:
#   - GWP_HOME 指向本项目 (deepdive_kai0/giga_world_policy)
#   - 权重/数据路径指向本项目同级目录 (../checkpoints, ../kai0/data/wam_fold_v1)
#   - 复用同一个已装好的 venv (torch 2.6 + av1 + giga-*),但 giga-datasets 需 editable
#     重装为本仓库版本(见末尾提示),否则 config 的 embodiment= 参数不生效。

# ---------- repo root (本脚本所在目录) ----------
export GWP_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GWP_VENV=/mnt/pfs/p46h4f/cosmos/.venv
# shellcheck disable=SC1091
source "$GWP_VENV/bin/activate"

# ---------- 缓存/临时目录放大盘 (根盘 overlay 仅 ~40G, 别撑爆) ----------
export UV_CACHE_DIR=/mnt/pfs/p46h4f/cosmos/.uv_cache
export UV_PYTHON_INSTALL_DIR=/mnt/pfs/p46h4f/cosmos/.uv_python
export TMPDIR=/mnt/pfs/p46h4f/cosmos/.tmp
export HF_HOME=/mnt/pfs/p46h4f/cosmos/.hf_home
export PIP_CACHE_DIR=/mnt/pfs/p46h4f/cosmos/.uv_cache/pip
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR"

# ---------- HuggingFace 走镜像 + 本地数据强制离线 (避免 lerobot ping HF Hub) ----------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------- 本项目预训练权重 & 数据路径 (供脚本/config 引用) ----------
export CKPT_ROOT="$(cd "$GWP_HOME/.." && pwd)/checkpoints"
# 训练用 Diffusers 格式 (models.pretrained): 含 transformer/ vae/ text_encoder/
export WAN_DIFFUSERS="$CKPT_ROOT/Wan2.2-TI2V-5B-Diffusers"
# T5 预处理用 (compute_t5_embedding --wan_path): 含 models_t5_umt5-xxl-enc-bf16.pth + google/umt5-xxl
export WAN_T5="$CKPT_ROOT/Wan2.2-T5"
# 数据集 (agilex 双臂 piper, 3 相机 cam_*, 叠衣服): visrobot01 + kairobot01
export GWP_DATA="$(cd "$GWP_HOME/.." && pwd)/kai0/data/wam_fold_v1"
# 归一化统计输出目录 (compute_norm_stats 生成 norm_stats_{vis,kai}.json)
export GWP_NORM="$GWP_HOME/assets_visrobot01"

cd "$GWP_HOME" || return
echo "[GigaWorld-Policy @ deepdive_kai0] $(python --version 2>&1) @ $GWP_VENV"
python - <<'PY' 2>/dev/null
import torch
print(f"  torch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}")
try:
    import inspect
    from giga_datasets.datasets.lerobot_dataset import LeRobotDataset
    ok = "embodiment" in inspect.signature(LeRobotDataset.__init__).parameters
    print(f"  giga_datasets embodiment= : {'OK' if ok else 'MISSING -> pip install -e ./third_party/giga-datasets'}")
except Exception as e:
    print(f"  giga_datasets check failed: {e}")
PY
echo "  WAN_DIFFUSERS=$WAN_DIFFUSERS"
echo "  WAN_T5       =$WAN_T5"
echo "  GWP_DATA     =$GWP_DATA"
echo "  GWP_NORM     =$GWP_NORM"
