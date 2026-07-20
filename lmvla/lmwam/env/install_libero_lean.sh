#!/bin/bash
# 精简装 LIBERO 模拟器(eval 用). 绝不含 thop/robomimic/transformers —— 它们拉 torch-cuda13 巨型链, eval 不需要(策略在独立 lawam env).
set -x
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
PY=/home/tim/miniconda3/envs/libero/bin/python
PIP="$PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --timeout 300"
LIB=/home/tim/workspace/deepdive_kai0/lmvla/LIBERO

# 1) 注册 libero 包(install_requires 空)
$PIP -e "$LIB" --no-deps

# 2) 模拟器必需依赖(全部 torch-free)
$PIP robosuite==1.4.0
$PIP bddl==1.0.1 gym==0.25.2 easydict==1.9 hydra-core==1.2.0 cloudpickle future matplotlib "opencv-python==4.6.0.66" einops "numpy==1.23.5"

echo "===== import check ====="
$PY - <<'PY'
import mujoco; print("mujoco", mujoco.__version__)
import robosuite; print("robosuite", robosuite.__version__)
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
print("suites:", list(d.keys()))
print("LIBERO LEAN OK")
PY
echo "===== LIBERO LEAN DONE ====="
