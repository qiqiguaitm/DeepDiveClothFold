#!/bin/bash
# 装 LIBERO 模拟器 env (LaWAM eval 用). 独立 conda env + mujoco 3.3.2.
set -x
source /home/tim/miniconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
PIP="pip install -i https://mirrors.aliyun.com/pypi/simple/ --timeout 300"
LIBERO_DIR=/home/tim/workspace/deepdive_kai0/lmvla/LIBERO

echo "===== [1/5] clone LIBERO ====="
if [ -d "$LIBERO_DIR/.git" ]; then echo "already cloned"; else
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR" || exit 11
fi

echo "===== [2/5] conda create libero py3.10 ====="
conda env list | grep -q "^libero " || conda create -n libero python=3.10 -y || exit 12

echo "===== [3/5] LIBERO requirements + install ====="
cd "$LIBERO_DIR"
conda run -n libero --no-capture-output pip install -U pip
conda run -n libero --no-capture-output $PIP -r requirements.txt || echo "!! requirements 部分失败, 继续"
conda run -n libero --no-capture-output $PIP -e . || echo "!! -e . 失败"

echo "===== [4/5] mujoco 3.3.2 (LaWAM 指定) ====="
conda run -n libero --no-capture-output $PIP mujoco==3.3.2

echo "===== [5/5] import check ====="
conda run -n libero --no-capture-output python - <<'PY'
import mujoco, libero
from libero.libero import benchmark
print("mujoco", mujoco.__version__)
bm = benchmark.get_benchmark_dict()
print("LIBERO suites:", list(bm.keys()))
print("LIBERO OK")
PY
echo "===== LIBERO ENV DONE ====="
