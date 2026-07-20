#!/bin/bash
set -x
source /home/tim/miniconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
PIP="pip install -i https://mirrors.aliyun.com/pypi/simple/ --timeout 300"
cd /home/tim/workspace/deepdive_kai0/lmvla/LIBERO
conda run -n libero --no-capture-output pip install -U pip
conda run -n libero --no-capture-output $PIP -r requirements.txt || echo "!! requirements 部分失败"
conda run -n libero --no-capture-output $PIP -e . || echo "!! -e . 失败"
conda run -n libero --no-capture-output $PIP mujoco==3.3.2
echo "===== import check ====="
conda run -n libero --no-capture-output python - <<'PY'
import mujoco; print("mujoco", mujoco.__version__)
from libero.libero import benchmark
print("suites:", list(benchmark.get_benchmark_dict().keys()))
print("LIBERO OK")
PY
echo "===== LIBERO INSTALL DONE ====="
