#!/bin/bash
# 建 lawam conda env (py3.10 + requirements + pip install -e). flash-attn 暂不装(requirements 里已注释), eval 先试 sdpa。
set -x
source /home/tim/miniconda3/etc/profile.d/conda.sh
cd /home/tim/workspace/deepdive_kai0/lmvla/lawam

# 清代理 + aliyun pip 镜像
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
PIP="pip install -i https://mirrors.aliyun.com/pypi/simple/ --timeout 300"

echo "===== [1/4] conda create lawam py3.10 ====="
conda create -n lawam python=3.10 -y || exit 11

echo "===== [2/4] pip -U + requirements ====="
conda run -n lawam --no-capture-output pip install -U pip
conda run -n lawam --no-capture-output $PIP -r requirements.txt || echo "!! requirements 部分失败, 继续"

echo "===== [3/4] pip install -e . ====="
conda run -n lawam --no-capture-output $PIP -e . || echo "!! -e . 失败"

echo "===== [4/4] import check ====="
conda run -n lawam --no-capture-output python - <<'PY'
import torch, starVLA
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("starVLA OK")
PY
echo "===== BUILD DONE ====="
