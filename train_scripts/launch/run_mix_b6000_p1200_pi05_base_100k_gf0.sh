#!/bin/bash
# 实验3: Task_A mix_b6000_p1200 init from pi05_base, 100k steps (gf0).
# Same dataset/hparams as 实验2 except num_train_steps=100_000 (decay also 100k).
# 100k step, peak_lr=1.5e-5 cosine to 1.5e-6, warmup=1k, ema=0.9999, batch=128.
# inline_eval_val_root: val_self_built (more sensitive to fine-tune effect).
# exp_name: task_a_mix_base6000_pure1200_new_norm_base_pi0.5_100000.
set -euo pipefail

export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
# new env var pointing at the mix_b6000_p1200 location on /vePFS
export KAI0_LOCAL_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0/data
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export PYTORCH_CKPT_BASE=/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

cd /vePFS/tim/workspace/deepdive_kai0/kai0

echo "[train] === START $(date) ==="
.venv/bin/python scripts/train.py pi05_flatten_fold_mix_b6000_p1200_init_pi05_base_100k \
  --exp_name=task_a_mix_base6000_pure1200_new_norm_base_pi0.5_100000 \
  --resume 2>&1
echo "[train] === END $(date) ==="
