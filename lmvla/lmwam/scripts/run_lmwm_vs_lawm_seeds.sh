#!/bin/bash
# 重新验证: LMWM(dual2q, v2 r场) vs LaWM(armB baseline) —— 本机变 seed 重复评测。
# env 块与 volc 模板 libero_eval_x4_8h20.yaml / libero_eval_2ckpt_x4_8h20.yaml 逐字对齐。
# 用法: ARM=dual2q SEED=101 GPU=0 TRIALS=50 bash run_lmwm_vs_lawm_seeds.sh
set -e
REPO=/vePFS/tim/workspace/deepdive_kai0
cd $REPO/lmvla/lawam
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# ★ 解释器与 volc 一致(kai0/.venv), 不用 lawam conda env
export PYTHON=$REPO/kai0/.venv/bin/python
export PATH=$REPO/kai0/.venv/bin:$PATH
export STAR_VLA_PYTHON=$PYTHON
export LIBERO_PYTHON=$PYTHON
export LIBERO_HOME=$REPO/lmvla/LIBERO
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com

ARM="${ARM:?need ARM=dual2q|armB}"
SEED="${SEED:?need SEED}"
GPU="${GPU:-0}"
TRIALS="${TRIALS:-50}"

# 架构相关 env 配错 = state_dict size mismatch, 故先全 unset 再按臂 export
unset LMWM_CKPT LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_DUAL LMWM_DUAL_2Q \
      LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_MS_TSCHED LMWM_HINT_DROPOUT

case "$ARM" in
  dual2q)
    CKPT=results/Checkpoints/libero/20260718_111535+lmwm_dual_2q_cnsh_volc/checkpoints/steps_12500_pytorch_model.pt
    export LMWM_CKPT=$REPO/lmvla/lmwm/checkpoints/lmwm_libero_rvalley/lmwm.pt   # v2 r 场 LMWM
    export LMWM_ADAPTER_DIR=$REPO/lmvla/lawam
    export LMWM_SWAP_TEACHER=1
    export LMWM_DUAL=1
    export LMWM_DUAL_2Q=1
    ;;
  armB)   # LaWM baseline: 无任何 LMWM 注入
    CKPT=results/Checkpoints/libero/armB_baseline_20k/checkpoints/steps_20000_pytorch_model.pt
    ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
[ -f "$CKPT" ] || { echo "FATAL: ckpt 不存在 $CKPT"; exit 13; }

export GPU_IDS="$GPU" EVAL_GPU_IDS="$GPU"
export PORT_BASE=$((5694 + GPU*20 + SEED%7*3))
export EVAL_SEED="$SEED"
export LIBERO_RUN_GROUP="revalidate_${ARM}" RUN_TAG="seed${SEED}"

SUITES=libero_10 \
NUM_TRIALS_PER_TASK="$TRIALS" \
NUM_WORKERS="${WORKERS:-1}" \
OUTPUT_ROOT=results/eval_runs/libero \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$CKPT"
echo "===== DONE ARM=$ARM SEED=$SEED ====="
