#!/bin/bash
# LAM capacity ladder on gf3's 8 GPUs: cnn vs transformer{small,mid,big} for BOTH subgoal streams.
# Answers: is oracle 0.79 a capacity limit (transformer keeps rising) or a task limit (saturates)?
#   ssh -p 7888 root@124.174.16.237 "cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0 && bash lmwm/scripts/run_gf3_lam_ladder.sh"
PY=kai0/.venv/bin/python
export CRAVE_REPO=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
mkdir -p logs/ladder
S="--steps 12000"
L(){ CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY lmwm/scripts/optimize_subgoal.py "${@:2}" > logs/ladder/gpu$1.log 2>&1 & echo "gpu$1 PID $!: ${*:2}"; }

# --- milestone+1 stream (cd128): capacity ladder ---
L 0 --mode milestone --code_dim 128 --arch cnn $S
L 1 --mode milestone --code_dim 128 --arch transformer --width 384 --depth 4 $S
L 2 --mode milestone --code_dim 128 --arch transformer --width 512 --depth 8 $S
L 3 --mode milestone --code_dim 128 --arch transformer --width 768 --depth 12 $S
# --- near-future h3 stream (cd64): capacity ladder ---
L 4 --mode nearfuture --horizon 3 --code_dim 64 --arch cnn $S
L 5 --mode nearfuture --horizon 3 --code_dim 64 --arch transformer --width 384 --depth 4 $S
L 6 --mode nearfuture --horizon 3 --code_dim 64 --arch transformer --width 512 --depth 8 $S
L 7 --mode nearfuture --horizon 3 --code_dim 64 --arch transformer --width 768 --depth 12 $S
echo "launched 8 LAM-ladder jobs; logs logs/ladder/gpu*.log"
