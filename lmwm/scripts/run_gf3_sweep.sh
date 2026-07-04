#!/bin/bash
# Launch the LMWM optimization sweep across gf3's 8 GPUs. Run FROM the gf3 repo dir.
#   ssh -p 7888 root@124.174.16.237 "cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0 && bash lmwm/scripts/run_gf3_sweep.sh"
PY=kai0/.venv/bin/python
export CRAVE_REPO=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
mkdir -p logs/sweep
launch() { CUDA_VISIBLE_DEVICES=$1 nohup $PY "${@:2}" > logs/sweep/gpu$1.log 2>&1 & echo "gpu$1 PID $!: ${*:2}"; }

# --- subgoal predictors: both horizons (near-future û_T + milestone+1), forward-from-current ---
launch 0 lmwm/scripts/optimize_subgoal.py --mode nearfuture --horizon 3 --code_dim 64
launch 1 lmwm/scripts/optimize_subgoal.py --mode nearfuture --horizon 5 --code_dim 64
launch 2 lmwm/scripts/optimize_subgoal.py --mode nearfuture --horizon 5 --code_dim 128
launch 3 lmwm/scripts/optimize_subgoal.py --mode milestone --code_dim 64
launch 4 lmwm/scripts/optimize_subgoal.py --mode milestone --code_dim 128
# --- decoder perfection: capacity + GDL sharpness ---
launch 5 lmwm/scripts/optimize_patch_decoder.py --dec medium --gdl 0
launch 6 lmwm/scripts/optimize_patch_decoder.py --dec big --gdl 0.5
launch 7 lmwm/scripts/optimize_patch_decoder.py --dec xl --gdl 0.5
echo "launched 8 jobs; logs in logs/sweep/gpu*.log"
