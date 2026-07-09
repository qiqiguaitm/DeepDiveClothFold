#!/bin/bash
cd /home/tim/workspace/deepdive_kai0
VENV=kai0/.venv/bin/python
LOG=lmwm/outputs/lmwm2
mkdir -p "$LOG"
$VENV lmwm/scripts/train_lmwm2.py \
  --datasets kai0,coffee,xvla --cond prevz --steps 6000 \
  --code_dim "$1" --tag "$2" \
  > "$LOG/$2.log" 2>&1
