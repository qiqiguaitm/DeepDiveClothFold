#!/bin/bash
# 驱动: LMWM(dual2q) vs LaWM(armB) 变 seed 重复评测, 每 seed 两臂同时占一张卡。
cd /vePFS/tim/workspace/deepdive_kai0/lmvla/lawam
for s in 101 102 103; do
  WORKERS=4 ARM=dual2q SEED=$s GPU=0 TRIALS=50 bash ./run_lmwm_vs_lawm_seeds.sh > /tmp/rv_dual2q_$s.log 2>&1 &
  p1=$!
  sleep 20
  WORKERS=4 ARM=armB   SEED=$s GPU=1 TRIALS=50 bash ./run_lmwm_vs_lawm_seeds.sh > /tmp/rv_armB_$s.log 2>&1 &
  p2=$!
  wait $p1 $p2
  echo "=== seed $s 完成 $(date) ==="
done
echo "ALL_REVALIDATE_DONE"
