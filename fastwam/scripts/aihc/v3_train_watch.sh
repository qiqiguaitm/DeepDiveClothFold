#!/usr/bin/env bash
# AIHC v3 训练状态监控(harness 后台任务用,三态退出唤醒):
#   ABNORMAL —— pod 日志出现 Traceback/OOM/exitcode,或 job Failed/Stopped
#   STALL    —— 30 分钟无新 step 行
#   DONE     —— job Succeeded 或 step 达到上限
# 用法:bash scripts/aihc/v3_train_watch.sh <jobId> [RUN_NAME]
set -uo pipefail
JOB=${1:?usage: v3_train_watch.sh <jobId> [run_name]}
RUN_NAME=${2:-aihc_5n8g_v3}
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/fastwam
GWP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
OUT="$REPO/runs/visrobot01_fold_uncond_1e-4/${RUN_NAME}"
last_step=-1; stall=0
while :; do
  st=$(cd "$GWP" && source env.sh >/dev/null 2>&1 && python3 scripts/aihc/job_status.py "$JOB" 2>/dev/null | awk -F': ' '/^status/{print $2}')
  line=$(grep -aoE "step=[0-9]+/[0-9]+ loss=[0-9.]+" "$OUT/pod_0.stdout" 2>/dev/null | tail -1)
  step=$(echo "$line" | grep -oE "step=[0-9]+" | grep -oE "[0-9]+" | head -1); step=${step:--1}
  nck=$(ls "$OUT/checkpoints/weights/"step_*.pt 2>/dev/null | wc -l)
  echo "[$(date '+%m-%d %H:%M')] job=${st:-?} $line ckpts=$nck"
  case "${st:-}" in Failed|Stopped|ManualTermination) echo "TRAIN_ABNORMAL job=$st"; break;; Succeeded) echo "TRAIN_DONE"; break;; esac
  if grep -qaE "OutOfMemoryError|Traceback|exitcode  : 1" "$OUT/pod_0.stdout" 2>/dev/null; then
    echo "TRAIN_ABNORMAL pod 日志异常:"; grep -aE "OutOfMemoryError|RuntimeError|Error" "$OUT/pod_0.stdout" | tail -3; break
  fi
  if [ "$step" -eq "$last_step" ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 6 ] && { echo "TRAIN_STALL(30min 无新 step)"; tail -c 400 "$OUT/pod_0.stdout" | tr '\r' '\n' | tail -3; break; }
  last_step=$step
  sleep 300
done
