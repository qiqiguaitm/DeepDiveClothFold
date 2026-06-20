#!/bin/bash
# Autonomous tracker: keep REPORT.md live, poll the 5-node cluster job, finalize when local
# training ends. Runs detached. Pairs with auto_eval_loop.sh (which produces mae_curve.jsonl).
set -uo pipefail
PROJ=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/cosmos_policy_pipper_fold_colth
R="$PROJ"/../cosmos_policy_pipper_fold_colth_runs
PY=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos-policy/.venv/bin/python
MSPY=/mnt/pfs/p46h4f/cosmos/.venv/bin/python      # has aihc_cli_py
GWP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
JOB=${JOB:-job-jppjfezfpg2c}
mkdir -p "$R/reports" "$R/logs"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$R/logs/report_daemon.log"; }

idle=0
while true; do
  # poll cluster status (best-effort, no proxy)
  ( cd "$GWP" && st=$(unset http_proxy https_proxy; "$MSPY" scripts/aihc/job_status.py "$JOB" 2>/dev/null | awk '/^status/{print $3}') ; \
    echo "$JOB: ${st:-unknown} (checked $(date '+%F %T'))" > "$R/reports/cluster_status.txt" ) 2>/dev/null || true
  # regenerate report from current curve + losses
  "$PY" "$PROJ/eval/make_report.py" >>"$R/logs/report_daemon.log" 2>&1 || log "report gen failed"

  # TRACK=cluster -> exit when the AIHC job reaches a terminal state; else (local) exit when no train proc.
  if [ "${TRACK:-local}" = "cluster" ]; then
    cst=$(cat "$R/reports/cluster_status.txt" 2>/dev/null)
    case "$cst" in
      *Succeeded*|*Failed*|*Stopped*) idle=$((idle+1)); log "cluster terminal ($cst) tick $idle/3";
        [ "$idle" -ge 3 ] && { log "cluster finished; final report"; "$PY" "$PROJ/eval/make_report.py" >>"$R/logs/report_daemon.log" 2>&1; break; } ;;
      *) idle=0 ;;
    esac
  else
    if ! pgrep -f "cosmos_policy.scripts.train" >/dev/null 2>&1; then
      idle=$((idle+1)); log "local training not running (idle tick $idle/3)"
      [ "$idle" -ge 3 ] && { log "training finished; final report"; "$PY" "$PROJ/eval/make_report.py" >>"$R/logs/report_daemon.log" 2>&1; break; }
    else idle=0; fi
  fi
  sleep 180 2>/dev/null || break
done
log "report_daemon exiting; REPORT.md final."
