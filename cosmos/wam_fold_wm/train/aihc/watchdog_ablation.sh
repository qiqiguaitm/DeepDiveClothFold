#!/usr/bin/env bash
# Resubmit watchdog for the t1b/t1c 2n8g cluster ablations.
# Serverless preemption + retry=3 can exhaust the AIHC fault-tolerance budget before a job
# reaches MAX_STEPS. This daemon resubmits a job when it goes terminally Failed; resubmit
# auto-resumes from the latest checkpoint in CKPT_DIR, so no progress is lost.
# Guards: resubmit only on TWO consecutive Failed reads (skip transient/internal-retry states),
# never resubmit ManualTermination (user intent), cap resubmits/tag, stop each tag at TARGET.
set -uo pipefail
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
JS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy/scripts/aihc/job_status.py
PY=/mnt/pfs/p46h4f/cosmos/.venv/bin/python3
export AIHC_IMG_PASSWORD="${AIHC_IMG_PASSWORD:-Vis@2026}"
TARGET=${TARGET:-4000}; MAXRESUB=${MAXRESUB:-10}; POLL=${POLL:-300}

declare -A JOBID=( [t1b]=job-9ftxofna7nya [t1c]=job-ve1rfq2ilwv4 )
declare -A RESUB=( [t1b]=0 [t1c]=0 )
declare -A FAILSEEN=( [t1b]=0 [t1c]=0 )

ckpt_max(){ ls -d "$COS"/wam_fold_wm_runs/train_out_${1}_2n8g/cosmos3/action/wam_fold_wm_nano_${1}/checkpoints/iter_* 2>/dev/null | sed 's#.*/iter_0*##' | sort -n | tail -1; }

echo "[watchdog] start $(date) target=$TARGET maxresub=$MAXRESUB"
while true; do
  alldone=1
  for tag in t1b t1c; do
    mx=$(ckpt_max "$tag"); mx=${mx:-0}
    if [ "$mx" -ge "$TARGET" ]; then echo "[watchdog] $tag DONE (iter_$mx >= $TARGET)"; continue; fi
    alldone=0
    st=$($PY "$JS" "${JOBID[$tag]}" 2>/dev/null | awk '/^status/{print $3}')
    if [ "$st" = "Failed" ]; then
      FAILSEEN[$tag]=$((FAILSEEN[$tag]+1))
      if [ "${FAILSEEN[$tag]}" -ge 2 ] && [ "${RESUB[$tag]}" -lt "$MAXRESUB" ]; then
        echo "[watchdog] $(date) $tag job=${JOBID[$tag]} Failed(x${FAILSEEN[$tag]}) ckpt=iter_$mx<$TARGET → RESUBMIT"
        newid=$(TAG=$tag bash "$COS"/wam_fold_wm/train/aihc/submit_ablation_2n8g.sh 2>/dev/null | grep -oE 'job-[a-z0-9]+' | head -1)
        if [ -n "$newid" ]; then JOBID[$tag]=$newid; RESUB[$tag]=$((RESUB[$tag]+1)); FAILSEEN[$tag]=0
          echo "[watchdog] $tag -> new job=$newid (resub ${RESUB[$tag]}/$MAXRESUB)"
        else echo "[watchdog] $tag resubmit FAILED (no job id)"; fi
      fi
    else
      FAILSEEN[$tag]=0   # any non-Failed read resets the counter
      echo "[watchdog] $(date) $tag job=${JOBID[$tag]} status=$st ckpt=iter_$mx resub=${RESUB[$tag]}"
    fi
  done
  [ "$alldone" = 1 ] && { echo "[watchdog] both tags reached iter_$TARGET — exiting $(date)"; break; }
  sleep "$POLL"
done
