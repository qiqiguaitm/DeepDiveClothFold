#!/usr/bin/env bash
# Autonomous ladder evaluator: watches t1a/t1b/t1c checkpoints, and when an eval-worthy
# iter lands, exports it -> HF and runs the ΔPSNR controllability eval (chunk16/stride3/
# shift2, matching the t1* training regime), appending a one-line result to a summary file.
# Survives session interruptions (run detached). Idempotent: skips already-evaluated iters.
set -uo pipefail
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
RUNS=$COS/wam_fold_wm_runs
SUM=$RUNS/reports/ladder_eval_summary.tsv
mkdir -p "$RUNS/reports"
[ -f "$SUM" ] || echo -e "tag\titer\tGT_PSNR\tdPSNR_gt_wrong\tdPSNR_gt_zero\tverdict\tts" > "$SUM"

# tag -> "run_subdir exp_name"; targets per tag (iters to eval)
declare -A RUN=( [t1a]="train_out_t1a_local wam_fold_wm_nano_t1a"
                 [t1b]="train_out_t1b_2n8g wam_fold_wm_nano_t1b"
                 [t1c]="train_out_t1c_2n8g wam_fold_wm_nano_t1c" )
declare -A TGT=( [t1a]="3000" [t1b]="1000 1500 2000 3000 4000" [t1c]="1000 1500 2000 3000 4000" )

pick_gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '$2>20000{print $1; exit}'; }   # first GPU with >20G free

evaluated(){ grep -qP "^$1\t$2\t" "$SUM"; }

echo "[autoeval] start $(date)"
for round in $(seq 1 240); do   # ~ up to 20h (5min poll)
  alldone=1
  for tag in t1a t1b t1c; do
    set -- ${RUN[$tag]}; sub=$1; exp=$2
    CK=$RUNS/$sub/cosmos3/action/$exp/checkpoints
    for it in ${TGT[$tag]}; do
      it9=$(printf "%09d" "$it")
      [ -d "$CK/iter_$it9" ] || { alldone=0; continue; }
      evaluated "$tag" "$it" && continue
      alldone=0
      gpu=$(pick_gpu); [ -z "$gpu" ] && { echo "[autoeval] no free GPU, wait"; break; }
      echo "[autoeval] $(date) eval $tag iter_$it on GPU$gpu"
      EXP=$RUNS/exported/wm_${tag}_iter$it
      # export (idempotent: export_ckpt skips if config.json exists)
      CKPT_BASE=$RUNS/$sub/cosmos3/action/$exp EXP_DIR=$EXP CUDA_VISIBLE_DEVICES=$gpu \
        bash $COS/wam_fold_wm/eval/export_ckpt.sh "$it" >/dev/null 2>&1
      OUT=$RUNS/reports/fd_eval_${tag}_iter$it
      CUDA_VISIBLE_DEVICES=$gpu bash $COS/wam_fold_wm/eval/run_fd_infer_v3.sh \
        --export-dir "$EXP" --chunk 16 --frame-stride 3 --shift 2.0 \
        --n-episodes 12 --num-steps 8 --guidance 3.0 --out-dir "$OUT" > "$OUT.log" 2>&1
      R=$OUT/fd_daction_report.json
      if [ -f "$R" ]; then
        row=$(/mnt/pfs/p46h4f/cosmos/.venv/bin/python3 -c "import json;d=json.load(open('$R'));a=d['aggregate'];print('%s\t%s\t%.3f\t%+.4f\t%+.4f\t%s'%('$tag','$it',a['mean_gt_psnr'],a['mean_dPSNR_gt_minus_other'],a['mean_dPSNR_gt_minus_zero'],d.get('verdict','')[:30]))" 2>/dev/null)
        [ -n "$row" ] && { echo -e "$row\t$(date +%H:%M)" >> "$SUM"; echo "[autoeval] DONE $tag iter_$it -> $row"; }
      else echo "[autoeval] eval FAILED $tag iter_$it (see $OUT.log)"; fi
    done
  done
  [ "$alldone" = 1 ] && { echo "[autoeval] all targets evaluated — exit $(date)"; break; }
  sleep 300
done
