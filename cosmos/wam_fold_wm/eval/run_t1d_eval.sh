#!/usr/bin/env bash
# Eval the t1d L3 (channel-concat action) model at a given iter. Sets WAM_COND_CONCAT=14 for BOTH
# export (so cond2llm is built/preserved) and fd_infer (so the inference packer builds cond_tokens).
set -uo pipefail
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
RUNS=$COS/wam_fold_wm_runs
IT=${1:?iter}
export WAM_COND_CONCAT=14
EXP=$RUNS/exported/wm_t1d_iter$IT
echo "[t1d-eval] $(date) export iter_$IT (COND_CONCAT=$WAM_COND_CONCAT)"
CKPT_BASE=$RUNS/train_out_t1d_local/cosmos3/action/wam_fold_wm_nano_t1a EXP_DIR=$EXP \
  WAM_COND_CONCAT=14 bash $COS/wam_fold_wm/eval/export_ckpt.sh "$IT" 2>&1 | tail -3
OUT=$RUNS/reports/fd_eval_t1d_iter$IT
WAM_COND_CONCAT=14 bash $COS/wam_fold_wm/eval/run_fd_infer_v3.sh \
  --export-dir "$EXP" --chunk 16 --frame-stride 3 --shift 2.0 \
  --n-episodes 12 --num-steps 8 --guidance 3.0 --out-dir "$OUT" 2>&1 | tail -6
R=$OUT/fd_daction_report.json
[ -f "$R" ] && /mnt/pfs/p46h4f/cosmos/.venv/bin/python3 -c "import json;d=json.load(open('$R'))['aggregate'];print('[t1d-eval] iter_$IT ΔPSNR(GT-wrong)=%+.4f GT=%.2f'%(d['mean_dPSNR_gt_minus_other'],d['mean_gt_psnr']))" || echo "[t1d-eval] no report"
