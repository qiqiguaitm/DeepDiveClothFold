#!/usr/bin/env bash
# 轮询 VAE latent 完成状态，完成后自动提交 FastWAM v3 + gwp-ori v3 AIHC jobs。
set -uo pipefail

VAE_DIR=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/visrobot01_v3_train/vae_latent
TOTAL_EP=2353
GWP_REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
FASTWAM_JOB=$GWP_REPO/scripts/aihc/aijob_fastwam_v3_fold_5n8g.json
GWPORI_JOB=$GWP_REPO/scripts/aihc/aijob_v3_abs_5n8g.json
SUBMIT=$GWP_REPO/scripts/aihc/submit_raw.py
LOG=/tmp/auto_submit_latent.log
POLL_SEC=60

echo "[auto-submit] started $(date +%F_%T), polling every ${POLL_SEC}s, target=${TOTAL_EP} latents" | tee "$LOG"

while true; do
    done=$(ls "$VAE_DIR"/*.pt 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] latents=${done}/${TOTAL_EP}" | tee -a "$LOG"

    if [ "$done" -ge "$TOTAL_EP" ]; then
        echo "[auto-submit] latent done! submitting jobs..." | tee -a "$LOG"
        source "$GWP_REPO/env.sh" 2>/dev/null

        echo "=== FastWAM v3 ===" | tee -a "$LOG"
        python3 "$SUBMIT" "$FASTWAM_JOB" 2>&1 | tee -a "$LOG"

        echo "=== gwp-ori v3 ===" | tee -a "$LOG"
        python3 "$SUBMIT" "$GWPORI_JOB" 2>&1 | tee -a "$LOG"

        echo "[auto-submit] DONE $(date +%F_%T)" | tee -a "$LOG"
        break
    fi

    sleep "$POLL_SEC"
done
