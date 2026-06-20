#!/usr/bin/env bash
# Self-healing bootstrap: downloads + installs + starts training.
# Resilient to flaky network — wget -c resumes, retries on failure.
set -euo pipefail
CP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos-policy
PROJ=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/cosmos_policy_pipper_fold_colth
source "$PROJ/setup/env.sh"
W="$RUNS/wheels"; mkdir -p "$W"
export UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_TIMEOUT=3600
cd "$CP"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUNS/logs/bootstrap.log"; }

# ---- Step 0: download remaining wheels with wget -c (one at a time) ----
download_one() {
  local f="$1" u="$2"
  local sz; sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  log "download $f (have $((sz/1048576)) MB)"
  while true; do
    wget -c -q --timeout=180 --tries=3 "$u" -O "$f" 2>&1
    local rc=$?
    local new; new=$(stat -c %s "$f" 2>/dev/null || echo 0)
    log "  $f: $((new/1048576)) MB (rc=$rc)"
    # wget exit 0 = success (complete file or download finished)
    if [ "$rc" -eq 0 ]; then
      log "  $f: complete ($((new/1048576)) MB)"
      return 0
    fi
    # No progress = stalled; wait and retry
    if [ "$new" -eq "$sz" ] && [ "$sz" -gt 0 ]; then
      log "  stalled; retry in 30s"; sleep 30
    fi
    sz=$new
  done
}

cd "$W"
download_one "torch-2.7.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl" \
  "https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl" || true
download_one "flash_attn-2.7.3+cu128.torch27-cp310-cp310-linux_x86_64.whl" \
  "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/flash_attn-2.7.3%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl" || true
download_one "natten-0.21.0+cu128.torch27-cp310-cp310-linux_x86_64.whl" \
  "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/natten-0.21.0%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl" || true
download_one "transformer_engine-2.2+cu128.torch27-cp310-cp310-linux_x86_64.whl" \
  "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/transformer_engine-2.2%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl" || true

# ---- Step 1: install all local wheels ----
cd "$W"
log "installing local wheels..."
"$UV" pip install --python 3.10 ./*.whl 2>&1 | tail -5 || {
  # individual fallback
  for f in *.whl; do
    log "  install $f..."
    "$UV" pip install --python 3.10 "$f" 2>&1 | tail -2 || log "  FAILED $f"
    sleep 1
  done
}

log "torch: $("$UV" pip show torch 2>/dev/null | grep Version | head -1 || echo MISSING)"

# ---- Step 2: uv sync to fill remaining small packages ----
log "uv sync (remaining pkgs)..."
while true; do
  "$UV" sync --extra cu128 --group aloha --python 3.10 2>&1 | tail -10
  if "$UV" run --extra cu128 --group aloha --python 3.10 python -c "import torch; print('CUDA',torch.cuda.is_available())" 2>/dev/null; then
    log "ENV READY — torch + cuda OK"
    break
  fi
  log "sync incomplete; retry in 60s..."
  sleep 60
done

# ---- Step 3: T5 embedding ----
if [ ! -f "$PIPPER_DATA/t5_embeddings.pkl" ]; then
  log "T5 embedding..."
  "$UV" run --extra cu128 --group aloha --python 3.10 \
    python "$PROJ/data/make_t5_embeddings.py" --data_dir "$PIPPER_DATA" --commands "fold cloth"
fi

# ---- Step 4: dryrun ----
log "DRYRUN..."
DRYRUN=1 NGPU=1 bash "$PROJ/train/train_8gpu.sh" > "$RUNS/logs/dryrun.log" 2>&1 || { log "DRYRUN FAILED"; exit 1; }
log "dryrun OK"

# ---- Step 5: smoke ----
log "SMOKE (20 steps)..."
MAXITER=20 SAVEITER=20 NGPU=8 bash "$PROJ/train/train_8gpu.sh" > "$RUNS/logs/smoke.log" 2>&1 || { log "SMOKE FAILED"; exit 1; }
log "smoke OK"

# ---- Step 6: full train ----
log "FULL TRAIN (30000 steps)..."
MAXITER=30000 SAVEITER=1000 NGPU=8 bash "$PROJ/train/train_8gpu.sh" > "$RUNS/logs/full_train.log" 2>&1
log "FULL TRAIN DONE rc=$?"
