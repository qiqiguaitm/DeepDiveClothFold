#!/usr/bin/env bash
# Autonomous driver: wait for env build, then T5 -> dryrun -> smoke -> full train.
# Each stage gated on the previous one's success. All logs under $RUNS/logs/.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../setup/env.sh"
cd "$CP_ROOT"
L="$RUNS/logs"; mkdir -p "$L"
ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*" | tee -a "$L/run_all.log"; }

say "DRIVER START"

# 1) Wait for uv sync to finish, then verify torch+cuda.
while pgrep -f "uvbin/uv sync --extra cu128 --group aloha" >/dev/null 2>&1; do sleep 20; done
say "uv sync finished; verifying torch"
"$UV" run --extra cu128 --group aloha --python 3.10 python - >"$L/verify.log" 2>&1 <<'PY' || { say "TORCH VERIFY FAILED (see verify.log)"; exit 1; }
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
PY
say "torch OK: $(tail -1 "$L/verify.log")"

# 2) T5 embedding for the single command "fold cloth" (downloads T5 on first run).
if [ ! -f "$PIPPER_DATA/t5_embeddings.pkl" ]; then
  say "computing T5 embedding -> $PIPPER_DATA/t5_embeddings.pkl"
  "$UV" run --extra cu128 --group aloha --python 3.10 \
    python "$PROJ/data/make_t5_embeddings.py" --data_dir "$PIPPER_DATA" --commands "fold cloth" \
    >"$L/t5.log" 2>&1 || { say "T5 FAILED (see t5.log)"; exit 1; }
fi
[ -f "$PIPPER_DATA/t5_embeddings.pkl" ] || { say "t5_embeddings.pkl missing after T5 step"; exit 1; }
say "T5 OK ($(du -h "$PIPPER_DATA/t5_embeddings.pkl" | cut -f1))"

# 3) Dry run: validate full config resolution + trigger base/ALOHA checkpoint downloads.
say "dryrun (config validate + ckpt download)"
DRYRUN=1 NGPU=1 bash "$PROJ/train/train_8gpu.sh" >"$L/dryrun.log" 2>&1 || { say "DRYRUN FAILED (see dryrun.log)"; exit 1; }
say "dryrun OK"

# 4) Smoke: a few steps on real data; validates dataloader, model, loss, RAM, checkpointing.
say "smoke train (MAXITER=20)"
MAXITER=20 SAVEITER=20 NGPU=8 bash "$PROJ/train/train_8gpu.sh" >"$L/smoke.log" 2>&1
SRC=$?
if [ $SRC -ne 0 ]; then say "SMOKE FAILED rc=$SRC (see smoke.log)"; exit 1; fi
say "smoke OK"

# 5) Full training (warm-started). LR decays at 20K (inherited ALOHA schedule). Stop when L1~0.01.
say "FULL TRAIN start (MAXITER=30000, SAVEITER=1000)"
MAXITER=30000 SAVEITER=1000 NGPU=8 bash "$PROJ/train/train_8gpu.sh" >"$L/full_train.log" 2>&1
say "FULL TRAIN exited rc=$?"
say "DRIVER DONE"
