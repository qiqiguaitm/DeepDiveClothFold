#!/usr/bin/env bash
# Cosmos3-Nano-Policy-DROID -> wam_fold smoke TRAINING VALIDATION.
# Mirrors tests/nano_training_smoke_test.py: convert->DCP -> train N steps -> assert min(loss)<loss[0].
# Run when the cosmos3 .venv is ready. Supervised first run (code is untested).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv
PY=$VENV/bin/python
MODEL=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/models/modelscope/Cosmos3-Nano-Policy-DROID
DCP=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/checkpoints/Cosmos3-Nano-Policy-DROID-dcp
WANVAE_DIR=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/checkpoints/Wan2.2-TI2V-5B-Diffusers
OUT=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/smoke_out
LOG=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports
TOML=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/train/recipe_nano.toml
export PYTHONPATH="$CF:${PYTHONPATH:-}"
export IMAGINAIRE_OUTPUT_ROOT="$OUT"; mkdir -p "$OUT" "$LOG" "$(dirname $DCP)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NGPU=${NGPU:-8}
# FFmpeg libs for torchcodec video decode (venv lacks plain libav*; conda ffmpeg8 has libavutil.so.60)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home
export HF_HUB_OFFLINE=1                      # Qwen tokenizer + Wan VAE pre-cached
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=600
cd "$CF"

echo "=== [0] env check $(date +%H:%M:%S) ==="
[ -x "$PY" ] || { echo "FATAL: no venv python $PY"; exit 2; }
"$PY" -c "import torch;assert torch.cuda.is_available();print('torch',torch.__version__,'cuda ok',torch.cuda.device_count(),'gpus')" || { echo "FATAL: torch/cuda not ready"; exit 2; }

echo "=== [1] convert Policy-DROID diffusers -> DCP $(date +%H:%M:%S) ==="
if [ -d "$DCP" ] && find "$DCP" -name ".metadata" | grep -q .; then
  echo "DCP already present, skip convert"
else
  "$PY" -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path "$MODEL" -o "$DCP" > "$LOG/convert_dcp.log" 2>&1
  rc=$?; echo "convert rc=$rc"; tail -15 "$LOG/convert_dcp.log"
  [ $rc -eq 0 ] || { echo "FATAL: convert failed (see convert_dcp.log)"; exit 3; }
fi

echo "=== [2] smoke train wam_fold (single GPU, ${SMOKE_ITERS:-10} steps) $(date +%H:%M:%S) ==="
export BASE_CKPT_DCP="$DCP"
# tokenizer wants a Wan2.2 VAE; try the diffusers vae safetensors, fall back to .pth if present
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
PORT=$(( ( $$ % 20000 ) + 30000 ))
"$VENV/bin/torchrun" --nproc_per_node=$NGPU --master_port=$PORT \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter=${SMOKE_ITERS:-10} \
  > "$LOG/smoke_train.log" 2>&1
rc=$?; echo "train rc=$rc"
echo "--- last 25 lines ---"; tail -25 "$LOG/smoke_train.log"

echo "=== [3] validate loss trend $(date +%H:%M:%S) ==="
"$PY" - "$LOG/smoke_train.log" << 'PYEOF'
import re,sys
t=open(sys.argv[1]).read()
rx=re.compile(r"\[RANK\s+0\]\s+Iteration\s+\d+:.*?Loss:\s+([-+0-9.eE]+)")
v=[float(x) for x in rx.findall(t)]
print("parsed losses:",v)
if not v: print("VALIDATION: NO LOSSES PARSED (training may have errored)"); sys.exit(1)
ok = (min(v) < v[0]) if len(v)>1 else (v[0]==v[0])
print(f"loss[0]={v[0]:.4f} min={min(v):.4f} -> {'PASS' if ok else 'FAIL (no downward step)'}")
print("done with training" in t.lower() and "TRAINING COMPLETED CLEANLY" or "TRAINING DID NOT REPORT CLEAN FINISH")
sys.exit(0 if ok else 1)
PYEOF
vrc=$?
echo "=== SMOKE_VALIDATION_DONE rc=$vrc $(date +%H:%M:%S) ==="
