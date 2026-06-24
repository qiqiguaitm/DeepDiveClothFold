#!/usr/bin/env bash
# Submit the Cosmos3-Nano-Policy wam_fold cross-rig 32-GPU (4n8g) AIHC job.
# Mirrors tau0_wm/finetune/aihc/submit_tau0_aihc.sh — image-pull password injected at submit
# time and never committed.
# Usage:  AIHC_IMG_PASSWORD='****' bash submit_cosmos_aihc.sh [SPEC.json] [RETRY=5]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPEC="${1:-$HERE/aijob_cosmos_wamfold_4n8g.json}"
POOL=${POOL:-aihc-serverless}; QUEUE=${QUEUE:-aihcq-z4v1apdppzwy}
RETRY=${RETRY:-5}
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD (image pull password) in env}"

TMP=$(mktemp /tmp/aijob_cosmos.XXXXXX.json)
trap 'shred -u "$TMP" 2>/dev/null || rm -f "$TMP"' EXIT
python3 - "$SPEC" "$TMP" "$AIHC_IMG_PASSWORD" <<'PY'
import json, sys
spec, out, pw = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(spec))
def setpw(o):
    if isinstance(o, dict):
        if isinstance(o.get('imageConfig'), dict): o['imageConfig']['password'] = pw
        for v in o.values(): setpw(v)
    elif isinstance(o, list):
        for v in o: setpw(v)
setpw(d)
json.dump(d, open(out, 'w'), ensure_ascii=False, indent=2)
print("[submit] image password injected")
PY
echo "[submit] aihc job create  pool=$POOL queue=$QUEUE retry=$RETRY"
aihc job create -f "$TMP" -p "$POOL" -q "$QUEUE" \
  --enable-fault-tolerance --fault-tolerance-args "--max-num-of-unconditional-retry=$RETRY"
