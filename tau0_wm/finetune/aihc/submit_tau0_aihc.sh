#!/usr/bin/env bash
# Submit the tau0 32-GPU P2 job, injecting the private-registry image password
# (mirrors giga_world_policy/scripts/aihc/resubmit_latent.sh — password never committed).
# Usage:  AIHC_IMG_PASSWORD='****' bash finetune/aihc/submit_tau0_aihc.sh [RETRY=5]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPEC="${1:-$HERE/aijob_tau0_4n8g.json}"   # optional spec path arg
POOL=aihc-serverless; QUEUE=aihcq-z4v1apdppzwy
RETRY=${RETRY:-5}
CONFIG_NAME=${CONFIG_NAME:-}   # optional: override a CONFIG-style env
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD (image pull password) in env}"

TMP=$(mktemp /tmp/aijob_tau0.XXXXXX.json)
trap 'shred -u "$TMP" 2>/dev/null || rm -f "$TMP"' EXIT
python3 - "$SPEC" "$TMP" "$AIHC_IMG_PASSWORD" <<'PY'
import json, sys
spec, out, pw = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(spec))
def setpw(o):
    if isinstance(o, dict):
        if 'imageConfig' in o and isinstance(o['imageConfig'], dict):
            o['imageConfig']['password'] = pw
        for v in o.values(): setpw(v)
    elif isinstance(o, list):
        for v in o: setpw(v)
setpw(d)
json.dump(d, open(out, 'w'), ensure_ascii=False, indent=2)
print("[submit] image password injected into imageConfig")
PY
echo "[submit] aihc job create (fault-tolerance retry=$RETRY)"
aihc job create -f "$TMP" -p "$POOL" -q "$QUEUE" \
  --enable-fault-tolerance --fault-tolerance-args "--max-num-of-unconditional-retry=$RETRY"
