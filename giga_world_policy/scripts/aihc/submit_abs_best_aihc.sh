#!/usr/bin/env bash
# 提交 GWP abs-best 4节点 bs/gpu=8×8卡 50k 训练到 AIHC(aihc CLI,镜像同 delta latent 任务)。
# 镜像拉取密码在提交时注入、不入库。镜像基础设施/队列与 delta 4n8g 一致。
# 用法:  AIHC_IMG_PASSWORD='****' bash scripts/aihc/submit_abs_aihc.sh [--dry]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPEC="$HERE/aijob_abs_best_5n8g.json"
POOL=${POOL:-aihc-serverless}; QUEUE=${QUEUE:-aihcq-z4v1apdppzwy}; RETRY=${RETRY:-8}
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD (image pull password)}"

TMP=$(mktemp /tmp/aijob_abs.XXXXXX.json)
trap 'shred -u "$TMP" 2>/dev/null || rm -f "$TMP"' EXIT
python3 - "$SPEC" "$TMP" "$AIHC_IMG_PASSWORD" <<'PY'
import json, sys
spec, out, pw = sys.argv[1:4]
d = json.load(open(spec))
def setpw(o):
    if isinstance(o, dict):
        if isinstance(o.get('imageConfig'), dict): o['imageConfig']['password'] = pw
        for v in o.values(): setpw(v)
    elif isinstance(o, list):
        for v in o: setpw(v)
setpw(d); json.dump(d, open(out, 'w'), ensure_ascii=False, indent=2)
print("[submit] image password injected; replicas=%d config=%s" % (
    d['jobSpec']['replicas'], next(e['value'] for e in d['jobSpec']['envs'] if e['name']=='CONFIG')))
PY

if [ "$DRY" = 1 ]; then
  echo "[submit] DRY — would run:"
  echo "  aihc job create -f <tmp> -p $POOL -q $QUEUE --enable-fault-tolerance --fault-tolerance-args \"--max-num-of-unconditional-retry=$RETRY\""
  echo "[submit] spec preview (password redacted):"; sed 's/"password": "[^"]*"/"password": "<redacted>"/' "$TMP"
  exit 0
fi
echo "[submit] aihc job create  pool=$POOL queue=$QUEUE retry=$RETRY"
aihc job create -f "$TMP" -p "$POOL" -q "$QUEUE" \
  --enable-fault-tolerance --fault-tolerance-args "--max-num-of-unconditional-retry=$RETRY"
