#!/usr/bin/env bash
# 重新提交 latent 训练 job(resume=True 从最新 ckpt 接续),带**自动容错重试**:
# 之前 3 次 job 都是 exitCode 1(疑似 NCCL/IB 瞬时故障)且 faultToleranceLimit=0 → 终态失败、
# 需人工重提。本脚本启用 --max-num-of-unconditional-retry 让 job 崩溃后自动拉起、从最新 ckpt 续训。
#
# 镜像密码绝不入库:从环境变量取(AIHC_IMG_PASSWORD),注入临时 spec,提交后 shred。
# 用法:  AIHC_IMG_PASSWORD='****' bash scripts/aihc/resubmit_latent.sh   [RETRY=5]
set -euo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
cd "$REPO"; source env.sh >/dev/null 2>&1 || true
SPEC=scripts/aihc/aijob_create_latent_4n8g.json
POOL=aihc-serverless; QUEUE=aihcq-z4v1apdppzwy
RETRY=${RETRY:-5}
CONFIG_NAME=${CONFIG_NAME:-}      # 可选:覆盖 spec 的 CONFIG env(如换 warmup 配置)
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD (image pull password) in env}"

TMP=$(mktemp /tmp/aijob_latent.XXXXXX.json)
trap 'shred -u "$TMP" 2>/dev/null || rm -f "$TMP"' EXIT
python - "$SPEC" "$TMP" "$AIHC_IMG_PASSWORD" "$CONFIG_NAME" <<'PY'
import json,sys
spec,out,pw,cfg=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
d=json.load(open(spec))
def setpw(o):
    if isinstance(o,dict):
        if 'imageConfig' in o and isinstance(o['imageConfig'],dict): o['imageConfig']['password']=pw
        if 'envs' in o and isinstance(o['envs'],list) and cfg:
            for e in o['envs']:
                if e.get('name')=='CONFIG': e['value']=cfg
        for v in o.values(): setpw(v)
setpw(d); json.dump(d,open(out,'w'),ensure_ascii=False,indent=2)
print('CONFIG =', cfg or '(spec default)')
PY
echo "[resubmit] creating job (fault-tolerance retry=$RETRY) from $SPEC"
aihc job create -f "$TMP" -p "$POOL" -q "$QUEUE" \
  --enable-fault-tolerance --fault-tolerance-args "--max-num-of-unconditional-retry=$RETRY"
