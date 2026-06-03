#!/bin/bash
# 提交 GigaWorld-Policy 叠衣服 full-FT 到百度 AIHC(PyTorchJob,2节点×8 A100)。
#
# 两种提交方式:
#  (1) 控制台(最稳,dreamzero 即此法):AIHC 控制台新建训练任务 → 框架 PyTorchJob → 副本数 2 →
#      镜像/资源/PFS 见 aijob_visrobot01_fold.json → 启动命令:
#        bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy/scripts/aihc/run_train_aihc.sh
#      → 开启 RDMA。平台会给每 pod 注入 WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT。
#  (2) OpenAPI(本脚本):curl POST aijobs。需先设环境变量(从 AIHC 控制台/凭证获取):
#        export AIHC_ENDPOINT=https://aihc.<region>.baidubce.com   # 形如 cn-beijing
#        export AIHC_TOKEN=<bearer/ak-sk 签名 token>
#      字段名以控制台 OpenAPI 为准,先 --dry 核对再去掉。
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DRY=0; SPEC="$HERE/aijob_visrobot01_fold.json"
for a in "$@"; do
    if [ "$a" = "--dry" ]; then DRY=1; else SPEC="$a"; fi
done

if [ "$DRY" = 1 ]; then
    echo "[submit] DRY — 将提交以下 spec(请核对资源池/镜像/PFS/命令):"
    cat "$SPEC"
    echo; echo "[submit] 提交命令(去掉 --dry 执行):"
    echo "  curl -sS -X POST \"\$AIHC_ENDPOINT/api/v1/aijobs\" -H \"Authorization: Bearer \$AIHC_TOKEN\" -H 'Content-Type: application/json' -d @\"$SPEC\""
    exit 0
fi

: "${AIHC_ENDPOINT:?set AIHC_ENDPOINT (e.g. https://aihc.cn-beijing.baidubce.com)}"
: "${AIHC_TOKEN:?set AIHC_TOKEN}"
echo "[submit] POST $AIHC_ENDPOINT/api/v1/aijobs  spec=$SPEC"
curl -sS -X POST "$AIHC_ENDPOINT/api/v1/aijobs" \
  -H "Authorization: Bearer $AIHC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d @"$SPEC"
echo
