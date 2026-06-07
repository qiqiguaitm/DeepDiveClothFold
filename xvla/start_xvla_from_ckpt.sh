#!/bin/bash
# X-VLA 通用「从 ckpt 一键启动」脚本 — 仿 start_scripts/kai/start_autonomy_from_ckpt.sh,
# 但针对真 X-VLA (Florence2, state_dict.pt + sidecar.json) 检查点。
#
# 给一个 ckpt 目录(或 ckpt_xvla/ 下的目录名)就拉起整套 server(:8003) + client
# (相机 + 双臂 + policy node, ee_pose 模式)。server 读 ckpt 内 sidecar.json 拿
# deploy_prompt / deploy_domain_id / action_format, 不需改 config.py。
#
# 用法:
#   ./xvla/start_xvla_from_ckpt.sh <ckpt>                 # observe-only (默认)
#   ./xvla/start_xvla_from_ckpt.sh <ckpt> --execute       # 起来即驱动臂
#   ./xvla/start_xvla_from_ckpt.sh <ckpt> --execute --trace  # +全链路 pipeline trace 落盘
#   ./xvla/start_xvla_from_ckpt.sh                        # 默认 X3.C smooth800
#
# --trace: 开启 pipeline trace — server/client 两侧逐帧落盘 (obs/20D/16D/14D/图) +
#   ros2 bag (控制+状态 topic) 到 ${KAI0_XVLA_LOG_DIR:-/tmp/xvla_stack}/trace_<ts>/。
#   跑完用 `python xvla/analyze_pipeline_trace.py <trace_dir>` 逐段核验 pipeline。
#
# <ckpt> 可以是:
#   - ckpt_xvla/ 下的目录名, 例: xvla_x3c_smooth800_step_final
#   - 绝对/相对路径, 例: /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_x3c_smooth800_step_final
#
# observe-only 起来后, 轨迹合理再另开终端翻开关:
#   ros2 topic pub /policy/execute std_msgs/Bool 'data: true' --once
#
# 环境变量 (透传, 见 start_xvla_stack.sh):
#   CUDA_VISIBLE_DEVICES / KAI0_XVLA_LOG_DIR / XVLA_SERVER_TIMEOUT / XVLA_SERVER_ARGS
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_BASE="/data1/DATA_IMP/checkpoints/ckpt_xvla"
DEFAULT_CKPT_NAME="xvla_x3c_smooth800_p0_step_final"  # P0 (ImageNet 归一化, 60k); 旧 30k 是 R1 未修版
STACK="$REPO_ROOT/start_scripts/xvla/start_xvla_stack.sh"

# 第 1 个位置参数 = ckpt (名字或路径); 空 → 默认。其余转发给 client (如 --execute)。
# 先把 --trace 从任意位置摘出来 (开 pipeline trace, 见 start_xvla_stack.sh §0),
# 其余参数原样保留: 第 1 个位置参数 = ckpt, 之后转发给 client (如 --execute)。
_ARGS=()
for _a in "$@"; do
  case "$_a" in
    --trace) export XVLA_TRACE=1 ;;
    *) _ARGS+=("$_a") ;;
  esac
done
set -- ${_ARGS[@]+"${_ARGS[@]}"}

CKPT_ARG="${1-}"; [ "$#" -gt 0 ] && shift || true
[ -z "$CKPT_ARG" ] && CKPT_ARG="$DEFAULT_CKPT_NAME"

# 解析 ckpt 目录: 已是目录就用; 否则当作 ckpt_xvla/ 下的名字。
if [ -d "$CKPT_ARG" ]; then
  CKPT_DIR="$(cd "$CKPT_ARG" && pwd)"
elif [ -d "$CKPT_BASE/$CKPT_ARG" ]; then
  CKPT_DIR="$CKPT_BASE/$CKPT_ARG"
else
  echo "ERROR: 找不到 ckpt: '$CKPT_ARG'" >&2
  echo "       既不是目录, 也不是 $CKPT_BASE/ 下的名字。" >&2
  echo "       可用 X-VLA ckpt:" >&2
  ls -1 "$CKPT_BASE" 2>/dev/null | grep -E '^xvla_x3' | sed 's/^/         /' >&2 || true
  exit 1
fi

# 校验是真 X-VLA ckpt (state_dict.pt + sidecar.json), 不是 pi0/JAX。
if [ ! -f "$CKPT_DIR/state_dict.pt" ]; then
  if [ -d "$CKPT_DIR/params" ] || [ -f "$CKPT_DIR/train_config.json" ]; then
    echo "ERROR: '$CKPT_DIR' 看起来是 pi0/JAX ckpt (params/ 或 train_config.json), 不是真 X-VLA。" >&2
    echo "       请改用: ./start_scripts/kai/start_autonomy_from_ckpt.sh '$CKPT_DIR'" >&2
  else
    echo "ERROR: $CKPT_DIR/state_dict.pt 不存在 — 不是有效的 X-VLA ckpt 目录。" >&2
  fi
  exit 1
fi
if [ ! -f "$CKPT_DIR/sidecar.json" ]; then
  echo "WARN: $CKPT_DIR/sidecar.json 缺失 — server 将回退默认 (domain_id=20, 通用 prompt)。" >&2
fi

# stage_a 旧 buggy 管线 ckpt 兜底拦截 (动作不正确, 禁止上真机)。
case "$(basename "$CKPT_DIR")" in
  *_stage_a_*)
    if [[ " $* " == *" --execute "* ]]; then
      echo "ERROR: '$(basename "$CKPT_DIR")' 是旧 buggy 管线 (block rot6d) ckpt, 动作不正确, 禁止 --execute 上真机。" >&2
      echo "       仅可 observe-only 做形状联调 (去掉 --execute)。" >&2
      exit 1
    fi
    echo "WARN: stage_a ckpt 仅供形状联调, 动作不正确, 切勿翻 /policy/execute 开关驱动臂。" >&2
    ;;
esac

echo "[start_xvla_from_ckpt] ckpt_dir=$CKPT_DIR"
if [ -f "$CKPT_DIR/sidecar.json" ]; then
  /data1/miniconda3/bin/python - "$CKPT_DIR/sidecar.json" <<'PY' 2>/dev/null || true
import json, sys
s = json.load(open(sys.argv[1]))
print(f"[start_xvla_from_ckpt] prompt={s.get('deploy_prompt')!r} "
      f"domain_id={s.get('deploy_domain_id')} action={s.get('action_format')} "
      f"chunk={s.get('action_chunk')} step={s.get('step')}")
PY
fi

# 委托给 server+client 一键 stack。ckpt 走第 1 位置参数, 其余 (--execute 等) 转发给 client。
exec "$STACK" "$CKPT_DIR" "$@"
