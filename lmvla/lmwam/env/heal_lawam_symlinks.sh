#!/bin/bash
# 幂等自愈: 重建 lawam submodule 工作树里"易被 git clean -fdx 抹掉"的关键软链。
#
# 背景(2026-07-21 事故): lawam 转 submodule 时的 `git clean -fdx`/重新 clone 抹掉了
#   latent_action_model/logs/dino_large_vae/lam_release/ 那层未跟踪软链, 真权重仍在 ckpts_dl/。
#   结果所有 LaWM/LMWM eval 在加载 LAM 时 FileNotFoundError。cron 本身无 clean(已核), 但手动
#   clean/再次重构会重演。本脚本让任何 train/eval entrypoint 顶部一句 `bash heal_lawam_symlinks.sh`
#   即可 fail-safe 自恢复, 不依赖人工。
#
# 只处理"可由 ckpts_dl/ 无损重建"的软链。dataset/ 下成千上万条嵌套软链不在此列——
#   它们同属 git-ignored, 只有 `git clean -fdx` 会抹, 防线是"submodule 内禁止 -fdx"(见 ENV_SELECTION_RULES §4)。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAWAM="$(cd "$HERE/../../lawam" && pwd)" || { echo "FATAL: 找不到 lawam submodule" >&2; exit 1; }
LOCAL="$(cd "$HERE/../.." && pwd)/lawam_local"   # = lmvla/lawam_local(submodule 外, 超项目侧)
cd "$LAWAM"

fail=0

# --- 顶层资产目录: 已迁到 lawam_local/ 的, 在 submodule 树内软链回(git clean -fdx 只删软链, 真数据在外) ---
# 安全规则: 仅当 lawam_local/D 存在(=已迁移)且 lawam/D 不是真实目录时才建软链, 绝不覆盖未迁移的真数据。
for D in results ckpts_dl dataset weights logs; do
  [ -d "$LOCAL/$D" ] || continue                       # 未迁移 → 跳过
  if [ -L "$D" ]; then continue; fi                    # 已是软链 → 幂等跳过
  if [ -d "$D" ] && [ ! -L "$D" ]; then
    echo "[heal][skip] $D 是真实目录(未迁移或迁移未完成), 不动它" >&2; continue
  fi
  ln -sfn "../lawam_local/$D" "$D" && echo "[heal] 顶层软链 $D -> ../lawam_local/$D"
done
# --- LAM release(dino_large_vae): logs/.../lam_release → ckpts_dl/ ---
LAM_REL="latent_action_model/logs/dino_large_vae/lam_release"   # 相对 lawam 根
[ -f "ckpts_dl/dino_large_vae.yaml" ] || { echo "FATAL: ckpts_dl/dino_large_vae.yaml 缺失, 无法自愈(需重下)" >&2; fail=1; }
[ -f "ckpts_dl/checkpoints/pytorch_model.pt" ] || { echo "FATAL: ckpts_dl/checkpoints/pytorch_model.pt 缺失, 无法自愈(需重下)" >&2; fail=1; }
if [ "$fail" = 0 ]; then
  mkdir -p "$LAM_REL/checkpoints"
  [ -e "$LAM_REL/dino_large_vae.yaml" ] || ln -sfn ../../../../ckpts_dl/dino_large_vae.yaml "$LAM_REL/dino_large_vae.yaml"
  [ -e "$LAM_REL/checkpoints/pytorch_model.pt" ] || ln -sfn ../../../../../ckpts_dl/checkpoints/pytorch_model.pt "$LAM_REL/checkpoints/pytorch_model.pt"
  for f in "$LAM_REL/dino_large_vae.yaml" "$LAM_REL/checkpoints/pytorch_model.pt"; do
    [ -e "$f" ] || { echo "FATAL: 软链自愈失败 $f" >&2; fail=1; }
  done
  [ "$fail" = 0 ] && echo "[heal] LAM release OK: $(ls -laL "$LAM_REL/checkpoints/pytorch_model.pt" | awk '{print $5}') bytes"
fi

[ "$fail" = 0 ] && echo "[heal_lawam_symlinks] 完成" || { echo "[heal_lawam_symlinks] 有失败项" >&2; exit 1; }
