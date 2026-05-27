# xvla/ — X-VLA upstream + 我们的扩展开发

> **结构原则**: `X-VLA/` 子目录 = upstream pristine submodule (不动); 其他 peer 目录 = 我们的扩展 / 实验记录 / wrappers, 不修改 upstream.

## 布局 (2026-05-27 重构后)

```
xvla/
├── README.md                ← 本文档
├── X-VLA/                   ★ git submodule → github.com/2toinf/X-VLA (pristine, NEVER modify)
│   └── (Florence2 + SoftPromptedTransformer + train.py / peft_train.py / deploy.py 等)
├── config/                  我们的 wrapper / config 扩展
├── data/                    数据集 manifests 和 norm_stats
│   ├── mixed_hard/          内部生成数据 (kai0_base + kai0_dagger + vis_v2_merged norm/meta)
│   ├── xvla_soft_fold/      XVLA-Soft-Fold 公开数据集说明
│   ├── *.yaml               datasets_yaml manifests (stage1/2/3/e3_6/mixed_repos)
├── scripts/                 launchers + dataset builders (与 upstream 解耦)
│   ├── build_mixed_hard*.py    数据集构建
│   └── *_16gpu.yaml            Volc YAML
└── src/                     我们对 upstream 的 wrapper module (subclass / hub 等)
```

## 与 upstream 的关系

- **`X-VLA/` 是 submodule**, 在 GitHub UI 上直接链到 `2toinf/X-VLA`
- 当前 pin 在 commit `ccd1992` (`fix robotwin client rotation6d wxyz`, 2026 main branch)
- **永远不要直接编辑 `X-VLA/` 内容** — 该目录的任何 mod 都意味着 fork, 不再 pristine

### 更新 upstream 到新 commit

```bash
cd deepdive_kai0
git submodule update --remote xvla/X-VLA        # 拉 upstream 最新
git -C xvla/X-VLA checkout <commit>             # 或固定到具体 commit
git add xvla/X-VLA                              # 更新 gitlink
git commit -m "chore(xvla): bump X-VLA submodule to <commit>"
```

### Clone 含 submodule

```bash
# 首次 clone:
git clone --recurse-submodules https://github.com/qiqiguaitm/deepdive_kai0.git

# 已 clone, 拉 submodule:
git submodule update --init --recursive
```

## 我们的扩展开发原则

1. **不改 upstream** — 所有 wrapper / mod / hook 写在 `xvla/` 下其他目录, 通过 `from X_VLA.xxx import yyy` 风格 import upstream
2. **Train pipeline override** — 如需改训练逻辑, 写 `xvla/src/train_<exp>.py` 调 upstream 模型, 不修改 `X-VLA/train.py`
3. **数据接入** — upstream 期望的数据格式适配在 `xvla/data/` 下做转换 (例如 LeRobot v2.1 → X-VLA dataset format)
4. **实验记录** — 跑完写到 `docs/training/history/experiments/xvla_*.md`, 不写到 X-VLA/ 内

## 历史背景 (旧 README 内容, 仅参考)

旧 `xvla/` 是 "pi0 框架上做 X-VLA-style domain conditioning" 的实验目录 (hard prompt / soft prompt / action conditioning 等 ablation, 实际仍跑 pi0 模型). 那些 conditioning 实验已完成, 详见 [`docs/training/history/experiments/xvla_conditioning_methods_results.md`](../docs/training/history/experiments/xvla_conditioning_methods_results.md).

新的真 X-VLA 工作 (Track X X3.A/B/C, Florence2-based) 通过本目录的 `X-VLA/` submodule + 我们的 wrapper 进行. Track X 实验结果见 [`docs/training/history/experiments/xvla_track_x_x3_ablation_results.md`](../docs/training/history/experiments/xvla_track_x_x3_ablation_results.md).
