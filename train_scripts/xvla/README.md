# train_scripts/xvla/ — X-VLA 架构训练脚本

> Track X / X-VLA Florence2 PyTorch 训练脚本归此处. 2026-05-29 从 uc01 `workspace/xvla_scripts/` 归位:
> - `data/` — kai/vis joint→EE6D 20D 转换 + multi-domain dataset (见 `data/README.md`, 含 ⚠️ Rot6D 排布冲突核定)
> - `launch/` — `xvla_train.py` (X3A/B/C configs) + `xvla_train_smoke.py`

## 与 `train_scripts/kai/` 的关系

| 目录 | 架构 | 训练框架 |
|---|---|---|
| `kai/` | pi0 / openpi | JAX/Flax (`kai0/scripts/train.py`) |
| `xvla/` (本目录) | X-VLA Florence2 | PyTorch (`xvla/X-VLA/train.py` 或 `peft_train.py`, submodule) |

## 未来子目录预期 (镜像 kai/)

```
train_scripts/xvla/
├── README.md
├── launch/         X-VLA training launchers (uc01/uc02/gf3 等)
├── data/           dataset builders / yaml manifests for X-VLA format
├── eval/           X-VLA-specific eval (action MAE in EE6D 20D)
├── volc/           Volc YAMLs for X-VLA training (PyTorch DDP)
└── monitor/        训练进度监控
```

## 何时往这里放东西?

- 调用 `xvla/X-VLA/train.py` 或 `peft_train.py` 的 launcher → `xvla/launch/`
- 把 LeRobot 数据转 X-VLA 期望格式的 builder → `xvla/data/`
- X-VLA ckpt eval 脚本 (EE6D 20D action MAE) → `xvla/eval/`
- 不要把 pi0-based "xvla_*" config 实验放这里 — 那些虽然名字带 xvla 但本质是 pi0 架构, 归 `kai/`
