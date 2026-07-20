# lmwam — 我们对 LaWAM 的修改层

> **设计原则:高内聚低耦合。** `lmvla/lawam` 保持为纯净的 RLinf/LaWAM submodule
> (`bd4a363`,0 改动)。本目录只装**我们拥有的东西**,与上游重叠的文件一律删除、由 submodule 提供。
> 全部内容 71K / 28 文件(重构前是 lawam 的 6M 完整副本)。

## 目录结构

| 目录 | 内容 | 与 lawam 的关系 |
|---|---|---|
| `patches/` | 对 11 个上游文件的修改(`diff -u`) | apply 到 submodule |
| `adapter/` | 我们的核心实现:`lmwm_adapter.py`、`lmwm_milestone_target.py` | 运行时经 `LMWM_ADAPTER_DIR` 注入,**无需 patch** |
| `scripts/` | `run_*.sh`、`robotwin_python_wrapper.sh` | 我们的运行脚本 |
| `configs/` | `train_robotwin_lmwm.yaml` | 我们的训练配置 |
| `env/` | `build_*.sh`、`dl_*.py`、`install_*.sh` | 环境搭建/下载 |

## 为什么是 patch + adapter 两种机制

- **adapter/(低耦合)**:LMWM 的核心逻辑是独立模块,`lawam.py` 靠 `sys.path.insert(LMWM_ADAPTER_DIR)`
  在运行时 import,**不侵入上游**。这是理想形态。
- **patches/(不得不)**:`lawam.py`(+176)、`flowmatching_expert.py`(+84)等是对 LaWAM 模型
  forward/init **数据流内部**的修改(LMWM 双通道注入、future tokens),散布在多个函数里,
  无法外置为独立文件。诚实地存为 patch,而非假装能解耦。这些改动本身已 env-gated
  (`LMWM_CKPT` 不设时走原逻辑),对上游是非破坏性的。

## 用法

```bash
cd lmvla/lmwam
./apply.sh              # 把 11 个 patch 应用到 ../lawam submodule
./apply.sh --check      # 只校验能否干净应用(不改文件)
./revert.sh             # 还原 lawam 到纯净状态

# 训练/评测时:
export LMWM_ADAPTER_DIR=$(pwd)/adapter   # lawam.py 据此 import 我们的实现
```

## 复现保证(2026-07-20 验证)

- `apply.sh` 应用到纯净 `bd4a363` 后,产生的改动与**重构前的原始工作区逐字节相同**
  (405 行改动,11/11 文件一致)。
- 11/11 patch 可被 `patch -R` 精确还原 → patch 与原改动逐行对应。
- apply 幂等:二次运行检测到"已应用"并跳过。

## 升级上游 LaWAM 时

1. `cd ../lawam && git fetch && git checkout <新版本>`
2. `cd ../lmwam && ./apply.sh --check` —— 若某 patch 冲突,说明上游改了对应文件,需手工 rebase 该 patch
3. 更新本 README 与 `.gitmodules` 记录的版本
