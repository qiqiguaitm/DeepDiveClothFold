# train_scripts/kai/data — 数据集构建脚本

> 构建/处理 Task_A 训练数据集的脚本集合。**输出位置遵循统一规范(见下)。**

## ⭐ 数据集存放规范 (强制)

**所有新构建的数据集一律输出到 `self_built/` 下**:

```
<KAI0_DATA_ROOT>/data/Task_A/self_built/<dataset_name>/
```

- 不要直接输出到 `Task_A/` 根目录(根目录只留 *非构建产物*:`vis_base/` 原始采集、`kai0_base/`·`kai0_dagger/`·`kai0_advantage/` HF 官方)。
- config.py 里训练 config 的 `repo_id` 也指向 `self_built/<name>`。
- 完整规范见 `docs/deployment/training_ops/storage_and_env.md §2.3`。

### 写新 build 脚本的标准 DST 写法(复制即用)

```python
from pathlib import Path
import os

# KAI0_DATA_ROOT 由 setup_env.sh 设置:
#   gf0  = /vePFS/tim/workspace/deepdive_kai0/kai0
#   uc   = /data/shared/ubuntu/workspace/deepdive_kai0/kai0
ROOT = Path(os.environ.get("KAI0_DATA_ROOT",
            "/vePFS/tim/workspace/deepdive_kai0/kai0")) / "data" / "Task_A"

DATASET_NAME = "my_new_dataset"
DST = ROOT / "self_built" / DATASET_NAME    # ← 一律落 self_built/
DST.mkdir(parents=True, exist_ok=True)
```

> 多数脚本用 `argparse` 暴露 `--out` / `--dst`,**default 必须是 `ROOT / "self_built" / <name>`**。这样不传参也自动落 self_built/。

### 数据源(构建输入)约定

| 用途 | 路径 |
|---|---|
| 原始采集 base(按 date `<date>-v2/`) | `Task_A/vis_base/`(gf0 真实本地盘,2026-05-28 起;原 TOS 软链 + vis_base_real 已合并到此) |
| HF 官方 base/dagger | `Task_A/kai0_base/`、`Task_A/kai0_dagger/`(或 `dataset/Kai0_official/Task_A/` on uc) |
| 跨 region 同步源 | TOS `tos://transfer-shanghai/KAI0/Task_A/...`(见 `docs/.../data_sync_tos.md`) |

## 脚本合规状态 (2026-05-28 审计)

**✅ 已合规(输出 → `self_built/`)**:
`build_A_0423_0527.py`、`build_vis_v2_full.py`、`build_vis_v2_merged.py`、
`build_task_a_mix_apr28_450.py`、`build_task_a_mix_vis600.py`、`build_task_a_mix_vis600_split.py`、
`build_task_a_mix_b6000_p1200.py`、`build_task_a_new_100.py`、`build_task_a_new_100_5_16_5_18.py`、
`build_task_a_new_pure_1200.py`、`build_task_a_pure_1200.py`、`build_task_a_pure_vis600.py`、
`build_xvla_exp1_hard_merged.py`、`sanity_pure_vis600.py`

**↺ 原地重构(`DST = SRC`,不产生新数据集,豁免)**:
`build_task_a_pure_1200_split.py`、`build_task_a_pure_vis600_split.py`、`build_task_a_new_pure_1200_split.py`

**⚠️ 例外 / 非 Task_A 构建(不适用本规范)**:
- `build_val_kai0_official.py` → `val_kai0_official/`(val 集)
- `label_dagger_positive.py` → `Task_A/dagger_labeled/`(dagger 标注,非构建数据集)
- `split_advantage_stage.py` → 原地处理 `Task_A/advantage`
- `prepare_task_p_splits.py`、`prepare_task_e_splits.py` → 其它任务(Task_P/Task_E)

**🗑 历史遗留(输出到顶层 `Task_A_*`,已弃用,勿再使用 / 新建请改 self_built)**:
- `build_task_a_mixed.py` → `Task_A_mixed_gf1/`(老 gf1)
- `build_task_a_visrobot01_only.py`、`build_task_a_vis_base.py` → `Task_A_visrobot01_only/`

## 自动同步

- **`sync_vis_base_from_tos.sh`** ⭐ —— gf0 每小时 cron **完整增量同步**: 用 `tosutil cp -r -u` 逐日期把 TOS (`tos://transfer-shanghai/KAI0/Task_A/base/`) 同步进 `vis_base/`(build 源持续保鲜)。遍历所有日期、`-u` 按 size/crc 跳过未变(只拉新/改、从不删本地 → 保护 vis_v2_* 软链), flock 防重叠, 日志 >5MB 轮转。安装: `crontab` `0 * * * *`; 依赖 cron 守护进程(`sudo service cron start`)+ gf0 上的 `~/tosutil`。详见 `docs/.../data_sync_tos.md §6.8`。

## 工具脚本(非数据集构建)

`compute_delta_norm_stats_fast.py`(算 norm_stats)、`gen_episodes_stats.py`/`generate_episodes_stats.py`、
`get_episodes.py`、`from_tos_file.py`/`to_tos_file.py`(TOS 传输)、`redownload_bad_videos.py`、
`fix_data.py`、`pack_inference_ckpt.py`、`split_advantage_stage.py`。
