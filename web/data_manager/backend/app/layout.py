"""On-disk layout helpers.

历史:
  v0 (扁平):   `<DATA_ROOT>/<Task_X_YYYY-MM-DD>/<subset>/...`
  v1 (按日期): `<DATA_ROOT>/<Task_X>/<YYYY-MM-DD>/<subset>/...`
  v2 (按子集): `<DATA_ROOT>/<Task_X>/<subset>/<YYYY-MM-DD>/...`   ← 当前

v2 把 subset 抬到日期之上, 让同一种数据 (base / dagger) 跨日期连成一棵树,
方便整 subset 的训练 / 同步 / 备份。

约定:
  * 内存 / SQLite / API URL / UI 里继续用 **compound** 形式 "Task_X_YYYY-MM-DD"
    作 task_id (最少改动面), 只是磁盘路径不同。
  * 所有构造/解析盘路径的点必须过 `compound_to_subset_root` / `path_to_compound`,
    不要直接 `DATA_ROOT / task_id`。
  * 写新 episode 一律走 v2; 读既有 episode 时 v2 → v1 → v0 优先级回退,
    支持"一部分已迁移, 一部分还没"的中间态。
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

from .config import DATA_ROOT

# Date string format on disk: YYYY-MM-DD optionally followed by "-v2".
# Convention switched 2026-05-09: new captures write "<date>-v2" leaf dir
# (matches the post-process / "finalized" tag that 4-23..4-30 already used).
# Bare-date dirs from before the switch still resolve through compound_to_subset_root.
NEW_DATE_SUFFIX = "-v2"

# "任意 task 名_YYYY-MM-DD[-v2]" — 锚在末尾, 避免误吞 task 名里的 '_'
_DATE_SUFFIX_RE = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2}(?:-v\d+)?)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:-v\d+)?$")


def today_compound(task: str) -> str:
    """`'Task_A'` → `'Task_A_2026-04-16-v2'` (today date + NEW_DATE_SUFFIX)."""
    return f"{task}_{datetime.date.today().strftime('%Y-%m-%d')}{NEW_DATE_SUFFIX}"


def split_compound(compound: str) -> tuple[str, str] | None:
    """`'Task_A_2026-04-16'` → `('Task_A', '2026-04-16')`; 没有日期后缀 → None."""
    m = _DATE_SUFFIX_RE.match(compound)
    return (m.group(1), m.group(2)) if m else None


def _v2_path(task: str, subset: str, date: str) -> Path:
    return DATA_ROOT / task / subset / date


def _v1_path(task: str, subset: str, date: str) -> Path:
    return DATA_ROOT / task / date / subset


def _v0_path(task: str, subset: str, date: str) -> Path:
    return DATA_ROOT / f"{task}_{date}" / subset


def compound_to_subset_root(compound: str, subset: str) -> Path:
    """`'Task_A_2026-04-16', 'base'` → 该 episode 子目录的真实磁盘路径.

    优先级: v2 (`<task>/<subset>/<date>`) → v1 (`<task>/<date>/<subset>`)
            → v0 (`<task>_<date>/<subset>`) → v2 (默认, caller mkdir).

    无日期后缀的 compound (训练产出的 'Task_A' 等) 直接 `DATA_ROOT/<compound>/<subset>`,
    不属于本布局体系, 走老路。
    """
    sp = split_compound(compound)
    if sp is None:
        return DATA_ROOT / compound / subset
    task, date = sp
    v2 = _v2_path(task, subset, date)
    if v2.exists():
        return v2
    v1 = _v1_path(task, subset, date)
    if v1.exists():
        return v1
    v0 = _v0_path(task, subset, date)
    if v0.exists():
        return v0
    return v2


def new_task_subset_root(task: str, subset: str) -> Path:
    """写新 episode 用: v2 disk layout + `-v2` date suffix
    (`<DATA_ROOT>/<task>/<subset>/<today>-v2`). The suffix matches the
    finalized-data convention applied to 4-23..4-30 historically and
    retro-applied to 5-06..5-09 on 2026-05-11."""
    return _v2_path(task, subset,
                    datetime.date.today().strftime("%Y-%m-%d") + NEW_DATE_SUFFIX)


def path_to_compound(p: Path) -> tuple[str, str] | None:
    """盘上一个 parquet 文件路径 → `(compound_task_id, subset)`.

    支持:
      v2  `.../TASK/SUBSET/DATE/data/chunk-000/episode_NNNNNN.parquet`
      v1  `.../TASK/DATE/SUBSET/data/chunk-000/episode_NNNNNN.parquet`
      v0  `.../TASK_DATE/SUBSET/data/chunk-000/episode_NNNNNN.parquet`
    None 表示非法路径 (不在 DATA_ROOT 下 / 层数不对 / 非 episode).

    v1 / v2 都是 task/X/Y 三段, 按 X 是否是 YYYY-MM-DD 区分:
      - X 是日期 → v1 (date 在中间)
      - X 不是日期 → v2 (subset 在中间)
    """
    try:
        rel = p.resolve().relative_to(DATA_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    # 至少要有 data/chunk-*/*.parquet 这三段
    if len(parts) < 4 or parts[-3] != "data" or not parts[-2].startswith("chunk-"):
        return None
    # 前缀: ... / TASK [/ X] [/ Y] / data, 其中前缀 2 段=v0, 3 段=v1 or v2
    prefix = parts[:-3]
    if len(prefix) == 2:  # v0: (compound, subset)
        compound, subset = prefix
        return compound, subset
    if len(prefix) == 3:
        a, b, c = prefix
        if _DATE_RE.match(b):  # v1: (task, date, subset)
            return f"{a}_{b}", c
        if _DATE_RE.match(c):  # v2: (task, subset, date)
            return f"{a}_{c}", b
    return None


def glob_all_episodes():
    """遍历 DATA_ROOT, yield 每个 episode.parquet 的 Path (v0 + v1 + v2 一起)."""
    # v0 扁平:   <DATA_ROOT>/*/*/data/chunk-*/episode_*.parquet            (深 5)
    # v1 按日期: <DATA_ROOT>/*/*/*/data/chunk-*/episode_*.parquet          (深 6)
    # v2 按子集: <DATA_ROOT>/*/*/*/data/chunk-*/episode_*.parquet          (深 6)
    # v1 / v2 同深度同 glob, 一次返回, path_to_compound 内部按 X 是否是日期区分.
    yield from DATA_ROOT.glob("*/*/data/chunk-*/episode_*.parquet")
    yield from DATA_ROOT.glob("*/*/*/data/chunk-*/episode_*.parquet")
