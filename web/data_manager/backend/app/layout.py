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
import os
import re
from pathlib import Path

from .config import DATA_ROOT

# Date string format on disk: YYYY-MM-DD optionally followed by "-vN".
#
# NOTE: two independent "version" axes share this suffix slot:
#   * disk LAYOUT version — -v2 = subset-above-date tree (switched 2026-05-09).
#   * dataset CONTENT version — -v3 = the 2026-06-15 capture changes (online
#     front-trim + gripper-action-from-master). All new captures use the v2
#     disk layout; the suffix now tracks the CONTENT version so V3 datasets are
#     distinguishable on disk from older -v2 captures.
#
# The suffix is chosen by the collect scripts (start_data_collect.sh /
# start_dagger_collect.sh) via KAI0_DATE_SUFFIX and read at call time, so the
# leaf dir always matches the bytes being written. Default "-v2" keeps any other
# caller (and legacy V2 capture) unchanged. Read paths accept any -vN (regex
# below), so older -v2 datasets still resolve.
DEFAULT_DATE_SUFFIX = "-v2"


def new_date_suffix() -> str:
    """Leaf date suffix for new captures (KAI0_DATE_SUFFIX, default -v2)."""
    return os.environ.get("KAI0_DATE_SUFFIX", DEFAULT_DATE_SUFFIX)


# Back-compat alias (evaluated at import; prefer new_date_suffix() for fresh env).
NEW_DATE_SUFFIX = new_date_suffix()

# "任意 task 名_YYYY-MM-DD[-v2]" — 锚在末尾, 避免误吞 task 名里的 '_'
_DATE_SUFFIX_RE = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2}(?:-v\d+)?)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:-v\d+)?$")


def today_compound(task: str) -> str:
    """`'Task_A'` → `'Task_A_2026-04-16-v3'` (today date + current date suffix)."""
    return f"{task}_{datetime.date.today().strftime('%Y-%m-%d')}{new_date_suffix()}"


def split_compound(compound: str) -> tuple[str, str] | None:
    """`'Task_A_2026-04-16'` → `('Task_A', '2026-04-16')`; 没有日期后缀 → None."""
    m = _DATE_SUFFIX_RE.match(compound)
    return (m.group(1), m.group(2)) if m else None


def _ver_of(date: str) -> str | None:
    """`'2026-06-15-v3'` → `'v3'`; bare date (no suffix) → None."""
    m = re.search(r"-(v\d+)$", date)
    return m.group(1) if m else None


def _vnest_path(task: str, subset: str, date: str) -> Path:
    """vN-nested layout (2026-06-15): `<task>/<subset>/<vN>/<date>-vN`.

    A version dir groups all captures of one content version (v2 legacy, v3 =
    front-trim + gripper-from-master) so they can be trained on separately. The
    date leaf keeps its -vN suffix. Dates with no -vN suffix fall back to the
    flat `<task>/<subset>/<date>` (legacy / unversioned)."""
    ver = _ver_of(date)
    return DATA_ROOT / task / subset / ver / date if ver else DATA_ROOT / task / subset / date


def _v2_path(task: str, subset: str, date: str) -> Path:
    return DATA_ROOT / task / subset / date


def _v1_path(task: str, subset: str, date: str) -> Path:
    return DATA_ROOT / task / date / subset


def _v0_path(task: str, subset: str, date: str) -> Path:
    return DATA_ROOT / f"{task}_{date}" / subset


def compound_to_subset_root(compound: str, subset: str) -> Path:
    """`'Task_A_2026-04-16', 'base'` → 该 episode 子目录的真实磁盘路径.

    优先级: vnest (`<task>/<subset>/<vN>/<date>`) → v2 (`<task>/<subset>/<date>`)
            → v1 (`<task>/<date>/<subset>`) → v0 (`<task>_<date>/<subset>`)
            → vnest (默认, caller mkdir).

    无日期后缀的 compound (训练产出的 'Task_A' 等) 直接 `DATA_ROOT/<compound>/<subset>`,
    不属于本布局体系, 走老路。
    """
    sp = split_compound(compound)
    if sp is None:
        return DATA_ROOT / compound / subset
    task, date = sp
    vnest = _vnest_path(task, subset, date)
    if vnest.exists():
        return vnest
    v2 = _v2_path(task, subset, date)
    if v2.exists():
        return v2
    v1 = _v1_path(task, subset, date)
    if v1.exists():
        return v1
    v0 = _v0_path(task, subset, date)
    if v0.exists():
        return v0
    return vnest


def new_task_subset_root(task: str, subset: str) -> Path:
    """写新 episode 用: vN-nested layout + current date suffix
    (`<DATA_ROOT>/<task>/<subset>/<vN>/<today>-vN`, e.g. base/v3/2026-06-15-v3).
    The version dir keeps each content version in its own subtree and is created
    on the fly when the recorder mkdir's the path. Version folder = explicit
    KAI0_DATASET_VERSION if set (collect scripts set it), else derived from the
    date suffix. Read-side (compound_to_subset_root) derives version from the
    date and is unaffected by these env vars."""
    date = datetime.date.today().strftime("%Y-%m-%d") + new_date_suffix()
    ver = os.environ.get("KAI0_DATASET_VERSION", "").strip() or _ver_of(date)
    return DATA_ROOT / task / subset / ver / date if ver else DATA_ROOT / task / subset / date


def path_to_compound(p: Path) -> tuple[str, str] | None:
    """盘上一个 parquet 文件路径 → `(compound_task_id, subset)`.

    支持:
      vnest `.../TASK/SUBSET/VER/DATE/data/chunk-000/episode_NNNNNN.parquet`
      v2    `.../TASK/SUBSET/DATE/data/chunk-000/episode_NNNNNN.parquet`
      v1    `.../TASK/DATE/SUBSET/data/chunk-000/episode_NNNNNN.parquet`
      v0    `.../TASK_DATE/SUBSET/data/chunk-000/episode_NNNNNN.parquet`
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
    if len(prefix) == 4:  # vnest: (task, subset, ver, date)
        a, b, c, d = prefix
        if re.fullmatch(r"v\d+", c) and _DATE_RE.match(d):
            return f"{a}_{d}", b
    return None


def glob_all_episodes():
    """遍历 DATA_ROOT, yield 每个 episode.parquet 的 Path (v0 + v1 + v2 一起)."""
    # v0 扁平:   <DATA_ROOT>/*/*/data/chunk-*/episode_*.parquet            (深 5)
    # v1 按日期: <DATA_ROOT>/*/*/*/data/chunk-*/episode_*.parquet          (深 6)
    # v2 按子集: <DATA_ROOT>/*/*/*/data/chunk-*/episode_*.parquet          (深 6)
    # v1 / v2 同深度同 glob, 一次返回, path_to_compound 内部按 X 是否是日期区分.
    # vnest 按版本: <DATA_ROOT>/<task>/<subset>/<vN>/<date>/data/chunk-*/...   (深 7)
    yield from DATA_ROOT.glob("*/*/data/chunk-*/episode_*.parquet")
    yield from DATA_ROOT.glob("*/*/*/data/chunk-*/episode_*.parquet")
    yield from DATA_ROOT.glob("*/*/*/*/data/chunk-*/episode_*.parquet")
