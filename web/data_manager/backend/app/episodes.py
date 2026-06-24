"""历史 episode 浏览/读取（采集员可用）。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .config import DATA_ROOT
from .layout import compound_to_subset_root
from .stats_service import service as stats

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _safe_tail(*parts: str) -> tuple[str, ...]:
    """URL 传进来的每段都要过白名单, 防 path traversal。返回原样 tuple 供 caller 拼。"""
    for p in parts:
        if not SAFE_NAME.match(p):
            raise HTTPException(status_code=400, detail=f"unsafe path component: {p}")
    return parts


def _subset_join(task_id: str, subset: str, *tail: str) -> Path:
    """`Task_A_2026-04-16, base, 'videos', 'chunk-000', ...` → 真实磁盘 Path,
    透明支持新层级 / 老扁平布局。同时做 traversal 校验。"""
    _safe_tail(task_id, subset, *tail)
    base = compound_to_subset_root(task_id, subset)
    full = base.joinpath(*tail).resolve()
    if not str(full).startswith(str(DATA_ROOT)):
        raise HTTPException(status_code=400, detail="path escapes DATA_ROOT")
    return full


def episode_video_path(task_id: str, subset: str, episode_id: int, camera: str) -> Path:
    if camera not in ("top_head", "hand_left", "hand_right"):
        raise HTTPException(status_code=400, detail="unknown camera")
    return _subset_join(task_id, subset, "videos", "chunk-000", camera, f"episode_{episode_id:06d}.mp4")


def episode_depth_zarr_path(task_id: str, subset: str, episode_id: int, camera: str) -> Path:
    # 新录制只为 D435 头顶相机生成 depth zarr (见 recorder.DEPTH_CAMERAS);
    # hand_left / hand_right 仅为兼容历史数据保留可达路径, 新数据下这两个 cam
    # 走到 main.py 的 file_exists 检查即返回 404, 不会泄露其他目录.
    if camera not in ("top_head", "hand_left", "hand_right"):
        raise HTTPException(status_code=400, detail="unknown camera")
    return _subset_join(task_id, subset, "videos", "chunk-000", f"{camera}_depth", f"episode_{episode_id:06d}.zarr")


def episode_meta(task_id: str, subset: str, episode_id: int) -> dict:
    fp = _subset_join(task_id, subset, "meta", "episodes.jsonl")
    if not fp.exists():
        raise HTTPException(status_code=404, detail="meta not found")
    for line in fp.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(d.get("episode_id", -1)) == episode_id:
            return d
    raise HTTPException(status_code=404, detail="episode not found")


def delete_episode(task_id: str, subset: str, episode_id: int) -> None:
    import shutil
    pq = _subset_join(task_id, subset, "data", "chunk-000", f"episode_{episode_id:06d}.parquet")
    if pq.exists():
        pq.unlink()
    for cam in ("top_head", "hand_left", "hand_right"):
        v = _subset_join(task_id, subset, "videos", "chunk-000", cam, f"episode_{episode_id:06d}.mp4")
        if v.exists():
            v.unlink()
        # depth: FFV1 .mkv (新), 单文件 .zarr.zip, 旧格式 .zarr 目录, 全清
        d = _subset_join(task_id, subset, "videos", "chunk-000", f"{cam}_depth", f"episode_{episode_id:06d}.zarr")
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        dz = d.with_name(d.name + ".zip")
        if dz.exists():
            dz.unlink()
        dmkv = d.with_name(d.name.removesuffix(".zarr") + ".mkv")
        if dmkv.exists():
            dmkv.unlink()
    # 同步从 meta 中删除该条
    meta_fp = _subset_join(task_id, subset, "meta", "episodes.jsonl")
    if meta_fp.exists():
        keep: list[str] = []
        for line in meta_fp.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if int(d.get("episode_id", -1)) == episode_id:
                    continue
            except json.JSONDecodeError:
                pass
            keep.append(line)
        meta_fp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    stats.remove_by_path(pq)
