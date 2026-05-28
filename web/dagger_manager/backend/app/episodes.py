"""History episode browsing + replay for dagger_manager.

Reads the Form C dual-dataset layout written by dagger_recorder_node:
    <DATA_ROOT>/<task>/<subset>/<date>-v2/
        ├── data/chunk-000/episode_NNNNNN.parquet
        ├── videos/chunk-000/<camera>/episode_NNNNNN.mp4   (AV1)
        └── meta/episodes.jsonl

episode_id resets per date dir, so an episode is uniquely keyed by
(subset, date, episode_id). subset ∈ {dagger, inference}.

All path building goes through _safe_join which (1) whitelists each URL
component and (2) verifies the resolved path stays under DATA_ROOT — same
traversal defense as data_manager/episodes.py.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .stack import DATA_ROOT

CAMERAS = ("top_head", "hand_left", "hand_right")
SUBSETS = ("dagger", "inference")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-.]+$")


def _safe(*parts: str) -> None:
    for p in parts:
        if not SAFE_NAME.match(p):
            raise HTTPException(400, f"unsafe path component: {p!r}")


def _date_root(task: str, subset: str, date: str) -> Path:
    _safe(task, subset, date)
    if subset not in SUBSETS:
        raise HTTPException(400, f"unknown subset {subset!r}")
    full = (DATA_ROOT / task / subset / date).resolve()
    if not str(full).startswith(str(DATA_ROOT.resolve())):
        raise HTTPException(400, "path escapes DATA_ROOT")
    return full


def list_tasks() -> list[dict]:
    """Every Task_* dir under DATA_ROOT, with a flag for whether it has any
    dagger/inference data (so the UI can grey out empty tasks)."""
    out: list[dict] = []
    if not DATA_ROOT.is_dir():
        return out
    for d in sorted(DATA_ROOT.iterdir()):
        if d.is_dir() and d.name.startswith("Task_"):
            has_data = (d / "dagger").is_dir() or (d / "inference").is_dir()
            out.append({"task": d.name, "has_data": has_data})
    return out


def list_episodes(task: str = "Task_A") -> list[dict]:
    """Enumerate every episode across both subsets and all date dirs.

    Sorted newest-first by (date desc, episode_id desc) so the most recent
    captures bubble to the top of the UI list.
    """
    out: list[dict] = []
    task_root = DATA_ROOT / task
    if not task_root.is_dir():
        return out
    for subset in SUBSETS:
        subset_root = task_root / subset
        if not subset_root.is_dir():
            continue
        for date_dir in sorted(subset_root.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            meta_fp = date_dir / "meta" / "episodes.jsonl"
            if not meta_fp.is_file():
                continue
            video_dir = date_dir / "videos" / "chunk-000"
            for line in meta_fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = int(d.get("episode_id", -1))
                if ep < 0:
                    continue
                # Has at least the head-cam mp4? (used to grey out broken rows)
                head_mp4 = video_dir / "top_head" / f"episode_{ep:06d}.mp4"
                out.append({
                    "subset": subset,
                    "date": date_dir.name,
                    "episode_id": ep,
                    "length": int(d.get("length", 0)),
                    "duration_s": float(d.get("duration_s", 0.0)),
                    "operator": d.get("operator", ""),
                    "prompt": d.get("prompt", ""),
                    "success": bool(d.get("success", True)),
                    "note": d.get("note", ""),
                    "created_at": d.get("created_at"),
                    "has_video": head_mp4.is_file(),
                })
    out.sort(key=lambda e: (e["date"], e["episode_id"]), reverse=True)
    return out


def episode_video_path(task: str, subset: str, date: str, episode_id: int,
                       camera: str) -> Path:
    if camera not in CAMERAS:
        raise HTTPException(400, f"unknown camera {camera!r}")
    root = _date_root(task, subset, date)
    return root / "videos" / "chunk-000" / camera / f"episode_{episode_id:06d}.mp4"


def episode_meta(task: str, subset: str, date: str, episode_id: int) -> dict:
    meta_fp = _date_root(task, subset, date) / "meta" / "episodes.jsonl"
    if not meta_fp.is_file():
        raise HTTPException(404, "meta not found")
    for line in meta_fp.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(d.get("episode_id", -1)) == episode_id:
            return d
    raise HTTPException(404, "episode not found")


def delete_episode(task: str, subset: str, date: str, episode_id: int) -> None:
    """Remove parquet + per-camera mp4 + meta line. Irreversible."""
    root = _date_root(task, subset, date)
    pq = root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    if pq.exists():
        pq.unlink()
    for cam in CAMERAS:
        v = root / "videos" / "chunk-000" / cam / f"episode_{episode_id:06d}.mp4"
        if v.exists():
            v.unlink()
        # depth zarr (dir) — only top_head emits it, but try all for safety
        dz = root / "videos" / "chunk-000" / f"{cam}_depth" / f"episode_{episode_id:06d}.zarr"
        if dz.exists():
            shutil.rmtree(dz, ignore_errors=True)
    meta_fp = root / "meta" / "episodes.jsonl"
    if meta_fp.is_file():
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
