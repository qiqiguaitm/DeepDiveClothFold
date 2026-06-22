#!/usr/bin/env python3
"""Validate lerobot-v2.1 episode video↔action alignment (PTS-zeroing + frame/row + timestamp).

Implements the checklist from docs/deployment/training_ops/dataset_trimming_and_pts.md §4.
Use it to confirm that trimmed data (online EpisodeWriter or offline build_no_release) is
standard-aligned BEFORE training — offline MAE cannot catch a PTS/timestamp skew.

Per episode it asserts:
  (1) each camera mp4's FIRST pts == 0          (video PTS zeroed)
  (2) decoded video frame count == parquet rows (no dropped/extra frames)
  (3) parquet timestamp == frame_index / fps    (0-based, lerobot-standard axis)
  --deep also does a decode-alignment pixel check: the frame fetched BY TIMESTAMP
  (how lerobot decodes) must equal the frame fetched BY INDEX, at sampled rows.

Usage:
  kai0/.venv/bin/python train_scripts/kai/data/validate_episode_pts.py <path> [--deep]
    <path> = a dataset leaf dir (…/<date>-vN, has data/chunk-000/*.parquet + videos/…)
             OR a single episode_XXXXXX.parquet
  [--cameras top_head,hand_left,hand_right]   (default: auto-detect under videos/chunk-000/)
  [--fps N]                                   (default: from meta/info.json, else 30)
  [--limit N]                                 (validate only the first N episodes)

Exit code 0 = all OK, 1 = any failure (CI-friendly).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

DEFAULT_CAMERAS = ("top_head", "hand_left", "hand_right")


def _fps_from_info(leaf: Path, fallback: float) -> float:
    info = leaf / "meta" / "info.json"
    if info.is_file():
        try:
            d = json.loads(info.read_text())
            return float(d.get("fps", fallback))
        except Exception:  # noqa: BLE001
            pass
    return fallback


def _video_pts(mp4: Path) -> list[int]:
    with av.open(str(mp4)) as c:
        return [f.pts for f in c.decode(c.streams.video[0])]


def _frame_by_index(mp4: Path, idx: int) -> np.ndarray:
    with av.open(str(mp4)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i == idx:
                return f.to_ndarray(format="rgb24")
    raise IndexError(idx)


def _frame_by_timestamp(mp4: Path, query_s: float) -> np.ndarray:
    """Mimic lerobot: seek to query_s and take the nearest decoded frame."""
    with av.open(str(mp4)) as c:
        vs = c.streams.video[0]
        best, best_dt = None, 1e9
        tb = float(vs.time_base)
        for f in c.decode(vs):
            t = (f.pts or 0) * tb
            dt = abs(t - query_s)
            if dt < best_dt:
                best, best_dt = f, dt
            elif t > query_s:
                break
        return best.to_ndarray(format="rgb24")


def validate_episode(pq_path: Path, cameras: list[str], fps: float, deep: bool) -> list[str]:
    """Return a list of failure strings ([] = OK)."""
    fails: list[str] = []
    leaf = pq_path.parent.parent.parent          # …/<leaf>/data/chunk-000/ep.parquet → <leaf>
    vid_root = leaf / "videos" / "chunk-000"
    tbl = pq.read_table(pq_path)
    n = tbl.num_rows
    cols = set(tbl.column_names)

    # (3) timestamp == frame_index/fps
    if "timestamp" in cols:
        ts = np.asarray(tbl.column("timestamp").to_pylist(), dtype=np.float64)
        exp = np.arange(n) / fps
        if not np.allclose(ts, exp, atol=1.0 / fps / 4):   # within ¼ frame
            bad = int(np.argmax(np.abs(ts - exp)))
            fails.append(f"timestamp != frame_index/fps (row {bad}: {ts[bad]:.4f} vs {exp[bad]:.4f})")
    else:
        fails.append("no 'timestamp' column")
    if "frame_index" in cols:
        fi = np.asarray(tbl.column("frame_index").to_pylist(), dtype=np.int64)
        if not np.array_equal(fi, np.arange(n)):
            fails.append("frame_index != arange(n)")

    ep_tag = pq_path.stem.replace("episode_", "")
    for cam in cameras:
        mp4 = vid_root / cam / f"episode_{ep_tag}.mp4"
        if not mp4.is_file():
            fails.append(f"{cam}: mp4 missing ({mp4.name})")
            continue
        ptss = _video_pts(mp4)
        # (1) first pts == 0
        if not ptss or ptss[0] != 0:
            fails.append(f"{cam}: first pts={ptss[0] if ptss else None} != 0 (PTS not zeroed)")
        # (2) frames == rows
        if len(ptss) != n:
            fails.append(f"{cam}: video frames {len(ptss)} != parquet rows {n}")
        # deep: decode-by-timestamp == decode-by-index at sampled rows
        if deep and len(ptss) == n and n > 1:
            for idx in {0, n // 2, n - 1}:
                a = _frame_by_index(mp4, idx)
                b = _frame_by_timestamp(mp4, idx / fps)
                d = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                if d > 1.0:    # near-zero expected (re-encode noise tolerance)
                    fails.append(f"{cam}: decode skew at row {idx} (by-ts vs by-idx pixel Δ={d:.1f})")
    return fails


def iter_parquets(path: Path):
    if path.is_file() and path.suffix == ".parquet":
        yield path
    elif path.is_dir():
        d = path / "data" / "chunk-000"
        yield from sorted((d if d.is_dir() else path).glob("episode_*.parquet"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="dataset leaf dir OR a single episode parquet")
    ap.add_argument("--cameras", default="", help="comma list (default: auto-detect)")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--deep", action="store_true", help="also pixel-check decode-by-timestamp")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pqs = list(iter_parquets(args.path))
    if not pqs:
        print(f"[validate] no episode parquet under {args.path}", file=sys.stderr)
        return 1
    if args.limit:
        pqs = pqs[: args.limit]

    leaf = pqs[0].parent.parent.parent
    fps = args.fps or _fps_from_info(leaf, 30.0)
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    else:
        vr = leaf / "videos" / "chunk-000"
        cameras = sorted(p.name for p in vr.iterdir() if p.is_dir() and not p.name.endswith("_depth")) \
            if vr.is_dir() else list(DEFAULT_CAMERAS)

    print(f"[validate] {len(pqs)} episode(s) under {leaf}  fps={fps:g}  cameras={cameras}  deep={args.deep}")
    n_ok = 0
    for p in pqs:
        fails = validate_episode(p, cameras, fps, args.deep)
        if fails:
            print(f"  [FAIL] {p.name}")
            for f in fails:
                print(f"         - {f}")
        else:
            n_ok += 1
            print(f"  [OK ] {p.name}")
    print(f"[validate] {n_ok}/{len(pqs)} OK")
    return 0 if n_ok == len(pqs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
