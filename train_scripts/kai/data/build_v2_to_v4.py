#!/usr/bin/env python3
"""Unified v2 -> v3 + v4 builder (Task_A base/dagger), disk-aware.

Per episode, compute ONE contiguous keep window [start:end] (front motion-onset
trim + tail-cap, reusing build_no_release's rules) and apply it to parquet, the 3
RGB videos, AND the depth zarr.zip — so all stay frame-aligned.

Output topology (real files live in the NEWEST version = v4; v3 symlinks up):
  v4/<date>-v4:  REAL trimmed parquet(gripper OLD frame) + REAL trimmed 3 mp4 +
                 REAL trimmed depth zarr.zip
  v3/<date>-v3:  trimmed parquet(gripper OLD frame, == v4 for now) +
                 SYMLINK videos -> v4 + SYMLINK depth -> v4

Gripper remap (v3 OLD frame -> v4 canonical 0-70mm) is a SEPARATE phase
(make_v4_gripper_remap-style) that rewrites ONLY v4 parquet's dims 6,13 — run
after ALL v3 exist so the global [q01,q99] is over the whole corpus.

Depth trim is a fast zip-chunk copy (chunks are 1-frame; no decompress).

Usage:
  kai0/.venv/bin/python train_scripts/kai/data/build_v2_to_v4.py --src base --date 2026-06-04-v2
  kai0/.venv/bin/python train_scripts/kai/data/build_v2_to_v4.py --src base --date all
  kai0/.venv/bin/python train_scripts/kai/data/build_v2_to_v4.py --src dagger --date all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import (  # noqa: E402
    CAM_DIRS,
    FPS,
    MARGIN,
    motion_onset,
    per_episode_stats,
    tail_cap_keep_indices,
)

ROOT = Path("/data1/DATA_IMP/KAI0/Task_A")
CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
DEPTH_DIR = "top_head_depth"
DEPTH_FEATURE = "observation.depth.top_head"
ENC_PRESET = os.environ.get("BUILD_ENC_PRESET", "veryfast")
ENC_THREADS = int(os.environ.get("BUILD_ENC_THREADS", "4"))


def keep_window(action: np.ndarray) -> tuple[int, int]:
    """Contiguous [start, end) to keep: front motion-onset trim + tail-cap."""
    front_cut = max(0, motion_onset(action) - MARGIN)
    tail_keep = tail_cap_keep_indices(action[front_cut:])  # contiguous arange(L)
    start = front_cut
    end = front_cut + int(len(tail_keep))
    return start, end


def trim_video(src_mp4: Path, dst_mp4: Path, start: int, end: int):
    """Re-encode keeping frames [start, end), reset PTS. Uses ffmpeg CLI: the old
    04-23~04-29 'camera flicker' videos have glitchy frames PyAV's muxer chokes on
    (Errno 22 mid-stream); ffmpeg tolerates them. Frame-accurate via select filter."""
    import subprocess

    L = end - start
    vf = f"select=between(n\\,{start}\\,{end - 1}),setpts=N/FRAME_RATE/TB"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_mp4),
           "-vf", vf, "-vsync", "0", "-an",
           "-c:v", "libx264", "-crf", "18", "-preset", ENC_PRESET,
           "-threads", str(ENC_THREADS), str(dst_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg {src_mp4.name}: {r.stderr.strip()[:200]}")
    import av
    c = av.open(str(dst_mp4))
    n = sum(1 for _ in c.decode(video=0))
    c.close()
    if n != L:
        raise RuntimeError(f"video {src_mp4.name}: {n} frames != {L} (start={start} end={end})")


def trim_depth(src_zip: Path, dst_mkv: Path, start: int, end: int):
    """Decode depth frames [start, end) from the zarr.zip and RE-STORE as a
    lossless FFV1 gray16le .mkv: ~58% smaller than the per-frame zstd zarr,
    lossless (verified), fast (~0.5s/ep), standard container. Reader support is in
    web/.../depth_archive.py."""
    import subprocess

    import numcodecs

    with zipfile.ZipFile(src_zip) as z:
        za = json.loads(z.read(".zarray"))
        codec = numcodecs.get_codec(za["compressor"])
        _, H, W = za["shape"]
        dt = np.dtype(za["dtype"])
        frames = []
        for i in range(start, end):
            cn = f"{i}.0.0"
            try:
                fr = np.frombuffer(codec.decode(z.read(cn)), dtype=dt).reshape(H, W)
            except KeyError:
                fr = np.zeros((H, W), dtype=dt)
            frames.append(np.ascontiguousarray(fr, dtype="<u2"))
    arr = np.stack(frames)
    tmp = Path(str(dst_mkv) + ".tmp.mkv")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray16le",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "ffv1", "-level", "3", str(tmp)]
    p = subprocess.run(cmd, input=arr.tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffv1 depth {src_zip.name}: {p.stderr.decode()[:200]}")
    os.replace(tmp, dst_mkv)


def _relsym(target: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def _process_ep(job: dict) -> dict:
    """Worker: trim ONE episode (parquet + 3 videos + depth) -> v4 REAL + v3 symlinks.
    Picklable top-level fn for ProcessPoolExecutor. Returns ep meta + stats."""
    v2_dir, v3_dir, v4_dir = Path(job["v2"]), Path(job["v3"]), Path(job["v4"])
    ep, start, end, off, has_depth = job["ep"], job["start"], job["end"], job["off"], job["depth"]
    L = end - start
    df = pd.read_parquet(v2_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    sub = df.iloc[start:end].copy().reset_index(drop=True)
    sub["frame_index"] = np.arange(L, dtype=np.int64)
    sub["episode_index"] = np.int64(ep)
    sub["index"] = np.arange(off, off + L, dtype=np.int64)
    sub["timestamp"] = (np.arange(L, dtype=np.float32) / FPS).astype(np.float32)
    v4_pq = v4_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    sub.to_parquet(v4_pq, index=False)
    shutil.copy2(v4_pq, v3_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    for cam in CAMERAS:
        sv = v2_dir / "videos" / "chunk-000" / CAM_DIRS[cam] / f"episode_{ep:06d}.mp4"
        v4v = v4_dir / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4"
        trim_video(sv, v4v, start, end)
        _relsym(v4v, v3_dir / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4")
    if has_depth:
        sd = v2_dir / "videos" / "chunk-000" / DEPTH_DIR / f"episode_{ep:06d}.zarr.zip"
        if sd.is_file():
            v4d = v4_dir / "videos" / "chunk-000" / DEPTH_DIR / f"episode_{ep:06d}.mkv"
            trim_depth(sd, v4d, start, end)
            _relsym(v4d, v3_dir / "videos" / "chunk-000" / DEPTH_DIR / f"episode_{ep:06d}.mkv")
    return {"date": v4_dir.name[:-3], "ep": ep, "len": L,
            "prompt": job["prompt"], "stats": per_episode_stats(sub)}


def _write_meta(v2_dir: Path, v3_dir: Path, v4_dir: Path, has_depth: bool, results: dict):
    eps = sorted(results)
    total = sum(results[e]["len"] for e in eps)
    for vd in (v4_dir, v3_dir):
        (vd / "meta").mkdir(parents=True, exist_ok=True)
        with (vd / "meta" / "episodes.jsonl").open("w") as f:
            for e in eps:
                f.write(json.dumps({"episode_index": e, "tasks": [results[e]["prompt"]],
                                    "length": results[e]["len"]}) + "\n")
        with (vd / "meta" / "episodes_stats.jsonl").open("w") as f:
            for e in eps:
                f.write(json.dumps({"episode_index": e, "stats": results[e]["stats"]}) + "\n")
        shutil.copy2(v2_dir / "meta" / "tasks.jsonl", vd / "meta" / "tasks.jsonl")
        info = json.loads((v2_dir / "meta" / "info.json").read_text())
        info["total_episodes"] = len(eps)
        info["total_frames"] = total
        info["total_videos"] = len(eps) * len(CAMERAS)
        info["total_chunks"] = 1
        info["chunks_size"] = max(1000, len(eps))
        info["splits"] = {"train": f"0:{len(eps)}"}
        if not has_depth:
            info["features"].pop(DEPTH_FEATURE, None)
            info.pop("depth_path", None)
        else:
            info["depth_path"] = ("videos/chunk-{episode_chunk:03d}/{video_key}_depth/"
                                  "episode_{episode_index:06d}.mkv")
            feat = info["features"].get(DEPTH_FEATURE, {})
            feat["dtype"] = "uint16_ffv1"
            feat["info"] = {"container": "matroska", "codec": "ffv1", "pix_fmt": "gray16le",
                            "unit": "millimeter", "depth.height": 480, "depth.width": 640, "depth.fps": FPS}
            info["features"][DEPTH_FEATURE] = feat
        (vd / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def build_all(date_dirs, workers: int):
    """Pass1: per-ep trim window + index offset (fast). Pass2: ONE process pool over
    ALL episodes across ALL dates (max core use). Pass3: per-date meta."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    meta = {}  # base -> (v2_dir, v3_dir, v4_dir, has_depth, results{})
    for v2_dir in date_dirs:
        base = v2_dir.name[:-3]
        subset_v = v2_dir.parent.parent
        v3_dir = subset_v / "v3" / f"{base}-v3"
        v4_dir = subset_v / "v4" / f"{base}-v4"
        parquets = sorted((v2_dir / "data" / "chunk-000").glob("episode_*.parquet"))
        has_depth = (v2_dir / "videos" / "chunk-000" / DEPTH_DIR).is_dir()
        src_eps = {}
        em = v2_dir / "meta" / "episodes.jsonl"
        if em.exists():
            for line in em.open():
                e = json.loads(line)
                src_eps[int(e.get("episode_index", e.get("episode_id", -1)))] = e
        (v4_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (v3_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (v4_dir / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)
            (v3_dir / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)
        if has_depth:
            (v4_dir / "videos" / "chunk-000" / DEPTH_DIR).mkdir(parents=True, exist_ok=True)
            (v3_dir / "videos" / "chunk-000" / DEPTH_DIR).mkdir(parents=True, exist_ok=True)
        off = 0
        for pqf in parquets:
            ep = int(pqf.stem.split("_")[1])
            action = np.stack(pd.read_parquet(pqf, columns=["action"])["action"].values)
            start, end = keep_window(action)
            jobs.append({"v2": str(v2_dir), "v3": str(v3_dir), "v4": str(v4_dir), "ep": ep,
                         "start": start, "end": end, "off": off, "depth": has_depth,
                         "prompt": src_eps.get(ep, {}).get("prompt", "Flatten and fold the cloth.")})
            off += end - start
        meta[base] = (v2_dir, v3_dir, v4_dir, has_depth, {})
    print(f"pass1: {len(jobs)} episode jobs over {len(meta)} dates; pass2 with {workers} workers", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_ep, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            meta[r["date"]][4][r["ep"]] = r
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} eps done", flush=True)
    for base, (v2_dir, v3_dir, v4_dir, has_depth, results) in meta.items():
        _write_meta(v2_dir, v3_dir, v4_dir, has_depth, results)
        print(f"  meta {base}: {len(results)} eps", flush=True)
    print("ALL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["base", "dagger", "both"], default="both")
    ap.add_argument("--date", default="all", help="<date>-v2 or 'all'")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("BUILD_WORKERS", "20")))
    args = ap.parse_args()
    srcs = ["base", "dagger"] if args.src == "both" else [args.src]
    date_dirs = []
    for s in srcs:
        v2root = ROOT / s / "v2"
        if args.date == "all":
            date_dirs += sorted(d for d in v2root.iterdir() if d.is_dir() and d.name.endswith("-v2"))
        else:
            date_dirs.append(v2root / args.date)
    print(f"v2->v3+v4: {len(date_dirs)} datasets ({args.src}), {args.workers} workers")
    build_all(date_dirs, args.workers)


if __name__ == "__main__":
    main()
