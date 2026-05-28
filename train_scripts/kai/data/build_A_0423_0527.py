"""Build A_0423_0527 dataset = TOS KAI0/Task_A/base/ all dates 04-23..05-27
EXCEPT calibration-drift period (05-16, 05-18..21).

Filters applied (按 README §3.3 + analysis CSVs):
  - EXCLUDE_DATES: 05-16 (stay-still ideal, D1) + 05-18..05-21 (gripper firmware calibration drift, v7)
  - EXCLUDE_CLASSC: 129 ep with |Δaction|>0.5 rad (CAN dropouts, D7)
  - TRIM end-snap 5 ep to T_new frames (末段归零 artifact, D8)

Source: /transfer-shanghai/KAI0/Task_A/base/{date}-v2/  (TOS-mounted)
Output: /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0423_0527/

LeRobot v2.1 layout. Globally renumber episode_index. Re-chunk to chunks_size=1000.
videos → symlinks to source (saves space).
Each parquet rewritten with renumbered episode_index + index, optional [:T_keep] truncation.

Plan: docs/training/future_plans/plans/A_0423_0527_excl_calibration_drift.md
"""
from __future__ import annotations
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base")
DST = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0423_0527")
# Analysis CSVs (Class C blacklist + End-snap trim) only on TOS; gf0 has /transfer-shanghai mount.
ANALYSIS = Path("/transfer-shanghai/KAI0/Task_A/base/analysis")
CHUNKS_SIZE = 1000
PROMPT = "Flatten and fold the cloth."
TEMPLATE_ID = "task_a_base"

# Dates 排除: 5-16 (D1 stay-still) + 5-18~5-21 (D7 校准漂移期)
EXCLUDE_DATES = {
    "2026-05-16-v2",
    "2026-05-18-v2", "2026-05-19-v2", "2026-05-20-v2", "2026-05-21-v2",
}

# Class C 黑名单 (analysis/07_classC_blacklist.csv, 129 ep)
EXCLUDE_CLASSC: set[tuple[str, str]] = set()
with open(ANALYSIS / "07_classC_blacklist.csv") as f:
    for row in csv.DictReader(f):
        EXCLUDE_CLASSC.add((row["date"], row["ep"]))
print(f"Loaded Class C blacklist: {len(EXCLUDE_CLASSC)} (date, ep_filename) pairs")

# End-snap trim (analysis/06_end_snap_trim.csv, 5 ep)
TRIM_MAP: dict[tuple[str, str], int] = {}
with open(ANALYSIS / "06_end_snap_trim.csv") as f:
    for row in csv.DictReader(f):
        TRIM_MAP[(row["date"], row["ep"])] = int(row["T_new"])
print(f"Loaded End-snap trim map: {len(TRIM_MAP)} (date, ep_filename) → keep_frames")

# Dates: 排除后剩余 (按字典序自然按 chronological)
ALL_DATES = sorted([d.name for d in SRC_ROOT.iterdir() if d.is_dir() and d.name.endswith("-v2")])
DATES = [d for d in ALL_DATES if d not in EXCLUDE_DATES]
print(f"Source dates after exclude: {DATES}  ({len(DATES)} dates)")
print(f"  Excluded: {sorted(EXCLUDE_DATES & set(ALL_DATES))}")

# Reset output
if DST.exists():
    print(f"Removing existing {DST}")
    shutil.rmtree(DST)
(DST / "data").mkdir(parents=True)
(DST / "videos").mkdir(parents=True)
(DST / "meta").mkdir(parents=True)

global_ep = 0
global_frame = 0
total_frames = 0
n_skip_classC = 0
n_trim = 0
episodes_records = []
tasks_set: dict[int, str] = {}
CAMS = None

for src_date in DATES:
    src = SRC_ROOT / src_date
    info_path = src / "meta" / "info.json"
    ep_path = src / "meta" / "episodes.jsonl"
    tasks_path = src / "meta" / "tasks.jsonl"

    if not info_path.exists() or not ep_path.exists():
        print(f"  SKIP {src_date} (missing meta)")
        continue

    src_info = json.load(open(info_path))
    if CAMS is None:
        CAMS = sorted([k for k in src_info.get("features", {}) if k.startswith("observation.images.")])
        print(f"Cameras (from first src): {CAMS}")

    src_eps = []
    with open(ep_path) as f:
        for line in f:
            src_eps.append(json.loads(line))

    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                tasks_set[t.get("task_index", 0)] = t.get("task", PROMPT)

    data_tpl = src_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    src_chunks_size = src_info.get("chunks_size", 1000)

    n_local = len(src_eps)
    n_added_local = 0
    n_classC_local = 0
    n_trim_local = 0
    for src_idx, src_ep in enumerate(src_eps):
        src_ep_index = src_idx
        src_chunk = src_ep_index // src_chunks_size
        src_parquet = src / data_tpl.format(episode_chunk=src_chunk, episode_index=src_ep_index)
        if not src_parquet.exists():
            continue

        ep_filename = src_parquet.name  # "episode_NNNNNN.parquet"
        key = (src_date, ep_filename)

        # Filter 1: Class C 黑名单
        if key in EXCLUDE_CLASSC:
            n_classC_local += 1
            n_skip_classC += 1
            continue

        # Load parquet
        t = pq.read_table(src_parquet).to_pandas()

        # Filter 2: End-snap 截尾 (如适用)
        if key in TRIM_MAP:
            T_keep = TRIM_MAP[key]
            if T_keep < len(t):
                t = t.iloc[:T_keep].reset_index(drop=True)
                n_trim_local += 1
                n_trim += 1

        n_frames = len(t)
        if n_frames == 0:
            continue

        t["episode_index"] = global_ep
        t["index"] = np.arange(global_frame, global_frame + n_frames, dtype=np.int64)

        # Output chunk
        dst_chunk = global_ep // CHUNKS_SIZE
        dst_data_dir = DST / "data" / f"chunk-{dst_chunk:03d}"
        dst_data_dir.mkdir(parents=True, exist_ok=True)
        dst_parquet = dst_data_dir / f"episode_{global_ep:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(t), dst_parquet)

        # Symlink videos
        for cam in CAMS:
            cam_short = cam.replace("observation.images.", "")
            candidates = [
                src / "videos" / f"chunk-{src_chunk:03d}" / cam / f"episode_{src_ep_index:06d}.mp4",
                src / "videos" / f"chunk-{src_chunk:03d}" / cam_short / f"episode_{src_ep_index:06d}.mp4",
            ]
            src_video = next((p for p in candidates if p.exists()), None)
            if not src_video:
                continue
            dst_video_dir = DST / "videos" / f"chunk-{dst_chunk:03d}" / cam
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_video_dir / f"episode_{global_ep:06d}.mp4"
            os.symlink(src_video, dst_video)

        # Episode record
        rec = dict(src_ep)
        rec["episode_index"] = global_ep
        rec["length"] = n_frames
        rec["duration_s"] = n_frames / 30.0
        rec["_src_dir"] = src_date
        rec["_src_idx"] = src_ep_index
        if key in TRIM_MAP:
            rec["_trimmed"] = True
            rec["_orig_length"] = int(pq.read_table(src_parquet).num_rows)
        episodes_records.append(rec)

        global_ep += 1
        global_frame += n_frames
        total_frames += n_frames
        n_added_local += 1

    print(f"  {src_date}: {n_added_local} eps added (skipped {n_classC_local} Class C, trimmed {n_trim_local} end-snap)")

# Write meta
total_chunks = (global_ep + CHUNKS_SIZE - 1) // CHUNKS_SIZE
total_videos = global_ep * (len(CAMS) if CAMS else 0)
src0_info = json.load(open(SRC_ROOT / DATES[0] / "meta" / "info.json"))
new_info = {
    "codebase_version": src0_info.get("codebase_version", "v2.1"),
    "robot_type": src0_info.get("robot_type", "agilex"),
    "total_episodes": global_ep,
    "total_frames": total_frames,
    "total_tasks": len(tasks_set) or 1,
    "total_videos": total_videos,
    "total_chunks": total_chunks,
    "chunks_size": CHUNKS_SIZE,
    "fps": src0_info.get("fps", 30),
    "splits": {"train": f"0:{global_ep}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": src0_info.get("features", {}),
}
json.dump(new_info, open(DST / "meta" / "info.json", "w"), indent=2)

with open(DST / "meta" / "episodes.jsonl", "w") as f:
    for rec in episodes_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if not tasks_set:
    tasks_set = {0: PROMPT}
with open(DST / "meta" / "tasks.jsonl", "w") as f:
    for ti in sorted(tasks_set):
        f.write(json.dumps({"task_index": ti, "task": tasks_set[ti]}, ensure_ascii=False) + "\n")

print(f"\n=== DONE ===")
print(f"  total_episodes: {global_ep}")
print(f"  total_frames:   {total_frames}")
print(f"  total_chunks:   {total_chunks}")
print(f"  total_videos:   {total_videos}")
print(f"  classC skipped: {n_skip_classC}")
print(f"  end-snap trim:  {n_trim}")
print(f"  out: {DST}")
print(f"\nNext: compute norm_stats")
print(f"  cd /path/to/kai0 && .venv/bin/python scripts/compute_norm_stats.py "
      f"--dataset-root {DST} --output {DST}/meta/norm_stats.json")
