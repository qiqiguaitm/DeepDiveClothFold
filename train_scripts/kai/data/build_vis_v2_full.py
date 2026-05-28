"""Build merged vis dataset from all v2 dates under vis_base/.

Sources: /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base/{date}-v2/
Output:  /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_full/

LeRobot v2.1 layout. Globally renumber episode_index. Re-chunk to chunks_size=1000.
videos → symlinks to source (saves space).
Each parquet rewritten with renumbered episode_index + index (global frame index).
"""
from __future__ import annotations
import json, os, shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base")
DST = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_full")
CHUNKS_SIZE = 1000
PROMPT = "Flatten and fold the cloth."
TEMPLATE_ID = "task_a_base"

# Ordered by date (chronological)
DATES = sorted([d.name for d in SRC_ROOT.iterdir() if d.is_dir() and d.name.endswith("-v2")])
print(f"Source dates: {DATES}")

# Reset output
if DST.exists():
    print(f"Removing existing {DST}")
    shutil.rmtree(DST)
(DST / "data").mkdir(parents=True)
(DST / "videos").mkdir(parents=True)
(DST / "meta").mkdir(parents=True)

global_ep = 0  # running episode_index
global_frame = 0  # running frame index across all eps
total_frames = 0
episodes_records = []  # for episodes.jsonl
tasks_set = {}  # task_index -> task

# Camera keys (look at first source to discover)
def get_cams(src_date):
    src = SRC_ROOT / src_date
    info = json.load(open(src / "meta" / "info.json"))
    return sorted([k for k in info.get("features", {}) if k.startswith("observation.images.")])

CAMS = None  # set on first iter

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

    # Read src episodes.jsonl
    src_eps = []
    with open(ep_path) as f:
        for line in f:
            src_eps.append(json.loads(line))

    # Read tasks for index 0 baseline (assume single task per dataset)
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                tasks_set[t.get("task_index", 0)] = t.get("task", PROMPT)

    # Track data path template
    data_tpl = src_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_tpl = src_info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    src_chunks_size = src_info.get("chunks_size", 1000)

    n_local = len(src_eps)
    print(f"  {src_date}: {n_local} eps")
    for src_idx, src_ep in enumerate(src_eps):
        # IMPORTANT: use src_idx (sequential 0..N-1) NOT src_ep["episode_index"] which is unreliable
        # (some src dates like 05-19 have all eps with episode_index=0 in jsonl, causing duplicates)
        src_ep_index = src_idx
        src_chunk = src_ep_index // src_chunks_size
        src_parquet = src / data_tpl.format(episode_chunk=src_chunk, episode_index=src_ep_index)
        if not src_parquet.exists():
            print(f"    SKIP ep {src_ep_index} (no parquet)")
            continue

        # Re-load parquet, renumber episode_index + index
        t = pq.read_table(src_parquet).to_pandas()
        n_frames = len(t)
        t["episode_index"] = global_ep
        t["index"] = np.arange(global_frame, global_frame + n_frames, dtype=np.int64)
        # frame_index stays 0..n_frames-1 within episode

        # Determine output chunk
        dst_chunk = global_ep // CHUNKS_SIZE
        dst_data_dir = DST / "data" / f"chunk-{dst_chunk:03d}"
        dst_data_dir.mkdir(parents=True, exist_ok=True)
        dst_parquet = dst_data_dir / f"episode_{global_ep:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(t), dst_parquet)

        # Symlink videos (try both naming conventions: 'observation.images.X' vs short 'X')
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

        # Episode record — IMPORTANT: use ACTUAL parquet row count for length, not src jsonl
        # (some src jsonl have wrong length entries, esp newer dates).
        rec = dict(src_ep)
        rec["episode_index"] = global_ep
        rec["length"] = n_frames  # actual parquet rows
        rec["duration_s"] = n_frames / 30.0
        rec["_src_dir"] = src_date
        rec["_src_idx"] = src_ep_index
        episodes_records.append(rec)

        global_ep += 1
        global_frame += n_frames
        total_frames += n_frames

    if global_ep % 100 == 0 or global_ep == n_local:
        print(f"    progress: {global_ep} total eps, {total_frames} frames")

# Write meta
total_chunks = (global_ep + CHUNKS_SIZE - 1) // CHUNKS_SIZE
n_videos_per_ep = len(CAMS)
total_videos = global_ep * n_videos_per_ep

# Carry forward feature schema from first src
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

# episodes.jsonl
with open(DST / "meta" / "episodes.jsonl", "w") as f:
    for rec in episodes_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# tasks.jsonl
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
print(f"  out: {DST}")
