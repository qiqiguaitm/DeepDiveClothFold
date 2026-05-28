#!/usr/bin/env python3
"""Merge all vis_base/*-v2 dirs into single lerobot dataset with contiguous 0..N-1 indices.

Use symlinks to data + videos (zero copy). Re-index everything.

Usage: python build_vis_v2_merged.py
Output: /vePFS/.../Task_A/vis_v2_merged/  (single lerobot dataset, 895 ep)
"""
from __future__ import annotations
import json, sys, os, shutil
from pathlib import Path

CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
SRC_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base")
DST = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged")


def main():
    if DST.exists():
        sys.exit(f"dst exists: {DST} (delete first)")
    DST.mkdir(parents=True)
    (DST / "data" / "chunk-000").mkdir(parents=True)
    (DST / "meta").mkdir()
    for cam in CAMERAS:
        (DST / "videos" / "chunk-000" / cam).mkdir(parents=True)

    # Collect source episodes in date order
    src_dirs = sorted(SRC_ROOT.glob("*-v2"))
    print(f"merging {len(src_dirs)} v2 dirs...")

    out_eps = []
    out_stats = []
    new_idx = 0
    total_frames = 0
    for sd in src_dirs:
        info = json.load((sd / "meta/info.json").open())
        chunks_size = info.get("chunks_size", 1000)
        # episodes.jsonl
        ep_lines = [json.loads(l) for l in (sd / "meta/episodes.jsonl").open()]
        ep_by_idx = {e["episode_index"]: e for e in ep_lines}
        stats_lines = [json.loads(l) for l in (sd / "meta/episodes_stats.jsonl").open()]
        stats_by_idx = {s["episode_index"]: s for s in stats_lines}

        for old_idx in sorted(ep_by_idx.keys()):
            old_chunk = old_idx // chunks_size
            sp = sd / f"data/chunk-{old_chunk:03d}/episode_{old_idx:06d}.parquet"
            if not sp.is_file():
                print(f"  skip missing {sp}", file=sys.stderr)
                continue
            # symlink parquet
            dp = DST / f"data/chunk-000/episode_{new_idx:06d}.parquet"
            os.symlink(sp.resolve(), dp)
            # symlink each video
            for cam in CAMERAS:
                sv = sd / f"videos/chunk-{old_chunk:03d}/{cam}/episode_{old_idx:06d}.mp4"
                if not sv.is_file():
                    print(f"  skip missing video {sv}", file=sys.stderr)
                    continue
                dv = DST / f"videos/chunk-000/{cam}/episode_{new_idx:06d}.mp4"
                os.symlink(sv.resolve(), dv)
            # rewrite ep meta
            ep = ep_by_idx[old_idx].copy()
            ep["episode_index"] = new_idx
            ep["_src_dir"] = sd.name
            ep["_src_idx"] = old_idx
            out_eps.append(ep)
            total_frames += ep["length"]
            # rewrite stats
            if old_idx in stats_by_idx:
                st = stats_by_idx[old_idx].copy()
                st["episode_index"] = new_idx
                out_stats.append(st)
            new_idx += 1
    print(f"merged {new_idx} episodes total, {total_frames} frames")

    # Write meta
    with (DST / "meta/episodes.jsonl").open("w") as f:
        for ep in out_eps:
            f.write(json.dumps(ep) + "\n")
    with (DST / "meta/episodes_stats.jsonl").open("w") as f:
        for st in out_stats:
            f.write(json.dumps(st) + "\n")
    # tasks.jsonl from first dir
    shutil.copy(src_dirs[0] / "meta/tasks.jsonl", DST / "meta/tasks.jsonl")
    # info.json patched
    info = json.load((src_dirs[0] / "meta/info.json").open())
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_videos"] = new_idx * len(CAMERAS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{new_idx}"}
    (DST / "meta/info.json").write_text(json.dumps(info, indent=2))
    print(f"done → {DST}")


if __name__ == "__main__":
    main()
