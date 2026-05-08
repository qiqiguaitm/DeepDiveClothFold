"""Sample N random episodes from a LeRobot v2.1 split (base/dagger) for inspection.

Copies parquet + mp4 (3 cameras) as-is, filters meta/{episodes,episodes_stats}.jsonl,
and extracts a few evenly-spaced representative frames per video.

Usage:
    python scripts/sample_examples.py \
        --src /home/tim/workspace/deepdive_kai0/kai0/data/Task_A/base \
        --dst /home/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/base_sample_200 \
        --n 200 --seed 42 --frames-per-video 9
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2


def chunk_dir(idx: int, chunks_size: int) -> str:
    return f"chunk-{idx // chunks_size:03d}"


def extract_frames(mp4: Path, out_dir: Path, n: int) -> int:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if n <= 1:
        idxs = [0]
    else:
        idxs = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
    saved = 0
    for slot, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(
            str(out_dir / f"frame_{slot:02d}_f{fi:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        saved += 1
    cap.release()
    return saved


def filter_jsonl(src_path: Path, dst_path: Path, keep: set[int]) -> int:
    kept = 0
    with src_path.open() as f, dst_path.open("w") as g:
        for line in f:
            rec = json.loads(line)
            if rec.get("episode_index") in keep:
                g.write(line)
                kept += 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames-per-video", type=int, default=9)
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    info = json.loads((src / "meta" / "info.json").read_text())
    total = info["total_episodes"]
    chunks_size = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"[{src.name}] total_episodes={total} chunks_size={chunks_size} cameras={video_keys}")

    rng = random.Random(args.seed)
    n = min(args.n, total)
    sampled = sorted(rng.sample(range(total), n))
    sampled_set = set(sampled)

    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "videos").mkdir(parents=True, exist_ok=True)
    (dst / "frames").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    shutil.copy2(src / "meta" / "info.json", dst / "meta" / "info.json")
    shutil.copy2(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    for meta_file in ("episodes.jsonl", "episodes_stats.jsonl"):
        if (src / "meta" / meta_file).exists():
            kept = filter_jsonl(src / "meta" / meta_file, dst / "meta" / meta_file, sampled_set)
            print(f"  [meta] {meta_file}: kept {kept}")

    if (src / "norm_stats.json").exists():
        shutil.copy2(src / "norm_stats.json", dst / "norm_stats.json")

    (dst / "sampled_episodes.json").write_text(
        json.dumps(
            {"seed": args.seed, "n": n, "total_in_src": total, "episode_indices": sampled},
            indent=2,
        )
    )

    missing = []
    for i, ep in enumerate(sampled):
        chunk = chunk_dir(ep, chunks_size)
        pq_src = src / "data" / chunk / f"episode_{ep:06d}.parquet"
        pq_dst = dst / "data" / chunk / f"episode_{ep:06d}.parquet"
        pq_dst.parent.mkdir(parents=True, exist_ok=True)
        if pq_src.exists():
            shutil.copy2(pq_src, pq_dst)
        else:
            missing.append(str(pq_src))

        for cam in video_keys:
            mp4_src = src / "videos" / chunk / cam / f"episode_{ep:06d}.mp4"
            mp4_dst = dst / "videos" / chunk / cam / f"episode_{ep:06d}.mp4"
            mp4_dst.parent.mkdir(parents=True, exist_ok=True)
            if not mp4_src.exists():
                missing.append(str(mp4_src))
                continue
            shutil.copy2(mp4_src, mp4_dst)
            extract_frames(mp4_src, dst / "frames" / f"episode_{ep:06d}" / cam, args.frames_per_video)

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  progress: {i + 1}/{n}")

    if missing:
        (dst / "missing_files.txt").write_text("\n".join(missing))
        print(f"  [warn] {len(missing)} missing files -> missing_files.txt")

    print(f"done -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
