#!/usr/bin/env python3
"""Build val_kai0_official from kai0_base + kai0_dagger holdouts.

Takes N_base ep from kai0_base + N_dag ep from kai0_dagger, evenly spaced.

Usage:
  python build_val_kai0_official.py [--n-base 15 --n-dag 15]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
DATA_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A")


def chunk_for(idx: int, chunks_size: int = 1000) -> int:
    return idx // chunks_size


def evenly_spaced(total: int, k: int) -> list[int]:
    """Pick k indices evenly spaced from [0, total)."""
    if k >= total:
        return list(range(total))
    return [round(i * (total - 1) / (k - 1)) for i in range(k)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-base", type=int, default=15)
    p.add_argument("--n-dag", type=int, default=15)
    p.add_argument("--dst", default=str(DATA_ROOT / "self_built" / "val_kai0_official"))
    args = p.parse_args()

    dst = Path(args.dst)
    if dst.exists():
        sys.exit(f"dst exists: {dst}")
    dst.mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir()
    for cam in CAMERAS:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    plan: list[tuple[Path, int]] = []  # (src_dataset_root, src_ep_idx)
    for src_name, n in (("kai0_base", args.n_base), ("kai0_dagger", args.n_dag)):
        src = DATA_ROOT / src_name
        info = json.loads((src / "meta" / "info.json").read_text())
        total = info["total_episodes"]
        chunks_size = info.get("chunks_size", 1000)
        picks = evenly_spaced(total, n)
        for idx in picks:
            plan.append((src, idx, chunks_size, src_name))
    print(f"picked {len(plan)} eps: {sum(1 for x in plan if x[3]=='kai0_base')} base + "
          f"{sum(1 for x in plan if x[3]=='kai0_dagger')} dagger")

    # Build meta lookup once per src
    src_meta_cache: dict[str, dict[int, dict]] = {}
    for src_name in ("kai0_base", "kai0_dagger"):
        src = DATA_ROOT / src_name
        eps = {}
        for line in (src / "meta" / "episodes.jsonl").open():
            d = json.loads(line)
            eps[d["episode_index"]] = d
        src_meta_cache[src_name] = eps

    src_eps_stats: dict[str, dict[int, dict]] = {}
    for src_name in ("kai0_base", "kai0_dagger"):
        src = DATA_ROOT / src_name
        stats = {}
        ep_stats_path = src / "meta" / "episodes_stats.jsonl"
        if ep_stats_path.exists():
            for line in ep_stats_path.open():
                d = json.loads(line)
                stats[d["episode_index"]] = d
        src_eps_stats[src_name] = stats

    # ---- copy parquet + videos ----
    total_frames = 0
    out_eps = []
    out_stats = []
    for new_idx, (src, src_idx, chunks_size, src_name) in enumerate(plan):
        src_chunk = chunk_for(src_idx, chunks_size)
        sname = f"episode_{src_idx:06d}.parquet"
        sp = src / "data" / f"chunk-{src_chunk:03d}" / sname
        dp = dst / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
        shutil.copy(sp, dp)
        # videos
        for cam in CAMERAS:
            sv = src / "videos" / f"chunk-{src_chunk:03d}" / cam / f"episode_{src_idx:06d}.mp4"
            dv = dst / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
            shutil.copy(sv, dv)
        # patched meta
        ep = src_meta_cache[src_name][src_idx].copy()
        ep["episode_index"] = new_idx
        ep["_orig_source"] = src_name
        ep["_orig_episode_index"] = src_idx
        out_eps.append(ep)
        total_frames += ep["length"]
        if src_idx in src_eps_stats[src_name]:
            st = src_eps_stats[src_name][src_idx].copy()
            st["episode_index"] = new_idx
            out_stats.append(st)

    # ---- meta files ----
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for ep in out_eps:
            f.write(json.dumps(ep) + "\n")
    if out_stats:
        with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
            for st in out_stats:
                f.write(json.dumps(st) + "\n")

    # tasks.jsonl from kai0_base (assume same task)
    shutil.copy(DATA_ROOT / "kai0_base" / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")

    # info.json patched
    info = json.loads((DATA_ROOT / "kai0_base" / "meta" / "info.json").read_text())
    info["total_episodes"] = len(out_eps)
    info["total_frames"] = total_frames
    info["total_videos"] = len(out_eps) * len(CAMERAS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(out_eps)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # norm_stats: copy from kai0_base
    if (DATA_ROOT / "kai0_base" / "norm_stats.json").exists():
        shutil.copy(DATA_ROOT / "kai0_base" / "norm_stats.json", dst / "norm_stats.json")

    print(f"done → {dst}")
    print(f"  episodes={len(out_eps)}  frames={total_frames}")


if __name__ == "__main__":
    main()
