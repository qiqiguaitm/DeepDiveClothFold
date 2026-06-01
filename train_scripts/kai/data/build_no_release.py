#!/usr/bin/env python3
"""Build A_0522_0526_{raw,no_release} from vis_base 2026-05-22 + 2026-05-26.

Root-cause probe Exp-1 (see docs/training/future_plans/plans/data_root_cause_probe_experiments.md):
  - `no_release`: trim the leading "cloth-release wait" still segment from every episode
    (arm stationary while operator drops the cloth on the table).
  - `raw`: SAME two days, NOT trimmed — the control, to isolate "trim effect" from "200-ep scale".

Both merge the two days into one lerobot-v2.1 dataset with episode_index re-indexed 0..199.
Only the 3 RGB cameras used in training are carried (top_head / hand_left / hand_right).
Depth (top_head_depth, zarr) is NOT carried — training does not read depth.

Trim rule (per episode):
  onset = first frame where mean |Δaction| over the 12 arm dims stays > THR for WIN frames
  cut   = max(0, onset - MARGIN)
  drop parquet rows [0:cut]; trim the 3 mp4s by cut frames; assert video_frames == parquet_rows.

Source meta quirks (vis_base):
  - episodes.jsonl uses "episode_id" (not "episode_index") and has NO "episode_index"/length-only fields.
  - NO episodes_stats.jsonl — we generate it (lerobot self_built datasets require it).

Usage:
  kai0/.venv/bin/python train_scripts/kai/data/build_no_release.py --mode no_release
  kai0/.venv/bin/python train_scripts/kai/data/build_no_release.py --mode raw
  (add --symlink-video to symlink raw-mode videos instead of copy; no_release always re-encodes)
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- constants ---- (REPO_ROOT overridable via env KAI0_REPO_ROOT for cross-cluster, e.g. cnbj)
_REPO = os.environ.get("KAI0_REPO_ROOT", "/vePFS/tim/workspace/deepdive_kai0")
VIS_BASE = Path(f"{_REPO}/kai0/data/Task_A/vis_base")
DST_ROOT = Path(f"{_REPO}/kai0/data/Task_A/self_built")
DATES = ["2026-05-22-v2", "2026-05-26-v2"]
CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
CAM_DIRS = {"observation.images.top_head": "top_head",
            "observation.images.hand_left": "hand_left",
            "observation.images.hand_right": "hand_right"}
FPS = 30
ARM_DIMS = list(range(0, 6)) + list(range(7, 13))   # 12 arm dims (exclude dim6 L_grip, dim13 R_grip)
THR = 3e-3      # rad/frame: sustained mean |Δa| over arm dims => "moving"
WIN = 10        # frames of sustained motion to call it the onset
MARGIN = 15     # keep this many frames before onset (avoid clipping the reach-start)


def motion_onset(action: np.ndarray) -> int:
    """First frame index where arm motion sustains above THR for WIN frames."""
    da = np.abs(np.diff(action[:, ARM_DIMS], axis=0)).mean(axis=1)  # (T-1,)
    run = 0
    for i, moving in enumerate(da > THR):
        run = run + 1 if moving else 0
        if run >= WIN:
            return i - WIN + 1
    return len(action)  # never moved (anomaly)


def trim_video_pyav(src_mp4: Path, dst_mp4: Path, cut: int, expected_frames: int):
    """Re-encode src_mp4 dropping the first `cut` frames. Assert output == expected_frames."""
    import av
    in_c = av.open(str(src_mp4))
    in_stream = in_c.streams.video[0]
    in_stream.thread_type = "AUTO"  # multithreaded decode
    out_c = av.open(str(dst_mp4), mode="w")
    out_stream = out_c.add_stream("libx264", rate=FPS)
    out_stream.width = in_stream.codec_context.width
    out_stream.height = in_stream.codec_context.height
    out_stream.pix_fmt = "yuv420p"
    # veryfast preset + crf18: near-visually-lossless, ~5-8x faster than default 'medium'.
    # threads=4 per encoder; episodes run in parallel so keep per-proc thread count modest.
    out_stream.options = {"crf": "18", "preset": "veryfast", "threads": "4"}

    written = 0
    idx = 0
    for frame in in_c.decode(video=0):
        if idx >= cut:
            new = frame.reformat(format="yuv420p")
            for pkt in out_stream.encode(new):
                out_c.mux(pkt)
            written += 1
        idx += 1
    for pkt in out_stream.encode():  # flush
        out_c.mux(pkt)
    in_c.close()
    out_c.close()
    if written != expected_frames:
        raise RuntimeError(
            f"video frame mismatch {src_mp4.name}: wrote {written}, parquet rows {expected_frames} "
            f"(decoded {idx} total, cut {cut})")


def _trim_job(job):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    src_mp4, dst_mp4, cut, new_len = job
    trim_video_pyav(Path(src_mp4), Path(dst_mp4), cut, new_len)
    return dst_mp4


def count_video_frames(mp4: Path) -> int:
    import av
    c = av.open(str(mp4))
    n = sum(1 for _ in c.decode(video=0))
    c.close()
    return n


def per_episode_stats(df: pd.DataFrame) -> dict:
    """Build lerobot episodes_stats 'stats' dict (scalar features only; images omitted)."""
    stats = {}
    for col in df.columns:
        vals = df[col].to_numpy()
        if vals.dtype == object:  # array-valued cell (action / state)
            arr = np.stack(vals).astype(np.float64)
        else:
            arr = vals.astype(np.float64).reshape(len(vals), -1)
        stats[col] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "count": [len(arr)],
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "no_release"], required=True)
    ap.add_argument("--symlink-video", action="store_true",
                    help="raw mode only: symlink videos instead of copy (saves disk)")
    ap.add_argument("--dry-run", action="store_true", help="compute cuts + report, write nothing")
    args = ap.parse_args()

    trim = (args.mode == "no_release")
    dst = DST_ROOT / ("A_0522_0526_no_release" if trim else "A_0522_0526_raw")

    if not args.dry_run:
        if dst.exists():
            sys.exit(f"dst already exists: {dst} (delete first)")
        (dst / "data" / "chunk-000").mkdir(parents=True)
        (dst / "meta").mkdir()
        for cam in CAMERAS:
            (dst / "videos" / "chunk-000" / CAM_DIRS[cam]).mkdir(parents=True)

    episodes_out, stats_out = [], []
    new_idx = 0
    total_frames = 0
    cut_report = []
    video_jobs = []  # (src_mp4, dst_mp4, cut, new_len) for parallel trim

    for date in DATES:
        src = VIS_BASE / date
        src_eps = {json.loads(l)["episode_id"]: json.loads(l)
                   for l in (src / "meta" / "episodes.jsonl").open()}
        parquets = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
        print(f"[{date}] {len(parquets)} episodes")

        for pq in parquets:
            old_id = int(pq.stem.split("_")[1])
            df = pd.read_parquet(pq)
            T0 = len(df)

            if trim:
                action = np.stack(df["action"].to_numpy()).astype(np.float64)
                onset = motion_onset(action)
                cut = max(0, onset - MARGIN)
            else:
                cut = 0
            cut_report.append(cut)
            new_len = T0 - cut

            if not args.dry_run:
                # --- parquet: drop head, re-index frame_index / index / timestamp / episode_index ---
                sub = df.iloc[cut:].copy().reset_index(drop=True)
                sub["frame_index"] = np.arange(new_len, dtype=np.int64)
                sub["episode_index"] = np.int64(new_idx)
                sub["index"] = np.arange(total_frames, total_frames + new_len, dtype=np.int64)
                sub["timestamp"] = (np.arange(new_len, dtype=np.float32) / FPS).astype(np.float32)
                out_pq = dst / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
                sub.to_parquet(out_pq, index=False)

                # --- videos: 3 RGB cams ---
                for cam in CAMERAS:
                    sv = src / "videos" / "chunk-000" / CAM_DIRS[cam] / f"episode_{old_id:06d}.mp4"
                    dv = dst / "videos" / "chunk-000" / CAM_DIRS[cam] / f"episode_{new_idx:06d}.mp4"
                    if trim:
                        video_jobs.append((str(sv), str(dv), cut, new_len))  # deferred to pool
                    elif args.symlink_video:
                        os.symlink(sv.resolve(), dv)
                    else:
                        shutil.copy(sv, dv)

                # --- meta rows ---
                src_meta = src_eps[old_id]
                ep_row = {
                    "episode_index": new_idx,
                    "tasks": [src_meta.get("prompt", "Flatten and fold the cloth.")],
                    "length": new_len,
                }
                episodes_out.append(ep_row)
                stats_out.append({"episode_index": new_idx, "stats": per_episode_stats(sub)})

            total_frames += new_len
            new_idx += 1

    # --- parallel video trim (the slow part: 600 mp4 re-encodes) ---
    if trim and not args.dry_run and video_jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        nproc = min(14, max(1, (os.cpu_count() or 8) // 4))  # 4 enc-threads each
        print(f"trimming {len(video_jobs)} videos with {nproc} workers (4 threads each)...")
        done = 0
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = {ex.submit(_trim_job, j): j for j in video_jobs}
            for fut in as_completed(futs):
                fut.result()  # raises on frame-mismatch assert
                done += 1
                if done % 60 == 0:
                    print(f"  {done}/{len(video_jobs)} videos done")
        print(f"  all {len(video_jobs)} videos trimmed + frame-count verified")

    # --- report ---
    cr = np.array(cut_report)
    print(f"\nmode={args.mode}  episodes={new_idx}  total_frames={total_frames}")
    if trim:
        print(f"  cut frames: median={np.median(cr):.0f}  mean={cr.mean():.1f}  "
              f"p90={np.percentile(cr,90):.0f}  max={cr.max():.0f}  min={cr.min():.0f}")
        print(f"  dropped {cr.sum()} frames total ({100*cr.sum()/(cr.sum()+total_frames):.1f}%)")

    if args.dry_run:
        print("DRY RUN — nothing written.")
        return

    # --- write meta ---
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for r in episodes_out:
            f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for r in stats_out:
            f.write(json.dumps(r) + "\n")
    shutil.copy(VIS_BASE / DATES[0] / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")

    info = json.loads((VIS_BASE / DATES[0] / "meta" / "info.json").read_text())
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_videos"] = new_idx * len(CAMERAS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{new_idx}"}
    # drop depth feature + depth_path (not carried)
    info["features"].pop("observation.depth.top_head", None)
    info.pop("depth_path", None)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"done → {dst}")
    print("  next: register config + run kai0/scripts/compute_norm_stats.py to (re)compute norm_stats")


if __name__ == "__main__":
    main()
