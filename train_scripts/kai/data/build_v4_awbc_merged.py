#!/usr/bin/env python3
"""Merge vis_base/v4 (13 dates) + vis_dagger/v4 (12 dates) → A_v4_base_dagger. action≠state (gripper-from-master). orig: vis_base/v3 (≤2026-05-18, 12 dates) + vis_dagger/v3 (8 dates) → one lerobot dataset
`self_built/vis_awbc_merged` for the vis-native AWBC pipeline (stage-label → AE → value → AWBC).

- Front-trimmed v3 sources (depth already dropped). Contiguous renumber episode_index 0..N-1.
- Videos symlinked to source realpath (no re-encode). Single chunk-000, chunks_size=max(1000,N).
- Drops dagger-only `intervention` column so ALL episodes share one schema (state/action/3cam/index…).
- task_index=0 + single prompt "Flatten and fold the cloth." (stage_progress_gt / advantage / pos-neg
  task_index get added by the LATER pipeline stages: infer_dagger → eval.py → discretize_advantage).
- Computes norm_stats.json (action_dim=32, pi05).

Usage: kai0/.venv/bin/python train_scripts/kai/data/build_vis_awbc_merged.py [--dry-run] [--no-norm]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import per_episode_stats  # noqa: E402

ROOT = Path("/home/tim/workspace/deepdive_kai0/kai0/data")
BASE_V4 = ROOT / "Task_A" / "vis_base" / "v4"
DAGGER_V4 = ROOT / "Task_A" / "vis_dagger" / "v4"
OUT = ROOT / "Task_A" / "self_built" / "A_v4_base_dagger"
CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
FPS = 30
PROMPT = "Flatten and fold the cloth."
CHUNK = 0
# base v3 dates ≤ 2026-05-18 (12), then dagger v3 dates (8) — date order within each group
BASE_DATES = ["2026-04-23-v4","2026-04-24-v4","2026-04-25-v4","2026-04-28-v4","2026-04-29-v4","2026-04-30-v4","2026-05-06-v4","2026-05-07-v4","2026-05-08-v4","2026-05-09-v4","2026-05-10-v4","2026-05-18-v4","2026-06-04-v4"]
DAGGER_DATES = ["2026-05-29-v4","2026-06-01-v4","2026-06-02-v4","2026-06-03-v4","2026-06-04-v4","2026-06-05-v4","2026-06-08-v4","2026-06-09-v4","2026-06-10-v4","2026-06-16-v4","2026-06-17-v4","2026-06-23-v4"]
KEEP_COLS = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]


def list_eps():
    """date-ordered (src_dir, src_ep_id, group) over base then dagger."""
    items = []
    for grp, root, dates in [("base", BASE_V4, BASE_DATES), ("dagger", DAGGER_V4, DAGGER_DATES)]:
        for d in dates:
            sd = root / d
            if not (sd / "meta" / "episodes.jsonl").exists():
                raise FileNotFoundError(f"missing source {sd}")
            for l in (sd / "meta" / "episodes.jsonl").open():
                l = l.strip()
                if not l:
                    continue
                e = json.loads(l)
                ep = e.get("episode_index", e.get("episode_id"))  # 双 schema: 新在线录制器=episode_id
                if ep is None:
                    continue
                ep = int(ep)
                # stale-manifest skip: parquet 已清但 manifest 残留 → 跳过 (v4 base 有 000000/002 空缺)
                if not (sd / "data" / f"chunk-{CHUNK:03d}" / f"episode_{ep:06d}.parquet").exists():
                    print(f"  skip stale {sd.name} ep{ep} (no parquet)", flush=True)
                    continue
                items.append((sd, ep, grp))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    a = ap.parse_args()

    items = list_eps()
    nb = sum(1 for x in items if x[2] == "base"); nd = sum(1 for x in items if x[2] == "dagger")
    print(f"sources: base={nb} ep ({len(BASE_DATES)} dates) + dagger={nd} ep ({len(DAGGER_DATES)} dates) = {len(items)} ep", flush=True)
    if a.dry_run:
        print("dry-run: nothing written"); return

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
    (OUT / "meta").mkdir()

    eps_meta, stats_out = [], []
    total_frames = 0
    for new_ep, (sd, src_ep, grp) in enumerate(items):
        df = pd.read_parquet(sd / "data" / f"chunk-{CHUNK:03d}" / f"episode_{src_ep:06d}.parquet")
        df = df[[c for c in KEEP_COLS if c in df.columns]].copy()  # drop intervention/extra → uniform schema
        n = len(df)
        df["episode_index"] = np.int64(new_ep)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS)
        df["task_index"] = np.int64(0)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       OUT / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet")
        for cam in CAMERAS:
            sv = sd / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{src_ep:06d}.mp4"
            if not (sv.exists() or sv.is_symlink()):  # 新在线录制器用裸相机名 (top_head 而非 observation.images.top_head)
                sv = sd / "videos" / f"chunk-{CHUNK:03d}" / cam.replace("observation.images.", "") / f"episode_{src_ep:06d}.mp4"
            if not (sv.exists() or sv.is_symlink()):
                raise FileNotFoundError(f"missing video {sv}")
            dv = OUT / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(sv.resolve()), dv)
        eps_meta.append({"episode_index": new_ep, "tasks": [PROMPT], "length": n,
                         "src": sd.name, "src_ep": src_ep, "group": grp})
        stats_out.append({"episode_index": new_ep, "stats": per_episode_stats(df)})
        total_frames += n

    # info.json: clone a base v3 source, fix counts
    info = json.loads((BASE_V4 / BASE_DATES[0] / "meta" / "info.json").read_text())
    info["total_episodes"] = len(items)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = len(items) * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, len(items))
    info["splits"] = {"train": f"0:{len(items)}"}
    info["features"].pop("observation.depth.top_head", None)
    info["features"].pop("intervention", None)
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (OUT / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (OUT / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    (OUT / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    print(f"  merged {len(items)} ep / {total_frames} frames -> {OUT}", flush=True)

    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
