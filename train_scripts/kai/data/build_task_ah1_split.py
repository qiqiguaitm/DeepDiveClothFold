#!/usr/bin/env python3
"""Build Task_AH1_170 (train) + Task_AH1_val (held-out val) for the Horizontal-Fold-v1
首次基线 (plan: docs/.../plans/pi05_fold_sop_paradigm_baselines.md §2.B).

决策 (§2.B.3, 2026-06-16 用户定档):
  - 数据 = Task_AH1 单日 200 ep (2026-06-15-v3); 无额外 ep 可留 val → 从 200 内切.
  - split = 按 episode_index 前 170 train + 末 30 held-out val (in-distribution).
  - 夹爪 = 原始 action (不裁).
  - prompt = "Flatten and fold the cloth. Horizontal Fold v1."
    (规范化: 数据原值 "Horizontally Fold v1." → "Horizontal" 以与 Vertical Fold v1 平行;
     train==deploy 一字不差, 覆盖数据原值.)

机制 (与 build_task_av1_200_split.py 一致): 单 chunk-000, 连续重排 episode_index, 视频 symlink 到
源 realpath (不重编码), drop depth 特征 (AH1 已无 depth, pop 容错), 写 episodes_stats + info + tasks,
最后 norm_stats_from_dataset 算 norm_stats.json (action_dim=32, pi05).
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
SRC_BASE = ROOT / "Task_AH1" / "base" / "v3"
DATE_ORDER = ["2026-06-15-v3"]
CAMERAS = ("top_head", "hand_left", "hand_right")
FPS = 30
PROMPT = "Flatten and fold the cloth. Horizontal Fold v1."
N_TRAIN = 170
CHUNK = 0


def list_src_eps():
    """Return episode_index-ordered global list of (date, src_ep_idx, src_dir, success)."""
    items = []
    for d in DATE_ORDER:
        sd = SRC_BASE / d
        eps = [json.loads(l) for l in (sd / "meta" / "episodes.jsonl").open()]
        # source uses "episode_id" (== parquet index 0..n-1); keep file order
        for e in sorted(eps, key=lambda x: int(x["episode_id"])):
            items.append((d, int(e["episode_id"]), sd, bool(e.get("success", True))))
    return items


def write_split(name: str, picks, info_template: dict):
    dst = ROOT / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    eps_meta, stats_out = [], []
    total_frames = 0
    for new_ep, (d, src_ep, sd, _succ) in enumerate(picks):
        src_pq = sd / "data" / f"chunk-{CHUNK:03d}" / f"episode_{src_ep:06d}.parquet"
        df = pd.read_parquet(src_pq)
        n = len(df)
        df["episode_index"] = np.int64(new_ep)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS)
        if "task_index" in df.columns:
            df["task_index"] = np.int64(0)
        dst_pq = dst / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dst_pq)

        for cam in CAMERAS:
            # source dirs are named "<cam>" (raw layout); output uses lerobot key "observation.images.<cam>"
            sv = sd / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{src_ep:06d}.mp4"
            if not (sv.exists() or sv.is_symlink()):
                raise FileNotFoundError(f"missing video {sv}")
            dv = dst / "videos" / f"chunk-{CHUNK:03d}" / f"observation.images.{cam}" / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(sv.resolve()), dv)

        eps_meta.append({"episode_index": new_ep, "tasks": [PROMPT], "length": n,
                         "src_date": d, "src_ep": src_ep})
        stats_out.append({"episode_index": new_ep, "stats": per_episode_stats(df)})
        total_frames += n

    # info.json: clone source, drop depth feature (if any), fix counts/chunks
    info = json.loads(json.dumps(info_template))
    info["features"].pop("observation.depth.top_head", None)
    info["total_episodes"] = len(picks)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = len(picks) * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, len(picks))
    info["splits"] = {"train": f"0:{len(picks)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    (dst / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    print(f"  [{name}] {len(picks)} ep / {total_frames} frames -> {dst}", flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    a = ap.parse_args()

    items = list_src_eps()
    counts = {d: sum(1 for x in items if x[0] == d) for d in DATE_ORDER}
    print(f"sources (episode_index order): {counts}  total={len(items)}", flush=True)
    if len(items) < N_TRAIN + 1:
        sys.exit(f"FATAL: only {len(items)} ep, need > {N_TRAIN}")

    train_picks = items[:N_TRAIN]
    val_picks = items[N_TRAIN:]
    tf = sum(1 for x in train_picks if not x[3])  # success==False count
    vf = sum(1 for x in val_picks if not x[3])
    print(f"train={len(train_picks)}  (success=False: {tf})", flush=True)
    print(f"val=  {len(val_picks)}  (success=False: {vf})", flush=True)
    if a.dry_run:
        print("dry-run: nothing written"); return

    info_template = json.loads((SRC_BASE / DATE_ORDER[0] / "meta" / "info.json").read_text())
    train_dst = write_split("Task_AH1_170", train_picks, info_template)
    val_dst = write_split("Task_AH1_val", val_picks, info_template)

    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        for dst in (train_dst, val_dst):
            print(f"  [norm_stats] computing for {dst.name} (action_dim=32)...", flush=True)
            compute_norm_stats(str(dst), action_dim=32)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
