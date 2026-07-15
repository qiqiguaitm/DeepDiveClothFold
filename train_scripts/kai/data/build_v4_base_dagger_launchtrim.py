#!/usr/bin/env python3
"""A_v4_base_dagger_launchtrim = base(vis_base/v4, 不裁) + 裁后 dagger(v4_launchtrim, 起爆点前裁).
launchpoint-trim plan §2.2 主实验(任务②-analog): base + 裁后(老 dagger 05-29~06-23 + fresh 06-29~07-03).
单变量 vs 任务② plus_freshdagger = 仅 dagger clip 双向起爆点前裁; 其余(base组成/gripper-snap/列/norm)逐字段同 merged.

- base: vis_base/v4 13 日期原样(symlink 原视频, 不裁), 同 build_v4_awbc_merged.
- dagger: v4_launchtrim/<date>(裁后 parquet + 已 PTS 归零重编码视频) glob episode_*, symlink 裁后视频.
  日期 = 老 dagger 12 + fresh 5(06-29~07-03); 排除 06-24/25/07-06/07(不在任务②, 保单变量).
- gripper force-snap action<8mm→0(与 merged 默认一致); 只留标准 7 列(防 schema CastError).
- 重排 episode_index; 重算 norm_stats(action_dim=32); 建 meta + episodes_stats.
之后: AE 打标(adv_est_v1) → discretize top-30% → 训练.
Run: kai0/.venv/bin/python train_scripts/kai/data/build_v4_base_dagger_launchtrim.py [--dry-run] [--no-norm]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import per_episode_stats  # noqa: E402
from build_v4_awbc_merged import snap_gripper_force, BASE_DATES, KEEP_COLS, CAMERAS, PROMPT, FPS, CHUNK  # noqa: E402

ROOT = Path("/home/tim/workspace/deepdive_kai0/kai0/data")
BASE_V4 = ROOT / "Task_A" / "vis_base" / "v4"
TRIM = ROOT / "Task_A" / "vis_dagger" / "v4_launchtrim"
OUT = ROOT / "Task_A" / "self_built" / "A_v4_base_dagger_launchtrim"
# 老 dagger(= A_v4_base_dagger dagger group, 806ep) + fresh(= 任务② fresh, 06-29~07-03, 506→505 裁后)
OLD_DAGGER = ["2026-05-29-v4","2026-06-01-v4","2026-06-02-v4","2026-06-03-v4","2026-06-04-v4","2026-06-05-v4","2026-06-08-v4","2026-06-09-v4","2026-06-10-v4","2026-06-16-v4","2026-06-17-v4","2026-06-23-v4"]
FRESH_DAGGER = ["2026-06-29-v4","2026-06-30-v4","2026-07-01-v4","2026-07-02-v4","2026-07-03-v4"]
DAGGER_DATES = OLD_DAGGER + FRESH_DAGGER


def list_eps():
    """(src_dir, src_ep, group, trimmed?) base(vis_base 不裁) 后 dagger(v4_launchtrim 裁后)."""
    items = []
    for d in BASE_DATES:
        sd = BASE_V4 / d
        for l in (sd / "meta" / "episodes.jsonl").open():
            l = l.strip()
            if not l:
                continue
            e = json.loads(l)
            ep = e.get("episode_index", e.get("episode_id"))
            if ep is None:
                continue
            ep = int(ep)
            if not (sd / "data" / f"chunk-{CHUNK:03d}" / f"episode_{ep:06d}.parquet").exists():
                print(f"  skip stale {sd.name} ep{ep}", flush=True); continue
            items.append((sd, ep, "base", False))
    for d in DAGGER_DATES:
        sd = TRIM / d
        if not sd.exists():
            raise FileNotFoundError(f"缺裁后目录 {sd} — 先跑 launchpoint_trim_dagger.py")
        parts = sorted((sd / "data" / f"chunk-{CHUNK:03d}").glob("episode_*.parquet"),
                       key=lambda p: int(p.stem.split("_")[1]))
        for p in parts:
            items.append((sd, int(p.stem.split("_")[1]), "dagger", True))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--gripper-snap-thresh", type=float, default=0.008)
    a = ap.parse_args()

    items = list_eps()
    nb = sum(1 for x in items if x[2] == "base"); nd = sum(1 for x in items if x[2] == "dagger")
    print(f"sources: base={nb}ep({len(BASE_DATES)}日期) + 裁后dagger={nd}ep({len(DAGGER_DATES)}日期) = {len(items)}ep", flush=True)
    if a.dry_run:
        print("dry-run"); return

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
    (OUT / "meta").mkdir()

    eps_meta, stats_out, total_frames, grip_changed = [], [], 0, 0
    for new_ep, (sd, src_ep, grp, trimmed) in enumerate(items):
        df = pd.read_parquet(sd / "data" / f"chunk-{CHUNK:03d}" / f"episode_{src_ep:06d}.parquet")
        df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
        grip_changed += snap_gripper_force(df, a.gripper_snap_thresh)
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
            if not (sv.exists() or sv.is_symlink()):
                sv = sd / "videos" / f"chunk-{CHUNK:03d}" / cam.replace("observation.images.", "") / f"episode_{src_ep:06d}.mp4"
            if not (sv.exists() or sv.is_symlink()):
                raise FileNotFoundError(f"missing video {sv}")
            dv = OUT / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(sv.resolve()), dv)
        eps_meta.append({"episode_index": new_ep, "tasks": [PROMPT], "length": n,
                         "src": sd.name, "src_ep": src_ep, "group": grp, "trimmed": trimmed})
        stats_out.append({"episode_index": new_ep, "stats": per_episode_stats(df)})
        total_frames += n

    info = json.loads((BASE_V4 / BASE_DATES[0] / "meta" / "info.json").read_text())
    info.update({"total_episodes": len(items), "total_frames": total_frames, "total_tasks": 1,
                 "total_videos": len(items) * len(CAMERAS), "total_chunks": 1,
                 "chunks_size": max(1000, len(items)), "splits": {"train": f"0:{len(items)}"}})
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
    gt = total_frames * 2
    print(f"  merged {len(items)}ep / {total_frames}帧 -> {OUT}; gripper-snap {grip_changed}/{gt}({100*grip_changed/max(gt,1):.1f}%)", flush=True)

    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
