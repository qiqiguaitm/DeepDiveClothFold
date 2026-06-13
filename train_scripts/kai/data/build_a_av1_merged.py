"""Build a_av1_merged: Task_A (横向折, domain 0) + Task_AV1 (竖向折 Vertical Fold v1, domain 1)
into ONE physical LeRobot dataset for the 1:1 frame-weighted co-train (plan:
docs/.../plans/pi05_task_a_av1_mixed_1to1_plan.md). Mechanism mirrors build_kai_vis_merged.py:
domain carried per-frame via task_index (0/1); 1:1 balancing done at train time by the
domain-weighted JAX sampler (NOT by copying here); videos symlinked to realpath.

Domains:
  - domain 0: A_smooth800_dagger_full (vis lerobot format, video dirs = observation.images.<cam>)
  - domain 1: Task_AV1/base/{2026-06-11-v2[:133], 2026-06-12-v2[:171]} (raw format, video dirs = <cam>)
              snapshot-capped to the plan's frozen 304ep (§7-1).

per-domain prompt (§7-2): domain0 "Flatten and fold the cloth." / domain1 "...Vertical Fold v1."
Output: kai0/data/Task_A/self_built/a_av1_merged. Run with kai0 venv; KAI0_REPO_ROOT overrides root (cnbj).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_no_release import per_episode_stats, CAMERAS, FPS  # CAMERAS = observation.images.<cam>

_REPO = os.environ.get("KAI0_REPO_ROOT", "/vePFS/tim/workspace/deepdive_kai0")
TA = Path(f"{_REPO}/kai0/data/Task_A")
AV1 = Path(f"{_REPO}/kai0/data/Task_AV1/base")
PROMPT0 = "Flatten and fold the cloth."
PROMPT1 = "Flatten and fold the cloth. Vertical Fold v1."
CAM_RAW = {c: c.split(".")[-1] for c in CAMERAS}  # observation.images.top_head -> top_head
KEEP = ["observation.state", "action"]
# domain1 AV1 frozen snapshot (plan §1/§7-1): date -> max ep count (by episode_id order)
AV1_DATES = [("2026-06-11-v2", 133), ("2026-06-12-v2", 171)]


def _resolve_video(sv: Path):
    if sv.is_symlink() or sv.exists():
        rp = Path(os.path.realpath(sv)); return rp if rp.exists() else None
    return None


def _domain0_eps():
    """A_smooth800_dagger_full: (parquet, old_idx, src_dir, cam_layout='lerobot')."""
    src = TA / "self_built" / "A_smooth800_dagger_full"
    cs = json.loads((src / "meta" / "info.json").read_text()).get("chunks_size", 1000)
    for pq in sorted((src / "data").glob("chunk-*/episode_*.parquet")):
        old = int(pq.stem.split("_")[1])
        yield pq, old, src, old // cs, "lerobot"


def _domain1_eps():
    """Task_AV1 dates, capped to snapshot: (parquet, old_idx, src_dir, chunk0, cam_layout='raw')."""
    for date, cap in AV1_DATES:
        sd = AV1 / date
        pqs = sorted((sd / "data").glob("chunk-000/episode_*.parquet"))[:cap]
        for pq in pqs:
            old = int(pq.stem.split("_")[1])
            yield pq, old, sd, 0, "raw"


def build(out_name, dry_run, symlink_video=True):
    dst = TA / "self_built" / out_name
    all_eps = ([(pq, old, sd, sc, lay, 0) for (pq, old, sd, sc, lay) in _domain0_eps()] +
               [(pq, old, sd, sc, lay, 1) for (pq, old, sd, sc, lay) in _domain1_eps()])
    n0 = sum(1 for e in all_eps if e[5] == 0); n1 = sum(1 for e in all_eps if e[5] == 1)
    print(f"[merge] domain0(Task_A)={n0}  domain1(AV1)={n1}  total={len(all_eps)} -> {dst}", flush=True)
    if dry_run:
        print("DRY — nothing written."); return
    if dst.exists():
        sys.exit(f"dst exists: {dst} (delete first)")
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir()
    for cam in CAMERAS:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    episodes_out, stats_out = [], []
    total_frames = 0; new_idx = 0; dropped = 0; f0 = 0; f1 = 0
    for (pq, old, sd, sc, lay, dom) in all_eps:
        svs = []
        for cam in CAMERAS:
            cdir = cam if lay == "lerobot" else CAM_RAW[cam]
            svs.append((cam, _resolve_video(sd / "videos" / f"chunk-{sc:03d}" / cdir / f"episode_{old:06d}.mp4")))
        if any(sv is None for _, sv in svs):
            dropped += 1; print(f"  SKIP {sd.name} old={old}: missing video", flush=True); continue
        df = pd.read_parquet(pq, columns=[c for c in KEEP])
        n = len(df)
        for col in ("observation.state", "action"):
            df[col] = [np.asarray(v, dtype=np.float32).reshape(-1)[:14] for v in df[col].to_numpy()]
        df = df.reset_index(drop=True)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["episode_index"] = np.int64(new_idx)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS).astype(np.float32)
        df["task_index"] = np.int64(dom)
        df = df[["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]]
        df.to_parquet(dst / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet", index=False)
        for cam, sv in svs:
            os.symlink(str(sv), dst / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4") if symlink_video \
                else __import__("shutil").copy(sv, dst / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4")
        episodes_out.append({"episode_index": new_idx, "tasks": [PROMPT0 if dom == 0 else PROMPT1], "length": n})
        stats_out.append({"episode_index": new_idx, "stats": per_episode_stats(df)})
        total_frames += n
        if dom == 0: f0 += n
        else: f1 += n
        new_idx += 1
        if new_idx % 300 == 0:
            print(f"  {new_idx}/{len(all_eps)} written ({dropped} skipped)", flush=True)
    print(f"  kept {new_idx} ep, skipped {dropped}", flush=True)

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for r in episodes_out: f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for r in stats_out: f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": PROMPT0}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": PROMPT1}) + "\n")
    info = json.loads((TA / "self_built" / "A_smooth800_dagger_full" / "meta" / "info.json").read_text())
    info["features"]["observation.state"]["shape"] = [14]
    info["features"]["action"]["shape"] = [14]
    info["features"].pop("observation.depth.top_head", None)
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_tasks"] = 2
    info["total_videos"] = new_idx * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, new_idx)
    info["splits"] = {"train": f"0:{new_idx}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    ratio = f0 / max(1, f1)
    print(f"done -> {dst}  ({new_idx} ep, {total_frames} frames)", flush=True)
    print(f"  FRAMES domain0(Task_A)={f0}  domain1(AV1)={f1}  → 1:1 domain_weights=(1.0, {ratio:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="a_av1_merged")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy-video", action="store_true")
    a = ap.parse_args()
    build(a.out, a.dry_run, symlink_video=not a.copy_video)


if __name__ == "__main__":
    main()
