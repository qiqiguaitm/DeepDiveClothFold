"""Build the cross-embodiment pre-merged dataset for the per-DS-norm + conditioning run.

Plan: docs/training/future_plans/plans/corrected_plan_a_conditioning_premerge.md (+ this session).

ONE physical LeRobot dataset (healthy single-source path — NOT datasets_yaml/ConcatDataset,
which is the proven 0.47-collapse path). Domain is carried per-frame via `task_index`
(kai=0, vis=1); a tiny `ReadDatasetIdFromTaskIndex` transform (added in transforms.py) maps
task_index -> obs.dataset_id at train time. Balancing kai:vis to ~1:1 is done at train time by
a domain-weighted sampler (DomainWeightedSampler), NOT by copying episodes here.

Sources:
  - kai0_base    (domain 0)  3055 ep   real videos
  - kai0_dagger  (domain 0)  3457 ep   real videos; state/action declared [1,14] in info but
                                        stored (14,) — we squeeze defensively to (14,)
  - A_smooth800_dagger_full (domain 1, vis) 1033 ep  videos are symlinks -> vis_base/v2 + vis_dagger/v2

Output: kai0/data/Task_A/self_built/kai_vis_merged  (single chunk-000, chunks_size=max(1000,N),
videos symlinked to realpath so no disk blow-up). norm_stats are computed separately per-domain
by build_kai_vis_norm.py (C2), NOT here.

Run with kai0 venv. KAI0_REPO_ROOT env overrides repo root (cross-cluster).
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_no_release import per_episode_stats, CAMERAS, FPS

_REPO = os.environ.get("KAI0_REPO_ROOT", "/vePFS/tim/workspace/deepdive_kai0")
TA = Path(f"{_REPO}/kai0/data/Task_A")
PROMPT = "Flatten and fold the cloth."
KEEP = ["observation.state", "action"]   # rebuild the rest (frame_index/index/episode_index/timestamp/task_index)

# (source_dir, domain_id)  — domain_id becomes task_index (kai=0, vis=1)
SOURCES = [
    (TA / "kai0_base", 0),
    (TA / "kai0_dagger", 0),
    (TA / "self_built" / "A_smooth800_dagger_full", 1),
]


def _resolve_video(sv: Path):
    """Return a real readable path for sv (follow symlinks to their realpath). None if missing."""
    if sv.is_symlink() or sv.exists():
        rp = Path(os.path.realpath(sv))
        return rp if rp.exists() else None
    return None


def _src_episodes(src_dir):
    """Yield (parquet_path, old_idx, src_chunk) over ALL chunks of a source (sources with
    >1000 ep span chunk-000/001/... ; videos live in the matching chunk dir)."""
    cs = json.loads((src_dir / "meta" / "info.json").read_text()).get("chunks_size", 1000)
    pqs = sorted((src_dir / "data").glob("chunk-*/episode_*.parquet"))
    for pq in pqs:
        old = int(pq.stem.split("_")[1])
        yield pq, old, old // cs


def build(out_name, dry_run, symlink_video=True):
    dst = TA / "self_built" / out_name
    all_eps = []  # (parquet_path, old_idx, src_dir, domain_id)
    for src, dom in SOURCES:
        if not src.exists():
            sys.exit(f"FATAL: source missing: {src}")
        eps = list(_src_episodes(src))
        all_eps += [(pq, old, sc, src, dom) for (pq, old, sc) in eps]
        print(f"  {src.name}: {len(eps)} ep (domain {dom})", flush=True)
    n_kai = sum(1 for e in all_eps if e[4] == 0)
    n_vis = sum(1 for e in all_eps if e[4] == 1)
    print(f"[merge] total {len(all_eps)} ep  (kai={n_kai}, vis={n_vis}, ratio {n_kai/max(1,n_vis):.2f}:1) -> {dst}", flush=True)
    if dry_run:
        print("DRY — nothing written."); return

    if dst.exists():
        sys.exit(f"dst exists: {dst} (delete first)")
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir()
    for cam in CAMERAS:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    episodes_out, stats_out = [], []
    total_frames = 0
    new_idx = 0
    dropped = 0
    for (pq, old, sc, src, dom) in all_eps:
        svs = [(cam, _resolve_video(src / "videos" / f"chunk-{sc:03d}" / cam / f"episode_{old:06d}.mp4")) for cam in CAMERAS]
        if any(sv is None for _, sv in svs):
            dropped += 1
            print(f"  SKIP ep (src={src.name} old={old}): missing video", flush=True)
            continue
        df = pd.read_parquet(pq, columns=[c for c in KEEP if c])
        n = len(df)
        # squeeze state/action to (14,) defensively (kai0_dagger declares [1,14] but stores (14,))
        for col in ("observation.state", "action"):
            df[col] = [np.asarray(v, dtype=np.float32).reshape(-1)[:14] for v in df[col].to_numpy()]
        df = df.reset_index(drop=True)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["episode_index"] = np.int64(new_idx)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS).astype(np.float32)
        df["task_index"] = np.int64(dom)   # domain carried in task_index -> dataset_id at train time
        df = df[["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]]
        df.to_parquet(dst / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet", index=False)
        for cam, sv in svs:
            dv = dst / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
            if symlink_video:
                os.symlink(str(sv), dv)
            else:
                shutil.copy(sv, dv)
        episodes_out.append({"episode_index": new_idx, "tasks": [PROMPT], "length": n})
        stats_out.append({"episode_index": new_idx, "stats": per_episode_stats(df)})
        total_frames += n
        new_idx += 1
        if new_idx % 500 == 0:
            print(f"  {new_idx}/{len(all_eps)} ep written ({dropped} skipped)", flush=True)
    print(f"  kept {new_idx} ep, skipped {dropped}", flush=True)

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for r in episodes_out:
            f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for r in stats_out:
            f.write(json.dumps(r) + "\n")
    # 2 tasks: index 0 = kai, index 1 = vis (same prompt; task_index encodes domain)
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": PROMPT}) + "\n")
    # info.json: base on kai0_base, fix shapes [14], counts, chunks_size>=N (single chunk-000)
    info = json.loads((TA / "kai0_base" / "meta" / "info.json").read_text())
    info["features"]["observation.state"]["shape"] = [14]
    info["features"]["action"]["shape"] = [14]
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_tasks"] = 2
    info["total_videos"] = new_idx * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, new_idx)   # single chunk-000 -> chunks_size must be >= N
    info["splits"] = {"train": f"0:{new_idx}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"done -> {dst}  ({new_idx} ep, {total_frames} frames, chunks_size={info['chunks_size']})", flush=True)
    print(f"  domain split: kai(task_index=0)={n_kai}  vis(task_index=1)={n_vis}  (minus {dropped} dropped)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kai_vis_merged")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy-video", action="store_true", help="copy instead of symlink (default symlink)")
    a = ap.parse_args()
    build(a.out, a.dry_run, symlink_video=not a.copy_video)


if __name__ == "__main__":
    main()
