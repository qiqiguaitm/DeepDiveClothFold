"""Build smooth800 + dagger mixed datasets (D1 full-mix, D2 1:1-mix) for the dagger-validity study.

Plan: docs/training/future_plans/plans/dagger_validity_and_finetune_comparison.md

Two sources, DIFFERENT layouts — handled here:
  - smooth800: <SB>/A_new_smooth_800/base  (811 ep, videos under feature-key dirs observation.images.*,
               parquet cols incl task_index, NO intervention)
  - dagger:    <repo>/kai0/data/Task_A/vis_dagger/v2/<date>-v2  (210 ep over 4 dates, videos under BARE cam
               dirs top_head/hand_left/hand_right, parquet has extra 'intervention' col → DROPPED to align)

Modes:
  --mode D1   smooth800全量(811) + dagger全量(210) = 1021 ep  → A_smooth800_dagger_full
  --mode D2   smooth800抽210(seed) + dagger全量(210) = 420 ep → A_smooth800_dagger_1to1

Both: drop 'intervention' col, episode_index reindex 0..N-1, copy videos (self-contained), per-ep stats,
auto norm_stats (reuses norm_stats_from_dataset). 14D joint absolute, no trim, no EE6D.

Run with kai0 venv. KAI0_REPO_ROOT env overrides repo root (cross-cluster).
"""
import argparse, json, os, random, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_no_release import per_episode_stats, CAMERAS, CAM_DIRS, FPS, _maybe_norm_stats

_REPO = os.environ.get("KAI0_REPO_ROOT", "/vePFS/tim/workspace/deepdive_kai0")
SB = Path(f"{_REPO}/kai0/data/Task_A/self_built")
SMOOTH = SB / "A_new_smooth_800" / "base"
DAGGER_ROOT = Path(f"{_REPO}/kai0/data/Task_A/vis_dagger/v2")
DAGGER_DATES = ["2026-05-29-v2", "2026-06-01-v2", "2026-06-02-v2", "2026-06-03-v2"]
PROMPT = "Flatten and fold the cloth."
DROP_COLS = ["intervention"]   # dagger-only col, drop to align smooth schema


def _resolve_video(sv: Path):
    """Return a real readable path for sv, repathing broken vis_base symlinks left by the
    vis_base restructure (vis_base/<date> → vis_base/v2/<date>). None if truly missing."""
    if sv.exists():
        return sv
    if sv.is_symlink():
        tgt = os.readlink(sv)
        if "/vis_base/2026-" in tgt:                       # old pre-restructure target
            alt = Path(tgt.replace("/vis_base/2026-", "/vis_base/v2/2026-"))
            if alt.exists():
                return alt
    return None


def _src_episodes(src_dir, feature_key_video):
    """Yield (parquet_path, video_dir_fn) for each episode of a source.
    feature_key_video=True → videos under observation.images.<cam>; False → bare cam dirs."""
    pqs = sorted((src_dir / "data" / "chunk-000").glob("episode_*.parquet"))
    for pq in pqs:
        old = int(pq.stem.split("_")[1])
        def vsrc(cam, _old=old, _src=src_dir, _fk=feature_key_video):
            sub = cam if _fk else CAM_DIRS[cam]
            return _src / "videos" / "chunk-000" / sub / f"episode_{_old:06d}.mp4"
        yield pq, old, vsrc


def build(mode, out_name, seed, dry_run, compute_norm, action_dim, symlink_video=False):
    dst = SB / out_name
    # dagger 全量 first (its count drives D2's 1:1 smooth subset size)
    dagger_eps = []
    for d in DAGGER_DATES:
        dd = DAGGER_ROOT / d
        if not dd.exists():
            sys.exit(f"FATAL: dagger date dir missing: {dd}")
        dagger_eps += list(_src_episodes(dd, feature_key_video=False))
    smooth_all = list(_src_episodes(SMOOTH, feature_key_video=True))
    if mode == "D2":
        rng = random.Random(seed)                           # reproducible 1:1 subset = len(dagger)
        smooth_eps = sorted(rng.sample(smooth_all, len(dagger_eps)), key=lambda e: e[1])
    else:  # D1 = smooth 全量
        smooth_eps = smooth_all
    all_eps = [(pq, old, vfn, "smooth") for (pq, old, vfn) in smooth_eps] + \
              [(pq, old, vfn, "dagger") for (pq, old, vfn) in dagger_eps]
    print(f"[{mode}] smooth={len(smooth_eps)} + dagger={len(dagger_eps)} = {len(all_eps)} ep → {dst}", flush=True)
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
    for (pq, old, vsrc, tag) in all_eps:
        # resolve all 3 cam videos first; skip episode if any unresolvable (broken vis_base symlink)
        svs = [(cam, _resolve_video(vsrc(cam))) for cam in CAMERAS]
        if any(sv is None for _, sv in svs):
            dropped += 1
            print(f"  SKIP ep (tag={tag} old={old}): broken/missing video", flush=True)
            continue
        df = pd.read_parquet(pq)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])   # align schema
        n = len(df)
        df = df.reset_index(drop=True)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["episode_index"] = np.int64(new_idx)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS).astype(np.float32)
        df.to_parquet(dst / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet", index=False)
        for cam, sv in svs:
            dv = dst / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
            if symlink_video:
                os.symlink(os.path.realpath(sv), dv)   # fully resolve (smooth/base→vis_base/v2) so repath-able to cnbj
            else:
                shutil.copy(sv, dv)
        episodes_out.append({"episode_index": new_idx, "tasks": [PROMPT], "length": n})
        stats_out.append({"episode_index": new_idx, "stats": per_episode_stats(df)})
        total_frames += n
        new_idx += 1
        if new_idx % 100 == 0:
            print(f"  {new_idx} ep written ({dropped} skipped)", flush=True)
    print(f"  kept {new_idx} ep, skipped {dropped} (broken videos)", flush=True)

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for r in episodes_out:
            f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for r in stats_out:
            f.write(json.dumps(r) + "\n")
    shutil.copy(SMOOTH / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    info = json.loads((SMOOTH / "meta" / "info.json").read_text())
    # use new_idx (episodes actually written) NOT len(all_eps) — broken-video eps are skipped,
    # so the pre-skip count would leave phantom trailing indices that make lerobot's file-exists
    # assert fail → it then queries the HF hub for a "safe version" and crashes under HF_HUB_OFFLINE.
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_videos"] = new_idx * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, new_idx)   # single chunk-000 → chunks_size ≥ N (else ep//1000 → chunk-001 → assert fail → offline HF crash)
    info["splits"] = {"train": f"0:{new_idx}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"done → {dst}  ({new_idx} ep kept / {len(all_eps)} requested, {total_frames} frames)", flush=True)
    _maybe_norm_stats(dst, compute_norm, action_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["D1", "D2"],
                    help="D1=full-mix(811+210), D2=1:1-mix(smooth抽210+dagger210)")
    ap.add_argument("--out", required=True, help="output dataset name under self_built/")
    ap.add_argument("--seed", type=int, default=42, help="D2 smooth subset sampling seed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--action-dim", type=int, default=32)
    ap.add_argument("--no-norm-stats", action="store_true")
    ap.add_argument("--symlink-video", action="store_true",
                    help="symlink videos to resolved source instead of copy (small dataset, repath-able to other cluster)")
    a = ap.parse_args()
    build(a.mode, a.out, a.seed, a.dry_run, not a.no_norm_stats, a.action_dim, a.symlink_video)


if __name__ == "__main__":
    main()
