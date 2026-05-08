#!/usr/bin/env python3
"""Build a visrobot01-only Task_A dataset on gf1.

Unifies visrobot01/KAI0/Task_A/<date>/{base,dagger} into a single LeRobot v2.1
dataset at DST_ROOT, keeping only episodes with complete data (parquet + 3 RGB
camera videos). Splits into base/ (train) and val/.

Usage:
    python build_task_a_visrobot01_only.py                   # run
    python build_task_a_visrobot01_only.py --val-size 30     # hold-out 30 ep
    python build_task_a_visrobot01_only.py --dry-run         # plan only
    python build_task_a_visrobot01_only.py --force           # overwrite dest
"""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VIS_ROOT = "/vePFS/visrobot01/KAI0/Task_A"
REFERENCE_INFO = "/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/base/meta/info.json"
DST_ROOT = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A_visrobot01_only"
SEED = 42
FPS = 30
CAMERAS = ("top_head", "hand_left", "hand_right")
PROMPT = "Flatten and fold the cloth."
CHUNK = 0


def _read_ep(meta_path: Path) -> list[dict]:
    out = []
    for line in meta_path.open():
        d = json.loads(line)
        ep_id = d.get("episode_index", d.get("episode_id"))
        if ep_id is None:
            raise ValueError(f"no episode index in {meta_path}: {d}")
        out.append({"src_ep": ep_id, "length": d["length"]})
    return out


def _ep_has_all_cams(kind_dir: Path, ep_id: int) -> bool:
    pq_f = kind_dir / "data" / f"chunk-{CHUNK:03d}" / f"episode_{ep_id:06d}.parquet"
    if not pq_f.exists():
        return False
    for cam in CAMERAS:
        mp4 = kind_dir / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{ep_id:06d}.mp4"
        if not mp4.exists():
            return False
    return True


def collect_visrobot01(root: Path) -> list[dict]:
    """Walk date-subdirs, keep only episodes with complete data."""
    items = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("2026"):
            continue
        for kind in ("base", "dagger"):
            kd = date_dir / kind
            ep_file = kd / "meta" / "episodes.jsonl"
            if not ep_file.exists():
                continue
            all_eps = _read_ep(ep_file)
            for e in all_eps:
                if _ep_has_all_cams(kd, e["src_ep"]):
                    items.append({
                        "src_dir": str(kd),
                        "src_ep": e["src_ep"],
                        "length": e["length"],
                        "source": f"{date_dir.name}/{kind}",
                    })
    return items


def copy_parquet(src: Path, dst: Path, new_ep: int, global_offset: int) -> int:
    t = pq.read_table(src)
    n = t.num_rows
    t = t.set_column(t.schema.get_field_index("episode_index"),
                     "episode_index", pa.array([new_ep] * n, type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("index"),
                     "index", pa.array(list(range(global_offset, global_offset + n)), type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("timestamp"),
                     "timestamp", pa.array((np.arange(n, dtype=np.float32) / FPS), type=pa.float32()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    return n


def symlink_video(info: dict, new_ep: int, dst_root: Path):
    src_vid_root = Path(info["src_dir"]) / "videos" / f"chunk-{CHUNK:03d}"
    for cam in CAMERAS:
        src = src_vid_root / cam / f"episode_{info['src_ep']:06d}.mp4"
        if not src.exists():
            raise FileNotFoundError(src)
        dst_cam = f"observation.images.{cam}"
        dst = dst_root / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def write_split(dst_split: Path, picks: list[dict]) -> int:
    (dst_split / "meta").mkdir(parents=True)
    new_episodes = []
    total_frames = 0
    for new_ep, info in enumerate(picks):
        src_pq = Path(info["src_dir"]) / "data" / f"chunk-{CHUNK:03d}" / f"episode_{info['src_ep']:06d}.parquet"
        dst_pq = dst_split / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        n = copy_parquet(src_pq, dst_pq, new_ep, total_frames)
        symlink_video(info, new_ep, dst_split)
        new_episodes.append({
            "episode_index": new_ep,
            "tasks": [PROMPT],
            "length": n,
            "orig_source": info["source"],
            "orig_ep": info["src_ep"],
        })
        total_frames += n
    info_template = json.loads(Path(REFERENCE_INFO).read_text())
    info_template["total_episodes"] = len(picks)
    info_template["total_frames"] = total_frames
    info_template["total_videos"] = len(picks) * len(CAMERAS)
    info_template["total_chunks"] = 1
    info_template["splits"] = {dst_split.name: f"0:{len(picks)}"}
    info_template["features"] = {k: v for k, v in info_template["features"].items()
                                  if not k.startswith("observation.depth.")}
    info_template.pop("depth_path", None)
    (dst_split / "meta" / "info.json").write_text(json.dumps(info_template, indent=2))
    with (dst_split / "meta" / "episodes.jsonl").open("w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")
    (dst_split / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    return total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-root", default=VIS_ROOT)
    ap.add_argument("--out-root", default=DST_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-size", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    items = collect_visrobot01(Path(args.vis_root))
    N_total = len(items)
    print(f"visrobot01 complete episodes: {N_total}")
    sources = sorted(set(x["source"] for x in items))
    for s in sources:
        cnt = sum(1 for x in items if x["source"] == s)
        print(f"  {s}: {cnt}")

    if N_total < args.val_size + 10:
        print(f"ERROR: too few eps ({N_total}) for val-size={args.val_size}", file=sys.stderr)
        sys.exit(3)

    # Stratified split: take val_per_source proportional from each source
    rng = random.Random(args.seed)
    val_target_per_source = args.val_size // len(sources)
    val_items, train_items = [], []
    for s in sources:
        src_items = [x for x in items if x["source"] == s]
        rng.shuffle(src_items)
        v = min(val_target_per_source, max(1, len(src_items) // 10))
        val_items.extend(src_items[:v])
        train_items.extend(src_items[v:])

    print(f"\nsplit: train={len(train_items)} val={len(val_items)}")

    if args.dry_run:
        print("\n--- dry-run: first 10 train / all val ---")
        for ep in train_items[:10]:
            print(f"  TRAIN {ep['source']}  ep={ep['src_ep']}  len={ep['length']}")
        for ep in val_items:
            print(f"  VAL   {ep['source']}  ep={ep['src_ep']}  len={ep['length']}")
        return

    dst = Path(args.out_root)
    if dst.exists():
        if args.force:
            print(f"[force] removing {dst}")
            shutil.rmtree(dst)
        else:
            print(f"ERROR: {dst} exists. Use --force.", file=sys.stderr); sys.exit(2)

    print(f"\nwriting train → {dst}/base ...")
    tf = write_split(dst / "base", train_items)
    print(f"  train: {len(train_items)} eps, {tf} frames")
    print(f"writing val → {dst}/val ...")
    vf = write_split(dst / "val", val_items)
    print(f"  val:   {len(val_items)} eps, {vf} frames")

    (dst / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "prompt": PROMPT,
        "source": "visrobot01 ONLY",
        "train_episodes": len(train_items),
        "train_frames": tf,
        "val_episodes": len(val_items),
        "val_frames": vf,
        "sources": sources,
    }, indent=2))

    print(f"\n✅ built: {dst}")
    print(f"   train: {len(train_items)} eps / {tf} frames")
    print(f"   val:   {len(val_items)} eps / {vf} frames")


if __name__ == "__main__":
    main()
