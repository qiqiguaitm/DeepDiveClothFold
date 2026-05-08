#!/usr/bin/env python3
"""Build mix_b6000_p1200 = 6512 official kai0_base+dagger + 1288 self_built (-new + mirror).

For 1:2 batch ratio (official:self_built), self_built is concat-duplicated 10×
in train set so default uniform sampler yields ~1:2 ratio.

Output structure:
  self_built/mix_b6000_p1200/
    ├── base/                  ~18,892 ep train (6412 official + 10×1258 self_built)
    ├── val_official/          100 ep (held-out from kai0_base+dagger)
    ├── val_self_built/        30 ep (held-out from -new + mirror, paired)
    └── manifest.json

Source paths:
  kai0_base:    {ROOT}/Task_A/kai0_base/         (3055 ep)
  kai0_dagger:  {ROOT}/Task_A/kai0_dagger/       (3457 ep)
  vis_base:     {SHARED_DS}/KAI0/Task_A/<date>-new/  (644 ep across 6 dates)
"""
from __future__ import annotations
import argparse, json, os, random, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Paths (adjust per machine via args)
DEFAULT_ROOT = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A")
DEFAULT_VIS_BASE_ROOT = Path("/data/shared/dataset/KAI0/Task_A")  # gf2/gf3
DEFAULT_OFFICIAL_ROOT = Path("/data/shared/dataset/Kai0_official/Task_A")  # gf2/gf3

CAMERAS = ("top_head", "hand_left", "hand_right")
PROMPT = "Flatten and fold the cloth."
FPS = 30
CHUNK = 0
LEFT_DIM = 7
RIGHT_DIM = 7

# ffmpeg: prefer imageio_ffmpeg (newer, encoder works in modern manner)
_IMAGEIO_FFMPEG = "/home/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
FFMPEG = _IMAGEIO_FFMPEG if Path(_IMAGEIO_FFMPEG).exists() else "/usr/bin/ffmpeg"


def _read_eps(meta_path: Path) -> list[dict]:
    out = []
    for line in meta_path.open():
        d = json.loads(line)
        ep_id = d.get("episode_index", d.get("episode_id"))
        if ep_id is None:
            raise ValueError(f"no episode index in {meta_path}: {d}")
        out.append({"src_ep": int(ep_id), "length": int(d["length"])})
    return out


def _ep_complete(parquet_p: Path, vid_root: Path, cam_naming: str, src_ep: int) -> bool:
    if not parquet_p.exists():
        return False
    for cam in CAMERAS:
        cam_dir = cam if cam_naming == "bare" else f"observation.images.{cam}"
        mp4 = vid_root / cam_dir / f"episode_{src_ep:06d}.mp4"
        if not mp4.exists():
            return False
    return True


def _probe_mp4(path: Path) -> tuple[Path, bool]:
    try:
        r = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"],
                           capture_output=True, timeout=30)
        return (path, r.returncode == 0)
    except Exception:
        return (path, False)


def collect_multichunk(root: Path, source_label: str) -> list[dict]:
    """Multi-chunk LeRobot dataset (kai0_base/kai0_dagger)."""
    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    eps = _read_eps(root / "meta" / "episodes.jsonl")
    items = []
    for e in eps:
        ch = e["src_ep"] // chunks_size
        pq_p = root / "data" / f"chunk-{ch:03d}" / f"episode_{e['src_ep']:06d}.parquet"
        vid_root = root / "videos" / f"chunk-{ch:03d}"
        if _ep_complete(pq_p, vid_root, "observation.images", e["src_ep"]):
            items.append({
                "src_dir": str(root),
                "src_ep": e["src_ep"],
                "src_chunk": ch,
                "length": e["length"],
                "source": source_label,
                "cam_naming": "observation.images",
                "kind": "official",
            })
    return items


def collect_vis_new(vis_root: Path, workers: int = 32) -> list[dict]:
    """vis_base/-new dirs (6 dates)."""
    candidates = []
    for date_dir in sorted(p for p in vis_root.iterdir()
                           if p.is_dir() and p.name.endswith("-new")):
        ep_file = date_dir / "meta" / "episodes.jsonl"
        if not ep_file.exists():
            continue
        for e in _read_eps(ep_file):
            pq_p = date_dir / "data" / f"chunk-{CHUNK:03d}" / f"episode_{e['src_ep']:06d}.parquet"
            vid_root = date_dir / "videos" / f"chunk-{CHUNK:03d}"
            if _ep_complete(pq_p, vid_root, "bare", e["src_ep"]):
                paths = [pq_p] + [vid_root / cam / f"episode_{e['src_ep']:06d}.mp4" for cam in CAMERAS]
                candidates.append((date_dir, e["src_ep"], e["length"], paths))

    print(f"  candidates with all 4 files: {len(candidates)}; probing mp4 integrity ...")
    bad = set()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe_mp4, mp): (i, mp)
                for i, (_, _, _, paths) in enumerate(candidates)
                for mp in paths[1:]}
        for fut in as_completed(futs):
            i, mp = futs[fut]
            _, ok = fut.result()
            if not ok:
                bad.add(i)
    if bad:
        print(f"  [warn] dropping {len(bad)} corrupt: {[(candidates[i][0].name, candidates[i][1]) for i in sorted(bad)]}")

    items = []
    for i, (d, ep, n, _) in enumerate(candidates):
        if i in bad:
            continue
        items.append({
            "src_dir": str(d),
            "src_ep": ep,
            "src_chunk": CHUNK,
            "length": n,
            "source": f"vis_base/{d.name}",
            "cam_naming": "bare",
            "kind": "self_orig",
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
    if "task_index" in t.column_names:
        t = t.set_column(t.schema.get_field_index("task_index"),
                         "task_index", pa.array([0] * n, type=pa.int64()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    return n


def write_parquet_mirrored(src: Path, dst: Path, new_ep: int, global_offset: int) -> int:
    """Write mirror parquet: swap left/right halves of state and action."""
    t = pq.read_table(src)
    n = t.num_rows
    state = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(t["action"].to_pylist(), dtype=np.float32)
    if state.shape[1] >= LEFT_DIM + RIGHT_DIM:
        state[:, :LEFT_DIM], state[:, LEFT_DIM:LEFT_DIM+RIGHT_DIM] = (
            state[:, LEFT_DIM:LEFT_DIM+RIGHT_DIM].copy(), state[:, :LEFT_DIM].copy())
    if action.shape[1] >= LEFT_DIM + RIGHT_DIM:
        action[:, :LEFT_DIM], action[:, LEFT_DIM:LEFT_DIM+RIGHT_DIM] = (
            action[:, LEFT_DIM:LEFT_DIM+RIGHT_DIM].copy(), action[:, :LEFT_DIM].copy())
    t = t.set_column(t.schema.get_field_index("observation.state"),
                     "observation.state", pa.array(state.tolist(), type=t.schema.field("observation.state").type))
    t = t.set_column(t.schema.get_field_index("action"),
                     "action", pa.array(action.tolist(), type=t.schema.field("action").type))
    t = t.set_column(t.schema.get_field_index("episode_index"),
                     "episode_index", pa.array([new_ep] * n, type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("index"),
                     "index", pa.array(list(range(global_offset, global_offset + n)), type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("timestamp"),
                     "timestamp", pa.array((np.arange(n, dtype=np.float32) / FPS), type=pa.float32()))
    if "task_index" in t.column_names:
        t = t.set_column(t.schema.get_field_index("task_index"),
                         "task_index", pa.array([0] * n, type=pa.int64()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    return n


def hflip_video(src: Path, dst: Path) -> tuple[bool, str]:
    """ffmpeg hflip → libx264 ultrafast (decode-friendly)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "hflip", "-c:v", "libx264",
           "-preset", "ultrafast", "-bf", "0",
           "-x264opts", "keyint=15:min-keyint=15:scenecut=0",
           "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return (p.returncode == 0, p.stderr[-200:] if p.returncode else "")


def write_official(info: dict, new_ep: int, dst: Path, global_offset: int) -> int:
    """Copy official ep (parquet rewrite + video symlink)."""
    src_pq = (Path(info["src_dir"]) / "data" / f"chunk-{info['src_chunk']:03d}" /
              f"episode_{info['src_ep']:06d}.parquet")
    dst_pq = dst / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
    n = copy_parquet(src_pq, dst_pq, new_ep, global_offset)
    src_vid_root = Path(info["src_dir"]) / "videos" / f"chunk-{info['src_chunk']:03d}"
    for cam in CAMERAS:
        src_cam = f"observation.images.{cam}"
        src = src_vid_root / src_cam / f"episode_{info['src_ep']:06d}.mp4"
        dst_cam = f"observation.images.{cam}"
        dst_vid = dst / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
        dst_vid.parent.mkdir(parents=True, exist_ok=True)
        if dst_vid.exists() or dst_vid.is_symlink():
            dst_vid.unlink()
        dst_vid.symlink_to(src.resolve())
    return n


def write_self_orig(info: dict, new_ep: int, dst: Path, global_offset: int) -> int:
    """Copy self-built original ep (parquet + video symlink)."""
    src_pq = (Path(info["src_dir"]) / "data" / f"chunk-{info['src_chunk']:03d}" /
              f"episode_{info['src_ep']:06d}.parquet")
    dst_pq = dst / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
    n = copy_parquet(src_pq, dst_pq, new_ep, global_offset)
    src_vid_root = Path(info["src_dir"]) / "videos" / f"chunk-{info['src_chunk']:03d}"
    for cam in CAMERAS:
        src_cam = cam   # bare naming
        src = src_vid_root / src_cam / f"episode_{info['src_ep']:06d}.mp4"
        dst_cam = f"observation.images.{cam}"
        dst_vid = dst / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
        dst_vid.parent.mkdir(parents=True, exist_ok=True)
        if dst_vid.exists() or dst_vid.is_symlink():
            dst_vid.unlink()
        dst_vid.symlink_to(src.resolve())
    return n


def write_self_mirror(orig_info: dict, new_ep: int, dst: Path, global_offset: int,
                       mirror_videos_dir: Path) -> int:
    """Write mirror ep:
    - parquet: swap left/right state/action
    - videos: hflip + swap hand_left↔hand_right (must be pre-encoded into mirror_videos_dir)
    """
    src_pq = (Path(orig_info["src_dir"]) / "data" / f"chunk-{orig_info['src_chunk']:03d}" /
              f"episode_{orig_info['src_ep']:06d}.parquet")
    dst_pq = dst / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
    n = write_parquet_mirrored(src_pq, dst_pq, new_ep, global_offset)
    # Video map: top_head (mirrored), hand_left (mirrored from hand_right), hand_right (mirrored from hand_left)
    src_date = Path(orig_info["src_dir"]).name
    src_ep = orig_info["src_ep"]
    cam_map = {
        "top_head": ("top_head", f"{src_date}_{src_ep:06d}.mp4"),
        "hand_left": ("hand_right", f"{src_date}_{src_ep:06d}.mp4"),
        "hand_right": ("hand_left", f"{src_date}_{src_ep:06d}.mp4"),
    }
    for dst_cam_pure, (src_cam_pure, fname) in cam_map.items():
        mirror_mp4 = mirror_videos_dir / src_cam_pure / fname  # pre-encoded mirror file
        if not mirror_mp4.exists():
            raise FileNotFoundError(f"mirror not pre-encoded: {mirror_mp4}")
        dst_cam = f"observation.images.{dst_cam_pure}"
        dst_vid = dst / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
        dst_vid.parent.mkdir(parents=True, exist_ok=True)
        if dst_vid.exists() or dst_vid.is_symlink():
            dst_vid.unlink()
        dst_vid.symlink_to(mirror_mp4.resolve())
    return n


def pre_encode_mirrors(self_origs: list[dict], mirror_videos_dir: Path, workers: int = 32) -> None:
    """Pre-encode hflip mirror videos for all self_built originals."""
    print(f"  pre-encoding {len(self_origs) * 3} mirror videos ...")
    jobs = []
    for info in self_origs:
        date = Path(info["src_dir"]).name
        ep = info["src_ep"]
        src_root = Path(info["src_dir"]) / "videos" / f"chunk-{info['src_chunk']:03d}"
        for cam in CAMERAS:  # encode each cam → store under mirror_videos_dir/<cam>/<date>_<ep>.mp4
            src = src_root / cam / f"episode_{ep:06d}.mp4"
            dst = mirror_videos_dir / cam / f"{date}_{ep:06d}.mp4"
            if dst.exists():
                continue
            jobs.append((src, dst))
    if not jobs:
        print("  all mirrors already encoded, skipping")
        return
    print(f"  encoding {len(jobs)} mirror videos with {workers} workers ...")

    done = 0
    fails = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(hflip_video, src, dst): (src, dst) for src, dst in jobs}
        for fut in as_completed(futs):
            src, dst = futs[fut]
            ok, err = fut.result()
            done += 1
            if not ok:
                fails += 1
                print(f"  FAIL {dst.name}: {err}")
            if done % 100 == 0 or done == len(jobs):
                print(f"   {done}/{len(jobs)}  fails={fails}")
    if fails:
        raise RuntimeError(f"{fails} mirror encodes failed")


def write_split(picks: list[dict], dst: Path, mirror_videos_dir: Path, info_template: dict, split_label: str):
    """Write a split: dst/{data,videos,meta}/."""
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    new_eps_meta = []
    total_frames = 0
    for new_ep, p in enumerate(picks):
        if p["kind"] == "official":
            n = write_official(p, new_ep, dst, total_frames)
        elif p["kind"] == "self_orig":
            n = write_self_orig(p, new_ep, dst, total_frames)
        elif p["kind"] == "self_mirror":
            n = write_self_mirror(p["orig_info"], new_ep, dst, total_frames, mirror_videos_dir)
        else:
            raise ValueError(p["kind"])
        new_eps_meta.append({
            "episode_index": new_ep,
            "tasks": [PROMPT],
            "length": n,
            "kind": p["kind"],
            "source": p.get("source") or p.get("orig_info", {}).get("source", "?"),
        })
        total_frames += n
    info_out = dict(info_template)
    info_out["total_episodes"] = len(picks)
    info_out["total_frames"] = total_frames
    info_out["total_videos"] = len(picks) * len(CAMERAS)
    info_out["total_chunks"] = 1
    info_out["chunks_size"] = max(1000, len(picks))
    info_out["splits"] = {split_label: f"0:{len(picks)}"}
    info_out["features"] = {k: v for k, v in info_out["features"].items()
                             if not k.startswith("observation.depth.")}
    info_out.pop("depth_path", None)
    (dst / "meta" / "info.json").write_text(json.dumps(info_out, indent=2))
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for em in new_eps_meta:
            f.write(json.dumps(em) + "\n")
    (dst / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    return len(picks), total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_ROOT / "self_built" / "mix_b6000_p1200"))
    ap.add_argument("--vis-base-root", default=str(DEFAULT_VIS_BASE_ROOT))
    ap.add_argument("--official-base", default=str(DEFAULT_OFFICIAL_ROOT / "base"))
    ap.add_argument("--official-dagger", default=str(DEFAULT_OFFICIAL_ROOT / "dagger"))
    ap.add_argument("--self-mult", type=int, default=10, help="self_built duplication for 1:2 batch ratio")
    ap.add_argument("--val-official", type=int, default=100)
    ap.add_argument("--val-self", type=int, default=30)  # 15 orig + 15 mirror pair
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reference-info", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-encode", action="store_true", help="skip mirror video encoding (already done)")
    args = ap.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        if args.force:
            print(f"[force] removing {out_root}")
            shutil.rmtree(out_root)
        else:
            print(f"ERROR: {out_root} exists. Use --force.", file=sys.stderr); sys.exit(2)

    print("=== sources ===")
    base = collect_multichunk(Path(args.official_base), "kai0_base")
    print(f"  kai0_base:    {len(base)} ep")
    dagger = collect_multichunk(Path(args.official_dagger), "kai0_dagger")
    print(f"  kai0_dagger:  {len(dagger)} ep")
    official = base + dagger
    print(f"  official total: {len(official)} ep")

    self_origs = collect_vis_new(Path(args.vis_base_root), workers=args.workers)
    print(f"  self_built (vis_base/-new): {len(self_origs)} ep clean")

    # Pre-encode all mirrors (one per orig)
    mirror_videos_dir = out_root.parent / "_mirror_videos"
    if not args.skip_encode:
        pre_encode_mirrors(self_origs, mirror_videos_dir, workers=args.workers)
    else:
        print("  --skip-encode: assuming mirrors already in", mirror_videos_dir)

    # Build mirror items (1:1 with self_origs)
    self_mirrors = [{"orig_info": info, "kind": "self_mirror",
                     "source": f"mirror({info['source']})", "length": info["length"]}
                    for info in self_origs]

    # Stratified val
    rng = random.Random(args.seed)
    rng.shuffle(official)
    rng.shuffle(self_origs)

    val_official_picks = official[:args.val_official]
    train_official = official[args.val_official:]

    val_orig_count = args.val_self // 2
    val_mir_count = args.val_self - val_orig_count
    val_self_orig = self_origs[:val_orig_count]
    val_self_orig_keys = set((o["src_dir"], o["src_ep"]) for o in val_self_orig)
    val_self_mirror = [m for m in self_mirrors
                        if (m["orig_info"]["src_dir"], m["orig_info"]["src_ep"]) in val_self_orig_keys][:val_mir_count]
    val_self_picks = val_self_orig + val_self_mirror

    # Train self: exclude val pairs (no leakage)
    train_self_orig = [o for o in self_origs
                        if (o["src_dir"], o["src_ep"]) not in val_self_orig_keys]
    train_self_mirror = [m for m in self_mirrors
                          if (m["orig_info"]["src_dir"], m["orig_info"]["src_ep"]) not in val_self_orig_keys]
    train_self_combined = train_self_orig + train_self_mirror

    print(f"\n=== split (seed={args.seed}) ===")
    print(f"  train official:    {len(train_official)} ep")
    print(f"  train self (1×):   {len(train_self_combined)} ep ({len(train_self_orig)} orig + {len(train_self_mirror)} mirror)")
    print(f"  train self ({args.self_mult}×, post oversample): {len(train_self_combined)*args.self_mult} ep")
    print(f"  val_official:      {len(val_official_picks)} ep")
    print(f"  val_self_built:    {len(val_self_picks)} ep ({len(val_self_orig)} orig + {len(val_self_mirror)} mirror)")
    print(f"")
    print(f"  TRAIN total: {len(train_official) + len(train_self_combined)*args.self_mult} ep")
    print(f"  Effective batch ratio (uniform sampler): "
          f"{len(train_official)}:{len(train_self_combined)*args.self_mult}")

    if args.dry_run:
        return

    # Build final train picks: official ×1 + self ×self_mult, then shuffle
    train_picks = list(train_official)
    for _ in range(args.self_mult):
        train_picks.extend(train_self_combined)
    rng.shuffle(train_picks)

    # Reference info (from kai0_base)
    if args.reference_info:
        info_template = json.loads(Path(args.reference_info).read_text())
    else:
        info_template = json.loads((Path(args.official_base) / "meta" / "info.json").read_text())

    print(f"\nwriting train -> {out_root}/base ...")
    nt, ft = write_split(train_picks, out_root / "base", mirror_videos_dir, info_template, "train")
    print(f"  {nt} ep / {ft} frames")

    print(f"\nwriting val_official -> {out_root}/val_official ...")
    nv1, fv1 = write_split(val_official_picks, out_root / "val_official", mirror_videos_dir,
                            info_template, "val")
    print(f"  {nv1} ep / {fv1} frames")

    print(f"\nwriting val_self_built -> {out_root}/val_self_built ...")
    nv2, fv2 = write_split(val_self_picks, out_root / "val_self_built", mirror_videos_dir,
                            info_template, "val")
    print(f"  {nv2} ep / {fv2} frames")

    (out_root / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "self_mult": args.self_mult,
        "train_total": nt,
        "train_official": len(train_official),
        "train_self_combined": len(train_self_combined),
        "val_official": nv1,
        "val_self_built": nv2,
        "prompt": PROMPT,
        "vis_base_root": args.vis_base_root,
        "official_sources": [args.official_base, args.official_dagger],
        "mirror_codec": "libx264 -preset ultrafast -bf 0 -x264opts keyint=15:scenecut=0",
    }, indent=2))

    print(f"\n[done] {out_root}")
    print(f"  train: {nt} ep / {ft} frames")
    print(f"  val_official: {nv1} / {fv1}, val_self_built: {nv2} / {fv2}")


if __name__ == "__main__":
    main()
