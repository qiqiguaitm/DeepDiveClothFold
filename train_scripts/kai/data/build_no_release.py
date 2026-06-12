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
# 2026-06-02: vis_base v2 数据归入 vis_base/v2/ 子目录; v3 (裁投放) 并列在 vis_base/v3/.
VIS_BASE = Path(f"{_REPO}/kai0/data/Task_A/vis_base/v2")          # 源: v2 各日期 <date>-v2
V3_ROOT = Path(f"{_REPO}/kai0/data/Task_A/vis_base/v3")           # per-date 模式输出: <date>-v3
V3_2_ROOT = Path(f"{_REPO}/kai0/data/Task_A/vis_base/v3.2")       # idle_downsample 输出: <date>-v3.2 (v3 前端裁 + 中段选择性)
VIS_DAGGER_V2 = Path(f"{_REPO}/kai0/data/Task_A/vis_dagger/v2")   # dagger 源: v2 各日期 <date>-v2
VIS_DAGGER_V3 = Path(f"{_REPO}/kai0/data/Task_A/vis_dagger/v3")   # dagger v3 输出: <date>-v3 (同 base 前端裁 + drop depth)
DST_ROOT = Path(f"{_REPO}/kai0/data/Task_A/self_built")           # 合并模式输出 (原 A_0522_0526_*)
DATES = ["2026-05-22-v2", "2026-05-26-v2"]
CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
CAM_DIRS = {"observation.images.top_head": "top_head",
            "observation.images.hand_left": "hand_left",
            "observation.images.hand_right": "hand_right"}
FPS = 30
# 并行: os.cpu_count() 在容器里常被 cgroup 误报 (本机报 13, 实际 56) → 用 sched_getaffinity.
# BUILD_WORKERS env 可覆盖. 每 worker ENC_THREADS 编码线程; workers×threads ≈ 核数.
def _avail_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8
ENC_THREADS = int(os.environ.get("BUILD_ENC_THREADS", "2"))
BUILD_WORKERS = int(os.environ.get("BUILD_WORKERS", str(max(1, _avail_cores() // ENC_THREADS))))
# 瓶颈实测是 PyAV decode→encode 流水 (单进程~0.4核, 加 worker 收益递减). preset 提速最有效:
# veryfast(默认, 近无损) → ultrafast(再快~2x, 文件略大但 VLA 训练无所谓). BUILD_PRESET 可覆盖.
ENC_PRESET = os.environ.get("BUILD_PRESET", "veryfast")
ARM_DIMS = list(range(0, 6)) + list(range(7, 13))   # 12 arm dims (exclude dim6 L_grip, dim13 R_grip)
THR = 3e-3      # rad/frame: sustained mean |Δa| over arm dims => "moving"
# v3.2 selective idle-downsample params (env-overridable). Conservative start:
IDLE_THR = float(os.environ.get("V32_IDLE_THR", "2e-3"))   # rad/frame: per-frame |Δa| below this = idle
KEEP_LEN = int(os.environ.get("V32_KEEP_LEN", "15"))       # idle run ≤ this many frames (≤0.5s) = short settle, kept whole
DOWNSAMPLE_K = int(os.environ.get("V32_K", "3"))           # long idle run: keep boundaries + every k-th frame
# v3.2 GRASP PROTECTION (2026-06-09 fix): the arm-only idle detector (ARM_DIMS excludes
# gripper) classifies grasp/regrasp as "idle" (arm holds still while gripper closes) → the
# careful grasp dwell got downsampled → real-machine under-reach / 抓不到衣角. Fix: force-keep
# a window around every gripper-state transition so grasp segments are never thinned.
GRIP_DIMS = [6, 13]                                        # L/R gripper action dims (excluded from ARM_DIMS)
GRIP_THR = float(os.environ.get("V32_GRIP_THR", "0.02"))  # |Δgrip| above this = gripper acting (grasp/release)
GRIP_GUARD = int(os.environ.get("V32_GRIP_GUARD", "30"))  # force-keep ±this many frames around a gripper transition (~1s dwell)
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
    out_stream.options = {"crf": "18", "preset": ENC_PRESET, "threads": str(ENC_THREADS)}

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


def idle_keep_indices(action: np.ndarray, idle_thr: float = IDLE_THR,
                      keep_len: int = KEEP_LEN, k: int = DOWNSAMPLE_K,
                      grip_thr: float = GRIP_THR, grip_guard: int = GRIP_GUARD) -> np.ndarray:
    """v3.2 selective idle handling: return sorted frame indices to KEEP.
    A frame counts as "moving" (always kept) if the ARM moves, the GRIPPER changes state,
    or it falls within ±grip_guard frames of a gripper transition (grasp/regrasp dwell
    protection — arm holds still while gripper closes, so the arm-only detector would
    otherwise call it idle and downsample the grasp). Short idle runs (≤keep_len) kept
    whole (functional settle); long idle runs (>keep_len) compressed: keep both boundaries
    + every k-th frame. (Boundaries preserved so action chunks don't jump across the seam.)"""
    T = len(action)
    if T <= 1:
        return np.arange(T)
    da = np.abs(np.diff(action[:, ARM_DIMS], axis=0)).mean(axis=1)      # (T-1,)
    moving = np.concatenate([da > idle_thr, [True]])                    # (T,); last frame kept
    # grasp protection: force-keep gripper transitions + a guard window around each.
    dg = np.abs(np.diff(action[:, GRIP_DIMS], axis=0)).max(axis=1)      # (T-1,)
    grip_evt = np.concatenate([[False], dg > grip_thr])                 # (T,); frame where new grip state reached
    if grip_evt.any() and grip_guard > 0:
        guard = np.zeros(T, dtype=bool)
        for e in np.nonzero(grip_evt)[0]:
            guard[max(0, e - grip_guard):min(T, e + grip_guard + 1)] = True
        moving = moving | guard
    keep = np.zeros(T, dtype=bool)
    i = 0
    while i < T:
        if moving[i]:
            keep[i] = True
            i += 1
            continue
        j = i
        while j < T and not moving[j]:
            j += 1
        L = j - i
        if L <= keep_len:
            keep[i:j] = True                       # short settle → keep whole
        else:
            keep[i] = True; keep[j - 1] = True     # boundaries
            keep[i:j][::k] = True                  # every k-th in the run
        i = j
    return np.nonzero(keep)[0]


def select_video_pyav(src_mp4: Path, dst_mp4: Path, keep_idx: np.ndarray, expected_frames: int):
    """Re-encode src_mp4 keeping ONLY the decode-indices in keep_idx (in order). Assert == expected."""
    import av
    keep_set = set(int(x) for x in keep_idx)
    in_c = av.open(str(src_mp4))
    in_stream = in_c.streams.video[0]
    in_stream.thread_type = "AUTO"
    out_c = av.open(str(dst_mp4), mode="w")
    out_stream = out_c.add_stream("libx264", rate=FPS)
    out_stream.width = in_stream.codec_context.width
    out_stream.height = in_stream.codec_context.height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "18", "preset": ENC_PRESET, "threads": str(ENC_THREADS)}
    written, idx = 0, 0
    for frame in in_c.decode(video=0):
        if idx in keep_set:
            for pkt in out_stream.encode(frame.reformat(format="yuv420p")):
                out_c.mux(pkt)
            written += 1
        idx += 1
    for pkt in out_stream.encode():
        out_c.mux(pkt)
    in_c.close(); out_c.close()
    if written != expected_frames:
        raise RuntimeError(f"v3.2 video frame mismatch {src_mp4.name}: wrote {written}, "
                           f"expected {expected_frames} (decoded {idx})")


def _select_job(job):
    src_mp4, dst_mp4, keep_idx, new_len = job
    select_video_pyav(Path(src_mp4), Path(dst_mp4), np.asarray(keep_idx), new_len)
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


def _maybe_norm_stats(dst, compute: bool, action_dim: int):
    """Auto-(re)compute norm_stats.json from the *just-built* dataset. Build success is preserved
    even if this fails (loud warning + manual fallback)."""
    manual = f"python {Path(__file__).resolve().parent}/norm_stats_from_dataset.py {dst} --action-dim {action_dim}"
    if not compute:
        print(f"  [norm_stats] skipped (--no-norm-stats). Run manually: {manual}", flush=True)
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from norm_stats_from_dataset import compute_norm_stats
        print(f"  [norm_stats] auto-computing from built dataset (action_dim={action_dim})...", flush=True)
        compute_norm_stats(dst, action_dim=action_dim)
    except Exception as e:
        print(f"  ⚠️ [norm_stats] AUTO-COMPUTE FAILED: {e}\n"
              f"     dataset built OK but norm_stats.json NOT written — run manually (kai0 venv):\n"
              f"     {manual}", flush=True)


def build_per_date_v3(date_v2: str, dry_run: bool = False, compute_norm: bool = True, action_dim: int = 32,
                      src_root: Path | None = None, dst_root: Path | None = None) -> dict:
    """Per-date v3: trim 投放 static head from every episode of <src_root>/<date>-v2,
    write <dst_root>/<date>-v3. PRESERVES original episode_index (no merge/renumber),
    drops depth (RGB-only), per-ep assert video frames == parquet rows.
    src_root/dst_root default to vis_base v2/v3; pass VIS_DAGGER_V2/V3 for dagger.

    Returns a report dict. Reuses motion_onset / trim_video_pyav / per_episode_stats."""
    src_root = src_root or VIS_BASE
    dst_root = dst_root or V3_ROOT
    src = src_root / date_v2
    date_v3 = date_v2.replace("-v2", "-v3")
    dst = dst_root / date_v3
    if not src.exists():
        raise FileNotFoundError(f"src date dir not found: {src}")

    parquets = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
    src_eps = {json.loads(l).get("episode_id", json.loads(l).get("episode_index")): json.loads(l)
               for l in (src / "meta" / "episodes.jsonl").open()}
    print(f"[{date_v2}→{date_v3}] {len(parquets)} episodes", flush=True)

    if not dry_run:
        if dst.exists():
            # complete = meta/info.json present (written last). 半成品 (被 kill 打断, 无 meta) → 删除重建.
            if (dst / "meta" / "info.json").exists():
                print(f"  skip {date_v3}: already complete (meta/info.json present)", flush=True)
                return {"date_v2": date_v2, "date_v3": date_v3, "skipped": True}
            print(f"  ⚠️ {date_v3}: incomplete (no meta/info.json) → removing + rebuilding", flush=True)
            shutil.rmtree(dst)
        (dst / "data" / "chunk-000").mkdir(parents=True)
        (dst / "meta").mkdir()
        for cam in CAMERAS:
            (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    episodes_out, stats_out, video_jobs, cut_report = [], [], [], []
    total_frames = 0
    for pq in parquets:
        ep_id = int(pq.stem.split("_")[1])   # preserve original episode number
        df = pd.read_parquet(pq)
        T0 = len(df)
        action = np.stack(df["action"].to_numpy()).astype(np.float64)
        onset = motion_onset(action)
        cut = max(0, onset - MARGIN)
        cut_report.append(cut)
        new_len = T0 - cut

        if not dry_run:
            sub = df.iloc[cut:].copy().reset_index(drop=True)
            sub["frame_index"] = np.arange(new_len, dtype=np.int64)
            sub["episode_index"] = np.int64(ep_id)
            # index column = global running index within THIS date (per-date dataset is standalone)
            sub["index"] = np.arange(total_frames, total_frames + new_len, dtype=np.int64)
            sub["timestamp"] = (np.arange(new_len, dtype=np.float32) / FPS).astype(np.float32)
            sub.to_parquet(dst / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet", index=False)
            for cam in CAMERAS:
                sv = src / "videos" / "chunk-000" / CAM_DIRS[cam] / f"episode_{ep_id:06d}.mp4"
                dv = dst / "videos" / "chunk-000" / cam / f"episode_{ep_id:06d}.mp4"  # feature-key dir
                video_jobs.append((str(sv), str(dv), cut, new_len))
            meta = src_eps.get(ep_id, {})
            episodes_out.append({"episode_index": ep_id,
                                 "tasks": [meta.get("prompt", "Flatten and fold the cloth.")],
                                 "length": new_len})
            stats_out.append({"episode_index": ep_id, "stats": per_episode_stats(sub)})
        total_frames += new_len

    cr = np.array(cut_report)
    rep = {"date_v2": date_v2, "date_v3": date_v3, "episodes": len(parquets),
           "total_frames": int(total_frames), "cut_median": int(np.median(cr)) if len(cr) else 0,
           "cut_max": int(cr.max()) if len(cr) else 0,
           "dropped_pct": float(100 * cr.sum() / (cr.sum() + total_frames)) if total_frames else 0.0}
    if dry_run:
        print(f"  DRY: cut median={rep['cut_median']} max={rep['cut_max']} dropped={rep['dropped_pct']:.1f}%", flush=True)
        return rep

    # parallel video trim (asserts frame-count == new_len inside _trim_job)
    if video_jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        nproc = BUILD_WORKERS
        print(f"  trimming {len(video_jobs)} videos ({nproc} workers)...", flush=True)
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for fut in as_completed({ex.submit(_trim_job, j): j for j in video_jobs}):
                fut.result()

    # meta
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for r in episodes_out:
            f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for r in stats_out:
            f.write(json.dumps(r) + "\n")
    shutil.copy(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = len(parquets)
    info["total_frames"] = total_frames
    info["total_videos"] = len(parquets) * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, len(parquets))   # single chunk-000 layout → chunks_size must be ≥ N,
    info["splits"] = {"train": f"0:{len(parquets)}"}  # else lerobot ep//1000 expects chunk-001 → file-assert → offline HF crash
    info["features"].pop("observation.depth.top_head", None)   # v3 drops depth
    info.pop("depth_path", None)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  done → {dst}  (cut median={rep['cut_median']} dropped={rep['dropped_pct']:.1f}%)", flush=True)
    _maybe_norm_stats(dst, compute_norm and not dry_run, action_dim)
    return rep


def build_per_date_v32(date_v3: str, dry_run: bool = False, compute_norm: bool = True, action_dim: int = 32,
                       idle_thr: float = IDLE_THR, keep_len: int = KEEP_LEN, k: int = DOWNSAMPLE_K) -> dict:
    """v3.2: from vis_base/v3/<date>-v3 (front already trimmed), selectively downsample MIDDLE idle
    runs (keep short settle, compress long pause) → vis_base/v3.2/<date>-v3.2. Preserves ep ids,
    rebuilds frame_index/index/timestamp, re-encodes the 3 mp4s to the kept frames (assert ==)."""
    src = V3_ROOT / date_v3
    date_v32 = date_v3.replace("-v3", "-v3.2")
    dst = V3_2_ROOT / date_v32
    if not src.exists():
        raise FileNotFoundError(f"v3 src not found: {src}")
    parquets = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
    src_eps = {json.loads(l).get("episode_id", json.loads(l).get("episode_index")): json.loads(l)
               for l in (src / "meta" / "episodes.jsonl").open()}
    print(f"[{date_v3}→{date_v32}] {len(parquets)} eps  (idle_thr={idle_thr} keep_len={keep_len} k={k})", flush=True)

    if not dry_run:
        if (dst / "meta" / "info.json").exists():
            print(f"  skip {date_v32}: already complete", flush=True)
            return {"date_v3": date_v3, "date_v32": date_v32, "skipped": True}
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "data" / "chunk-000").mkdir(parents=True)
        (dst / "meta").mkdir()
        for cam in CAMERAS:
            (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    episodes_out, stats_out, video_jobs = [], [], []
    total_frames, total_in = 0, 0
    for pq in parquets:
        ep_id = int(pq.stem.split("_")[1])
        df = pd.read_parquet(pq)
        T0 = len(df)
        action = np.stack(df["action"].to_numpy()).astype(np.float64)
        keep_idx = idle_keep_indices(action, idle_thr, keep_len, k)
        new_len = len(keep_idx)
        total_in += T0
        if not dry_run:
            sub = df.iloc[keep_idx].copy().reset_index(drop=True)
            sub["frame_index"] = np.arange(new_len, dtype=np.int64)
            sub["episode_index"] = np.int64(ep_id)
            sub["index"] = np.arange(total_frames, total_frames + new_len, dtype=np.int64)
            sub["timestamp"] = (np.arange(new_len, dtype=np.float32) / FPS).astype(np.float32)
            sub.to_parquet(dst / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet", index=False)
            for cam in CAMERAS:
                sv = src / "videos" / "chunk-000" / cam / f"episode_{ep_id:06d}.mp4"  # v3 = feature-key dirs
                dv = dst / "videos" / "chunk-000" / cam / f"episode_{ep_id:06d}.mp4"
                video_jobs.append((str(sv), str(dv), keep_idx.tolist(), new_len))
            meta = src_eps.get(ep_id, {})
            episodes_out.append({"episode_index": ep_id,
                                 "tasks": meta.get("tasks", ["Flatten and fold the cloth."]),
                                 "length": new_len})
            stats_out.append({"episode_index": ep_id, "stats": per_episode_stats(sub)})
        total_frames += new_len

    rep = {"date_v3": date_v3, "date_v32": date_v32, "episodes": len(parquets),
           "frames_in": total_in, "frames_out": total_frames,
           "compressed_pct": float(100 * (total_in - total_frames) / total_in) if total_in else 0.0}
    if dry_run:
        print(f"  DRY: {total_in}→{total_frames} frames (compressed {rep['compressed_pct']:.1f}%)", flush=True)
        return rep

    if video_jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"  re-encoding {len(video_jobs)} videos ({BUILD_WORKERS} workers)...", flush=True)
        with ProcessPoolExecutor(max_workers=BUILD_WORKERS) as ex:
            for fut in as_completed({ex.submit(_select_job, j): j for j in video_jobs}):
                fut.result()

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for r in episodes_out:
            f.write(json.dumps(r) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for r in stats_out:
            f.write(json.dumps(r) + "\n")
    shutil.copy(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = len(parquets)
    info["total_frames"] = total_frames
    info["total_videos"] = len(parquets) * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, len(parquets))
    info["splits"] = {"train": f"0:{len(parquets)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  done → {dst}  ({total_in}→{total_frames}, compressed {rep['compressed_pct']:.1f}%)", flush=True)
    _maybe_norm_stats(dst, compute_norm and not dry_run, action_dim)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "no_release"],
                    help="legacy merge mode: 5-22+5-26 → single self_built/A_0522_0526_{raw,no_release}")
    ap.add_argument("--per-date", nargs="+", metavar="DATE",
                    help="per-date v3 mode: trim 投放 head, write <root>/v3/<date>-v3 (preserve ep ids, "
                         "no depth). Pass dates like 2026-05-22-v2, or 'all' for every <date>-v2 under the chosen root.")
    ap.add_argument("--src-kind", choices=["base", "dagger"], default="base",
                    help="per-date v3 source: 'base' (vis_base/v2→v3, default) or 'dagger' (vis_dagger/v2→v3).")
    ap.add_argument("--symlink-video", action="store_true",
                    help="raw mode only: symlink videos instead of copy (saves disk)")
    ap.add_argument("--dry-run", action="store_true", help="compute cuts + report, write nothing")
    ap.add_argument("--action-dim", type=int, default=32, help="norm_stats action_dim to pad to (pi0/pi05=32)")
    ap.add_argument("--no-norm-stats", action="store_true",
                    help="skip auto norm_stats after build (default: auto-(re)compute from the built dataset)")
    ap.add_argument("--merge-dates", nargs="+", metavar="DATE",
                    help="generalized merge: merge these date dirs into ONE self_built dataset "
                         "(e.g. 2026-05-18-v3 2026-05-19-v3 ...). Requires --merge-out + --mode. "
                         "trim per --mode (use raw for already-trimmed v3 sources).")
    ap.add_argument("--merge-src", choices=["v2", "v3", "v3.2"], default="v2",
                    help="source root for --merge-dates: v2 (raw) / v3 (front-trimmed) / v3.2 (idle-downsampled)")
    ap.add_argument("--merge-out", metavar="NAME",
                    help="output dataset name under self_built/ for --merge-dates")
    ap.add_argument("--per-date-v32", nargs="+", metavar="DATE",
                    help="v3.2 selective idle-downsample: from vis_base/v3/<date>-v3 compress middle pauses "
                         "→ vis_base/v3.2/<date>-v3.2. Pass dates like 2026-05-18-v3, or 'all' for every -v3.")
    ap.add_argument("--idle-thr", type=float, default=IDLE_THR, help="v3.2 idle |Δa| threshold")
    ap.add_argument("--keep-len", type=int, default=KEEP_LEN, help="v3.2 short-settle keep length (frames)")
    ap.add_argument("--k", type=int, default=DOWNSAMPLE_K, help="v3.2 long-pause keep-every-k")
    args = ap.parse_args()

    # ---- v3.2 selective idle-downsample mode ----
    if args.per_date_v32:
        if args.per_date_v32 == ["all"]:
            dates = sorted(d.name for d in V3_ROOT.iterdir() if d.is_dir() and d.name.endswith("-v3"))
        else:
            dates = args.per_date_v32
        print(f"v3.2 idle_downsample: {len(dates)} dates → vis_base/v3.2/ "
              f"(idle_thr={args.idle_thr} keep_len={args.keep_len} k={args.k})", flush=True)
        reps = [build_per_date_v32(d, dry_run=args.dry_run, compute_norm=not args.no_norm_stats,
                                   action_dim=args.action_dim, idle_thr=args.idle_thr,
                                   keep_len=args.keep_len, k=args.k) for d in dates]
        print("\n=== v3.2 summary ===")
        for r in reps:
            if r.get("skipped"):
                print(f"  {r['date_v32']}: SKIPPED")
            else:
                print(f"  {r['date_v32']}: {r['episodes']} ep, {r['frames_in']}→{r['frames_out']} "
                      f"frames (compressed {r['compressed_pct']:.1f}%)")
        return

    # ---- per-date v3 mode ----
    if args.per_date:
        src_root, dst_root = (VIS_DAGGER_V2, VIS_DAGGER_V3) if args.src_kind == "dagger" else (VIS_BASE, V3_ROOT)
        if args.per_date == ["all"]:
            dates = sorted(d.name for d in src_root.iterdir() if d.is_dir() and d.name.endswith("-v2"))
        else:
            dates = args.per_date
        print(f"per-date v3 ({args.src_kind}): {len(dates)} dates → {dst_root}", flush=True)
        reps = [build_per_date_v3(d, dry_run=args.dry_run,
                                  compute_norm=not args.no_norm_stats, action_dim=args.action_dim,
                                  src_root=src_root, dst_root=dst_root) for d in dates]
        print("\n=== per-date v3 summary ===")
        for r in reps:
            if r.get("skipped"):
                print(f"  {r['date_v3']}: SKIPPED (already complete)")
            else:
                print(f"  {r['date_v3']}: {r['episodes']} ep, {r['total_frames']} frames, "
                      f"cut median={r['cut_median']}, dropped {r['dropped_pct']:.1f}%")
        return

    if not args.mode:
        sys.exit("must pass --mode {raw,no_release} (optionally with --merge-dates) or --per-date DATE... (v3)")

    trim = (args.mode == "no_release")
    if args.merge_dates:
        if not args.merge_out:
            sys.exit("--merge-dates requires --merge-out NAME")
        merge_dates = args.merge_dates
        src_root = {"v3": V3_ROOT, "v3.2": V3_2_ROOT}.get(args.merge_src, VIS_BASE)
        dst = DST_ROOT / args.merge_out
        print(f"merge: {len(merge_dates)} dates from vis_base/{args.merge_src} → self_built/{args.merge_out} "
              f"(trim={trim})", flush=True)
    else:
        merge_dates = DATES
        src_root = VIS_BASE
        dst = DST_ROOT / ("A_0522_0526_no_release" if trim else "A_0522_0526_raw")
    # v3 per-date dirs store videos under feature-key dirs (observation.images.*); v2 uses bare cam names (CAM_DIRS).
    src_feat_dirs = bool(args.merge_dates and args.merge_src in ("v3", "v3.2"))

    if not args.dry_run:
        if dst.exists():
            sys.exit(f"dst already exists: {dst} (delete first)")
        (dst / "data" / "chunk-000").mkdir(parents=True)
        (dst / "meta").mkdir()
        for cam in CAMERAS:
            # dst dir uses the FULL feature key (lerobot video_path {video_key}=observation.images.*),
            # NOT the bare cam name. Source (vis_base) uses bare names via CAM_DIRS.
            (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    episodes_out, stats_out = [], []
    new_idx = 0
    total_frames = 0
    cut_report = []
    video_jobs = []  # (src_mp4, dst_mp4, cut, new_len) for parallel trim

    for date in merge_dates:
        src = src_root / date
        src_eps = {(json.loads(l).get("episode_id", json.loads(l).get("episode_index"))): json.loads(l)
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
                    sv = src / "videos" / "chunk-000" / (cam if src_feat_dirs else CAM_DIRS[cam]) / f"episode_{old_id:06d}.mp4"
                    dv = dst / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"  # feature-key dir
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
        nproc = BUILD_WORKERS  # 4 enc-threads each
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
    info["chunks_size"] = max(1000, new_idx)   # single chunk-000 → chunks_size ≥ N (else ep//1000 → chunk-001 → assert fail)
    info["splits"] = {"train": f"0:{new_idx}"}
    # drop depth feature + depth_path (not carried)
    info["features"].pop("observation.depth.top_head", None)
    info.pop("depth_path", None)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"done → {dst}")
    _maybe_norm_stats(dst, not args.no_norm_stats, args.action_dim)
    print("  next: register config (norm_stats already (re)computed above from the built dataset)")


if __name__ == "__main__":
    main()
