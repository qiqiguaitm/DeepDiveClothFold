#!/usr/bin/env python3
"""
space_mirror.py

Space Mirror core functionality: dual-arm data mirroring and data augmentation
- Swap left/right arm data (parquet, json, jsonl)
- Flip videos (horizontal mirroring)
- Merge original and mirrored datasets
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Import merge function from merge_lerobot
# Add current directory to path to import merge_lerobot from same directory
_utils_dir = Path(__file__).parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))
try:
    from merge_lerobot import merge_repos
    MERGE_AVAILABLE = True
except (ImportError, RuntimeError):
    # ImportError: file missing. RuntimeError: lerobot pkg not installed (its
    # internal guard re-raises). Either way merge is unavailable; create-mirror
    # still works.
    MERGE_AVAILABLE = False
    merge_repos = None  # type: ignore


# ==================== Core Utility Functions ====================

# Per-joint sign pattern for body-mirror across the sagittal (YZ) plane on the
# Agilex Piper 6-DOF + gripper chain (yaw-pitch-pitch-yaw-pitch-roll + grip).
# Joints whose axis flips under mirror (yaw j0, wrist yaw j3, wrist roll j5)
# need their sign negated; pitches (j1/j2/j4) keep sign; gripper (j6) is a
# scalar opening with no rotational sign meaning.
#
# A pure left↔right swap WITHOUT this sign flip is the bug we hit in the first
# round of mirror tests: the rerun joints and the mirrored mp4 visually
# disagreed because the recorded poses, after swap, weren't mirror-of-original
# — they were "right-arm joint values executed by the left arm", a different
# configuration in body frame.
PIPER_JOINT_MIRROR_SIGN = (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def _resolve_sign(sign_pattern, ndim: int) -> np.ndarray:
    """Normalize sign_pattern arg → length-ndim ndarray of ±1.

    sign_pattern:
      None                       → use PIPER_JOINT_MIRROR_SIGN (default)
      False / 'none' / 'noflip'  → all-ones (legacy pure-swap behavior)
      tuple/list/np.ndarray      → custom per-joint sign vector
    """
    if sign_pattern is None:
        sp = PIPER_JOINT_MIRROR_SIGN
    elif sign_pattern is False or (isinstance(sign_pattern, str)
                                   and sign_pattern.lower() in ("none", "noflip", "off")):
        return np.ones(ndim, dtype=np.float64)
    else:
        sp = sign_pattern
    arr = np.asarray(sp, dtype=np.float64)
    if arr.shape == (ndim,):
        return arr
    raise ValueError(f"sign_pattern must be length {ndim}, got shape {arr.shape}")


def swap_arms_in_array(arr: np.ndarray, left_dim: int = 7, right_dim: int = 7,
                        sign_pattern=None) -> np.ndarray:
    """Mirror a [left(left_dim) | right(right_dim)] joint vector across the
    body's sagittal plane: swap L↔R AND negate yaw/roll dims so the resulting
    pose is the geometric mirror, not just relocated joint values.

    sign_pattern: see _resolve_sign(). Default = PIPER_JOINT_MIRROR_SIGN.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.ndim == 0:
        return arr

    arr_flat = arr.flatten()
    total_dim = left_dim + right_dim

    if len(arr_flat) != total_dim:
        raise ValueError(
            f"Array dimension mismatch: expected {total_dim} dims (left{left_dim} + right{right_dim}), "
            f"got {len(arr_flat)} dims"
        )

    sign_l = _resolve_sign(sign_pattern, left_dim)
    sign_r = _resolve_sign(sign_pattern, right_dim) if right_dim != left_dim else sign_l

    left_arm = arr_flat[:left_dim] * sign_l
    right_arm = arr_flat[left_dim:left_dim + right_dim] * sign_r
    swapped = np.concatenate([right_arm, left_arm])

    if arr.ndim > 1:
        swapped = swapped.reshape(arr.shape)

    return swapped


def swap_array_dims_list(arr: List[float], left_dim: int = 7, right_dim: int = 7,
                         keep_padding: bool = True, sign_pattern=None) -> List[float]:
    """Same as swap_arms_in_array but for plain Python lists (JSON/JSONL).
    Default applies PIPER_JOINT_MIRROR_SIGN; pass sign_pattern=False for legacy
    pure-swap (use only for stats fields where sign doesn't apply, e.g. std)."""
    if not isinstance(arr, list):
        arr = list(arr)

    total_dim = left_dim + right_dim

    if len(arr) < total_dim:
        raise ValueError(
            f"Insufficient array dimensions: expected at least {total_dim} dims (left{left_dim} + right{right_dim}), "
            f"got {len(arr)} dims"
        )

    sign_l = _resolve_sign(sign_pattern, left_dim)
    sign_r = _resolve_sign(sign_pattern, right_dim) if right_dim != left_dim else sign_l

    left_arm = [a * float(s) for a, s in zip(arr[:left_dim], sign_l)]
    right_arm = [a * float(s) for a, s in zip(arr[left_dim:left_dim + right_dim], sign_r)]
    swapped = right_arm + left_arm

    if keep_padding and len(arr) > total_dim:
        padding = arr[total_dim:]
        swapped = swapped + padding

    return swapped


_EP_NAME_RE = __import__("re").compile(r"^episode_(\d+)\.")


def _ep_id_from_parquet_name(name: str) -> Optional[int]:
    """`episode_000042.parquet` → 42; non-matching → None."""
    m = _EP_NAME_RE.match(name)
    return int(m.group(1)) if m else None


# ==================== Parquet Processing ====================

def swap_arms_in_parquet(
    input_path: Path,
    output_path: Path,
    columns: Optional[List[str]] = None,
    left_dim: int = 7,
    right_dim: int = 7,
    sign_pattern=None,
) -> Tuple[str, bool, str]:
    """Process a single parquet file: swap left↔right halves AND apply joint
    sign mirror (default Piper) for action/state columns. Pass sign_pattern=False
    for legacy pure-swap (only valid if you know your URDF doesn't need it)."""
    try:
        df = pd.read_parquet(str(input_path))
        
        if columns is None:
            columns_to_process = []
            for col in ['observation.state', 'action']:
                if col in df.columns:
                    columns_to_process.append(col)
        else:
            columns_to_process = [col for col in columns if col in df.columns]
        
        if not columns_to_process:
            return (str(input_path), False, "No columns found to process")
        
        for col in columns_to_process:
            if col not in df.columns:
                continue
            
            if df[col].dtype != object:
                return (
                    str(input_path),
                    False,
                    f"Column {col} is not object type (not nested array), skipping"
                )
            
            swapped_values = []
            for idx, val in enumerate(df[col]):
                try:
                    if isinstance(val, (list, tuple)):
                        arr = np.array(val)
                    elif isinstance(val, np.ndarray):
                        arr = val.copy()
                    else:
                        return (
                            str(input_path),
                            False,
                            f"Unsupported data type for column {col} row {idx}: {type(val)}"
                        )
                    
                    swapped_arr = swap_arms_in_array(arr, left_dim, right_dim, sign_pattern)
                    swapped_values.append(swapped_arr)
                
                except Exception as e:
                    return (
                        str(input_path),
                        False,
                        f"Error processing column {col} row {idx}: {str(e)}"
                    )
            
            df[col] = swapped_values
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(output_path), index=False)
        
        return (
            str(input_path),
            True,
            f"Successfully processed {len(columns_to_process)} columns: {', '.join(columns_to_process)}"
        )
    
    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


def process_parquet_files(
    input_dir: Path,
    output_dir: Path,
    columns: Optional[List[str]] = None,
    left_dim: int = 7,
    right_dim: int = 7,
    num_workers: int = 4,
    episode_filter: Optional[List[int]] = None,
    sign_pattern=None,
) -> int:
    """Batch process parquet files. Returns # of episodes processed.

    episode_filter: optional list of episode ids to keep (e.g. [0,1,2]); None = all.
    """
    parquet_files = list(input_dir.rglob('*.parquet'))
    if episode_filter is not None:
        keep = set(int(e) for e in episode_filter)
        parquet_files = [f for f in parquet_files
                         if _ep_id_from_parquet_name(f.name) in keep]
    
    if not parquet_files:
        print(f"Warning: No parquet files found in {input_dir}")
        return 0

    print(f"Found {len(parquet_files)} parquet files")

    def get_output_path(input_file: Path) -> Path:
        relative = input_file.relative_to(input_dir)
        return output_dir / relative

    tasks = [(f, get_output_path(f)) for f in parquet_files]

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(swap_arms_in_parquet, inp, out, columns, left_dim, right_dim, sign_pattern): inp
            for inp, out in tasks
        }

        for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="Processing parquet"):
            input_path = future_to_file[future]
            try:
                result_path, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"✗ [{result_path}] {message}")
                    fail_count += 1
            except Exception as e:
                print(f"✗ [{input_path}] Processing exception: {str(e)}")
                fail_count += 1

    print(f"Parquet processing complete: {success_count} succeeded, {fail_count} failed")
    return success_count


# ==================== JSON Processing ====================

def process_norm_stats_json(
    input_path: Path,
    output_path: Path,
    left_dim: int = 7,
    right_dim: int = 7,
    sign_pattern=None,
) -> Tuple[str, bool, str]:
    """Process norm_stats.json file.

    Apply sign-flip to `mean` (mean of flipped data = sign-flipped mean), and
    legacy pure-swap to std/q01/q99 — those would need careful handling under
    sign flip (std stays positive, q01↔-q99 swap, etc.). For correctness,
    re-run compute_norm_stats on the mirrored dataset rather than relying on
    these transformed values."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "norm_stats" not in data:
            return (str(input_path), False, "Field 'norm_stats' not found in JSON file")

        norm_stats = data["norm_stats"]

        # mean: gets full sign-flip+swap; std/q01/q99: pure swap (sign_pattern=False)
        STAT_SIGN = {"mean": sign_pattern, "std": False, "q01": False, "q99": False}
        for key in ["state", "actions"]:
            if key not in norm_stats:
                continue
            stat_item = norm_stats[key]
            for stat_key, sp in STAT_SIGN.items():
                if stat_key in stat_item:
                    try:
                        stat_item[stat_key] = swap_array_dims_list(
                            stat_item[stat_key],
                            left_dim,
                            right_dim,
                            keep_padding=True,
                            sign_pattern=sp,
                        )
                    except Exception as e:
                        print(f"Warning: Error processing norm_stats.{key}.{stat_key}: {e}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return (str(input_path), True, "Processing successful (recommend recompute_norm_stats afterwards)")

    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


# ==================== JSONL Processing ====================

def swap_stats_dims(
    stats_dict: Dict[str, Any],
    left_dim: int = 7,
    right_dim: int = 7,
    sign_pattern=None,
) -> Dict[str, Any]:
    """Swap left/right arm data in stats dictionary. Sign-flip is applied to
    `mean` only; std/min/max are pure-swapped (see process_norm_stats_json
    docstring for why)."""
    # Swap hand_left and hand_right
    if "observation.images.hand_left" in stats_dict and "observation.images.hand_right" in stats_dict:
        hand_left = stats_dict["observation.images.hand_left"]
        hand_right = stats_dict["observation.images.hand_right"]
        stats_dict["observation.images.hand_left"] = hand_right
        stats_dict["observation.images.hand_right"] = hand_left

    STAT_SIGN = {"mean": sign_pattern, "std": False, "min": False, "max": False}
    for key in ["observation.state", "action"]:
        if key not in stats_dict:
            continue
        stat_item = stats_dict[key]
        for stat_key, sp in STAT_SIGN.items():
            if stat_key in stat_item:
                try:
                    stat_item[stat_key] = swap_array_dims_list(
                        stat_item[stat_key], left_dim, right_dim, sign_pattern=sp,
                    )
                except Exception as e:
                    print(f"Warning: Error processing {key}.{stat_key}: {e}")

    return stats_dict


def process_episodes_stats_jsonl(
    input_path: Path,
    output_path: Path,
    left_dim: int = 7,
    right_dim: int = 7,
    sign_pattern=None,
) -> Tuple[str, bool, str]:
    """Process episodes_stats.jsonl file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        processed_lines = []
        processed_count = 0
        error_count = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue

            try:
                data = json.loads(line)

                if "stats" in data and isinstance(data["stats"], dict):
                    data["stats"] = swap_stats_dims(data["stats"], left_dim, right_dim, sign_pattern)
                    processed_count += 1
                
                processed_lines.append(json.dumps(data, ensure_ascii=False))
            
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Error: JSON parsing failed at line {line_num}: {e}")
                processed_lines.append(line)
            except Exception as e:
                error_count += 1
                print(f"Error: Processing failed at line {line_num}: {e}")
                processed_lines.append(line)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
            if processed_lines and processed_lines[-1]:
                f.write('\n')
        
        message = f"Successfully processed {processed_count} entries"
        if error_count > 0:
            message += f", {error_count} errors"
        
        return (str(input_path), True, message)
    
    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


# ==================== Video Processing ====================

def flip_video(input_path: str, output_path: str) -> Tuple[str, bool, str]:
    """Flip a single video file (horizontal mirroring)"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return (input_path, False, f"Unable to open video file: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return (input_path, False, f"Unable to create output video file: {output_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            flipped_frame = cv2.flip(frame, 1)
            out.write(flipped_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return (input_path, True, f"Successfully processed {frame_count} frames")
    
    except Exception as e:
        return (input_path, False, f"Error: {str(e)}")


def process_videos(
    input_dir: Path,
    output_dir: Path,
    num_workers: int = 4,
) -> None:
    """Batch process video files"""
    video_files = list(input_dir.rglob('*.mp4'))
    
    if not video_files:
        print(f"Warning: No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    def get_output_path(input_file: Path) -> Path:
        relative = input_file.relative_to(input_dir)
        return output_dir / relative
    
    tasks = [(str(f), str(get_output_path(f))) for f in video_files]
    
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {
            executor.submit(flip_video, inp, out): inp
            for inp, out in tasks
        }
        
        for future in tqdm(as_completed(future_to_video), total=len(tasks), desc="Processing videos"):
            input_path = future_to_video[future]
            try:
                result_path, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"✗ [{result_path}] {message}")
                    fail_count += 1
            except Exception as e:
                print(f"✗ [{input_path}] Processing exception: {str(e)}")
                fail_count += 1
    
    print(f"Video processing complete: {success_count} succeeded, {fail_count} failed")


# ==================== Dataset Merging ====================

def merge_lerobot_datasets(
    src_paths: List[str],
    tgt_path: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "agilex",
    features: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> None:
    """Merge multiple LeRobot datasets by calling merge_lerobot.merge_repos"""
    if not MERGE_AVAILABLE:
        raise RuntimeError("merge_lerobot module not available. Cannot perform merge operation.")
    
    merge_repos(
        src_paths=src_paths,
        tgt_path=tgt_path,
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        force=force
    )


# ==================== Video / Depth helpers (per-camera) ====================

# Two naming conventions are used in the wild for video sub-dirs:
#   bare name      → top_head / hand_left / hand_right            (kai0 collected data)
#   prefixed name  → observation.images.top_head / .hand_left / …  (some HF mirrors)
# Match either when sourcing.
def _find_cam_dir(parent: Path, cam_name: str) -> Optional[Path]:
    for candidate in (cam_name, f"observation.images.{cam_name}"):
        p = parent / candidate
        if p.is_dir():
            return p
    return None


def _pack_zarr_dir_to_zip(zarr_dir: Path) -> Path:
    """Pack a `.zarr/` dir into a sibling `.zarr.zip` (ZIP_STORED, contents at
    root), removing the dir. Mirrors data_manager depth_archive.pack_zarr_dir."""
    import os
    import shutil
    import zipfile
    zp = Path(str(zarr_dir) + ".zip")
    tmp = Path(str(zp) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(zarr_dir):
            for name in files:
                fp = Path(root) / name
                zf.write(fp, fp.relative_to(zarr_dir).as_posix())
    os.replace(tmp, zp)
    shutil.rmtree(zarr_dir, ignore_errors=True)
    return zp


def flip_depth_zarr(src_zarr: Path, dst_zarr: Path) -> None:
    """Horizontal-flip every frame of a uint16 depth zarr [T, H, W].
    Handles both packed `.zarr.zip` and legacy `.zarr/` dir; the output mirrors
    the input format (zip in → zip out)."""
    import shutil
    import tempfile
    import zipfile

    import zarr  # lazy
    is_zip = src_zarr.suffix == ".zip"
    src_tmp = None
    if is_zip:
        src_tmp = tempfile.mkdtemp(prefix="kai0_depthz_src_")
        with zipfile.ZipFile(src_zarr) as zf:
            zf.extractall(src_tmp)
        z_in = zarr.open(src_tmp, mode="r")
    else:
        z_in = zarr.open(str(src_zarr), mode="r")
    try:
        # write flipped frames to a `.zarr/` dir (strip trailing ".zip" if any)
        out_dir = dst_zarr.with_name(dst_zarr.name[:-4]) if is_zip else dst_zarr
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        z_out = zarr.open(str(out_dir), mode="w",
                          shape=z_in.shape, chunks=z_in.chunks, dtype=z_in.dtype)
        for i in range(int(z_in.shape[0])):
            z_out[i] = np.ascontiguousarray(z_in[i][:, ::-1])
        if is_zip:
            _pack_zarr_dir_to_zip(out_dir)
    finally:
        if src_tmp is not None:
            shutil.rmtree(src_tmp, ignore_errors=True)


def _flip_videos_for_episodes(src_dir: Path, dst_dir: Path,
                              episode_filter: Optional[List[int]],
                              num_workers: int) -> int:
    """Flip mp4s under src_dir/episode_NNNNNN.mp4 → dst_dir/. Returns count."""
    if not src_dir.is_dir():
        return 0
    files = sorted(src_dir.glob("*.mp4"))
    if episode_filter is not None:
        keep = set(int(e) for e in episode_filter)
        files = [f for f in files if _ep_id_from_parquet_name(f.name) in keep]
    tasks = [(str(f), str(dst_dir / f.name)) for f in files]
    if not tasks:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(flip_video, inp, out): inp for inp, out in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc=f"  → {dst_dir.name}"):
            _, ok, msg = fut.result()
            if ok:
                success += 1
            else:
                print(f"  ✗ {futs[fut]}: {msg}")
    return success


def _flip_zarrs_for_episodes(src_dir: Path, dst_dir: Path,
                             episode_filter: Optional[List[int]]) -> int:
    """Flip depth zarr dirs under src_dir/episode_NNNNNN.zarr/ → dst_dir/.
    Sequential (zarr writes are I/O bound but mostly fine without pool); skip
    silently if src_dir absent."""
    if not src_dir.is_dir():
        return 0
    zdirs = sorted([p for p in src_dir.iterdir()
                    if (p.is_dir() and p.suffix == ".zarr")
                    or (p.is_file() and p.name.endswith(".zarr.zip"))])
    if episode_filter is not None:
        keep = set(int(e) for e in episode_filter)
        zdirs = [p for p in zdirs if _ep_id_from_parquet_name(p.name) in keep]
    success = 0
    for src_z in tqdm(zdirs, desc=f"  → {dst_dir.name}"):
        try:
            flip_depth_zarr(src_z, dst_dir / src_z.name)
            success += 1
        except Exception as e:
            print(f"  ✗ {src_z}: {e}")
    return success


# ==================== Meta helpers ====================

def _filter_meta_jsonl(src_path: Path, dst_path: Path,
                       episode_filter: Optional[List[int]]) -> int:
    """Copy `episodes.jsonl`, keeping only entries whose episode id ∈ filter
    (or all if None). Accepts either `episode_index` (LeRobot v2) or
    `episode_id` (kai0 collected) as the id field. Returns kept count."""
    if not src_path.is_file():
        return 0
    keep = None if episode_filter is None else set(int(e) for e in episode_filter)
    out_lines = []
    with open(src_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
                ep_id = obj.get("episode_index", obj.get("episode_id"))
                if keep is None or ep_id in keep:
                    out_lines.append(line)
            except json.JSONDecodeError:
                out_lines.append(line)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")
    return len(out_lines)


def _patch_info_json(src_path: Path, dst_path: Path,
                     n_episodes: int, n_frames: int, n_cams: int) -> None:
    """Read src info.json, update totals + splits, write to dst."""
    if not src_path.is_file():
        return
    with open(src_path) as f:
        info = json.load(f)
    info["total_episodes"] = int(n_episodes)
    info["total_frames"] = int(n_frames)
    info["total_videos"] = int(n_episodes * n_cams)
    info["splits"] = {"train": f"0:{int(n_episodes)}"}
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ==================== Main Functions ====================

# Camera roles whose RGB+depth are mirrored. top_head: in-place flip;
# hand_left ↔ hand_right: swap+flip.
CAMS_FLIP_INPLACE = ("top_head",)
CAMS_SWAP_PAIR = ("hand_left", "hand_right")


def create_mirror_dataset(
    src_path: str,
    tgt_path: str,
    left_dim: int = 7,
    right_dim: int = 7,
    num_workers: int = 4,
    sign_pattern=None,
    episode_filter: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Create mirrored dataset.

    sign_pattern: per-joint sign vector for body-mirror; default Piper
                  [-1,1,1,-1,1,-1,1] (j0/j3/j5 negate, others keep).
                  Pass False for legacy pure-swap.
    episode_filter: optional list of episode_index to keep (None = all).

    Returns dict with summary {episodes, total_frames, src, dst}.
    """
    src_root = Path(src_path).expanduser().resolve()
    tgt_root = Path(tgt_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_root}")
    print("=" * 60)
    print("Space Mirror — create mirrored dataset")
    print("=" * 60)
    print(f"  source       : {src_root}")
    print(f"  target       : {tgt_root}")
    print(f"  episode_filter: {episode_filter if episode_filter is not None else 'all'}")
    eff_sign = _resolve_sign(sign_pattern, left_dim).tolist()
    print(f"  joint sign   : {eff_sign}  (PIPER default = {list(PIPER_JOINT_MIRROR_SIGN)})")
    print()

    # ── 1. norm_stats.json (sign-flipped mean, plain swap for std/q01/q99) ──
    print("[1/5] norm_stats.json …")
    src_norm = src_root / "norm_stats.json"
    if src_norm.exists():
        _, ok, msg = process_norm_stats_json(
            src_norm, tgt_root / "norm_stats.json",
            left_dim, right_dim, sign_pattern)
        print("  " + ("✓ " + msg if ok else "✗ " + msg))
    else:
        print("  (no norm_stats.json — skip)")

    # ── 2. episodes_stats.jsonl ──
    print("[2/5] meta/episodes_stats.jsonl …")
    src_es = src_root / "meta" / "episodes_stats.jsonl"
    if src_es.exists():
        _, ok, msg = process_episodes_stats_jsonl(
            src_es, tgt_root / "meta" / "episodes_stats.jsonl",
            left_dim, right_dim, sign_pattern)
        print("  " + ("✓ " + msg if ok else "✗ " + msg))
    else:
        print("  (no episodes_stats.jsonl — skip)")

    # ── 3. parquet (action + state, sign + swap, optional ep filter) ──
    print("[3/5] data/chunk-*/episode_*.parquet …")
    n_eps = process_parquet_files(
        src_root / "data", tgt_root / "data", None,
        left_dim, right_dim, num_workers,
        episode_filter=episode_filter, sign_pattern=sign_pattern)

    # ── 4. videos: top_head flip + hand_left⇄hand_right swap+flip; same for *_depth ──
    print("[4/5] videos (RGB + depth zarr) …")
    src_videos = src_root / "videos"
    n_cams_present = 0
    if src_videos.is_dir():
        chunks = sorted([d for d in src_videos.iterdir()
                         if d.is_dir() and d.name.startswith("chunk-")])
        for chunk in chunks:
            tgt_chunk = tgt_root / "videos" / chunk.name
            # top_head (RGB + depth): in-place flip
            for cam in CAMS_FLIP_INPLACE:
                src_rgb = _find_cam_dir(chunk, cam)
                if src_rgb is not None:
                    _flip_videos_for_episodes(src_rgb, tgt_chunk / cam,
                                              episode_filter, num_workers)
                    n_cams_present += 1
                src_dep = _find_cam_dir(chunk, f"{cam}_depth")
                if src_dep is not None:
                    _flip_zarrs_for_episodes(src_dep, tgt_chunk / f"{cam}_depth",
                                             episode_filter)
            # hand_left ↔ hand_right (RGB + depth): swap+flip
            cam_l, cam_r = CAMS_SWAP_PAIR
            sl_rgb, sr_rgb = _find_cam_dir(chunk, cam_l), _find_cam_dir(chunk, cam_r)
            if sl_rgb is not None and sr_rgb is not None:
                _flip_videos_for_episodes(sr_rgb, tgt_chunk / cam_l,
                                          episode_filter, num_workers)
                _flip_videos_for_episodes(sl_rgb, tgt_chunk / cam_r,
                                          episode_filter, num_workers)
                n_cams_present += 2
            sl_dep = _find_cam_dir(chunk, f"{cam_l}_depth")
            sr_dep = _find_cam_dir(chunk, f"{cam_r}_depth")
            if sl_dep is not None and sr_dep is not None:
                _flip_zarrs_for_episodes(sr_dep, tgt_chunk / f"{cam_l}_depth",
                                         episode_filter)
                _flip_zarrs_for_episodes(sl_dep, tgt_chunk / f"{cam_r}_depth",
                                         episode_filter)

    # ── 5. meta files: episodes.jsonl filtered, info.json patched, tasks copied ──
    print("[5/5] meta files (episodes.jsonl / info.json / tasks.jsonl) …")
    src_meta = src_root / "meta"
    tgt_meta = tgt_root / "meta"
    tgt_meta.mkdir(parents=True, exist_ok=True)
    kept = _filter_meta_jsonl(src_meta / "episodes.jsonl",
                              tgt_meta / "episodes.jsonl", episode_filter)
    # tasks.jsonl + relabel_meta.json are episode-agnostic — copy as is
    for fname in ("tasks.jsonl", "relabel_meta.json"):
        s = src_meta / fname
        if s.is_file():
            shutil.copy2(s, tgt_meta / fname)
    # frame totals: re-read mirrored parquets
    total_frames = 0
    for pq in (tgt_root / "data").rglob("*.parquet"):
        try:
            import pyarrow.parquet as _pq
            total_frames += _pq.read_metadata(str(pq)).num_rows
        except Exception:
            pass
    n_cams_per_episode = max(1, n_cams_present // max(1, len(chunks) if src_videos.is_dir() else 1))
    _patch_info_json(src_meta / "info.json", tgt_meta / "info.json",
                     n_episodes=kept if episode_filter else n_eps,
                     n_frames=total_frames,
                     n_cams=n_cams_per_episode)

    summary = {
        "source": str(src_root), "target": str(tgt_root),
        "episodes": kept if episode_filter else n_eps,
        "total_frames": total_frames,
        "joint_sign": eff_sign,
    }
    print()
    print(f"✓ done — {summary['episodes']} episodes, {summary['total_frames']} frames")
    print(f"  consider re-running compute_norm_stats on {tgt_root} for proper")
    print(f"  normalization (current norm_stats.json has only sign-flipped mean).")
    print("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Space Mirror: Dual-arm data mirroring and data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create mirrored dataset
  python space_mirror.py create-mirror --src-path /path/to/original --tgt-path /path/to/mirror
  
  # Merge original and mirrored datasets
  python space_mirror.py merge --src-paths /path/to/original /path/to/mirror --tgt-path /path/to/merged --repo-id my_dataset
  
  # Full pipeline (create mirror and merge)
  python space_mirror.py full --src-path /path/to/original --mirror-path /path/to/mirror --merge-path /path/to/merged --repo-id my_dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # create-mirror command
    parser_create = subparsers.add_parser('create-mirror', help='Create mirrored dataset')
    parser_create.add_argument('--src-path', required=True, help='Source dataset path')
    parser_create.add_argument('--tgt-path', required=True, help='Target mirrored dataset path')
    parser_create.add_argument('--left-dim', type=int, default=7, help='Left arm dimension (default: 7)')
    parser_create.add_argument('--right-dim', type=int, default=7, help='Right arm dimension (default: 7)')
    parser_create.add_argument('--num-workers', type=int, default=4, help='Number of parallel worker processes (default: 4)')
    parser_create.add_argument('--sign-pattern', type=str, default=None,
        help=("comma-separated 7 floats overriding the per-joint sign vector "
              "(default = Piper [-1,1,1,-1,1,-1,1]). "
              "Use 'noflip' for legacy pure-swap (only correct if URDF doesn't need it)."))
    parser_create.add_argument('--episodes', type=str, default=None,
        help=("optional episode subset, e.g. '0-4' or '0,2,5' — useful for "
              "quick verification before mirroring a full dataset."))
    
    # merge command
    parser_merge = subparsers.add_parser('merge', help='Merge datasets')
    parser_merge.add_argument('--src-paths', nargs='+', required=True, help='Source dataset paths list')
    parser_merge.add_argument('--tgt-path', required=True, help='Target merged dataset path')
    parser_merge.add_argument('--repo-id', required=True, help='Dataset repo_id')
    parser_merge.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    parser_merge.add_argument('--robot-type', type=str, default='agilex', help='Robot type (default: agilex)')
    parser_merge.add_argument('--features-json', type=str, default=None, help='Path to features.json file')
    parser_merge.add_argument('--force', action='store_true', help='Force merge (ignore conflicts)')
    
    # full command
    parser_full = subparsers.add_parser('full', help='Full pipeline: create mirror and merge')
    parser_full.add_argument('--src-path', required=True, help='Source dataset path')
    parser_full.add_argument('--mirror-path', required=True, help='Mirrored dataset path')
    parser_full.add_argument('--merge-path', required=True, help='Merged dataset path')
    parser_full.add_argument('--repo-id', required=True, help='Dataset repo_id')
    parser_full.add_argument('--left-dim', type=int, default=7, help='Left arm dimension (default: 7)')
    parser_full.add_argument('--right-dim', type=int, default=7, help='Right arm dimension (default: 7)')
    parser_full.add_argument('--num-workers', type=int, default=4, help='Number of parallel worker processes (default: 4)')
    parser_full.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    parser_full.add_argument('--robot-type', type=str, default='agilex', help='Robot type (default: agilex)')
    parser_full.add_argument('--features-json', type=str, default=None, help='Path to features.json file')
    parser_full.add_argument('--force', action='store_true', help='Force merge (ignore conflicts)')
    parser_full.add_argument('--sign-pattern', type=str, default=None,
        help='see create-mirror --sign-pattern')
    parser_full.add_argument('--episodes', type=str, default=None,
        help='see create-mirror --episodes')
    
    args = parser.parse_args()

    def _parse_sign(s):
        if s is None:
            return None
        if s.lower() in ("noflip", "none", "off", "false"):
            return False
        return [float(x) for x in s.split(",")]

    def _parse_episodes(s):
        if s is None:
            return None
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return sorted(set(out))

    if args.command == 'create-mirror':
        create_mirror_dataset(
            args.src_path,
            args.tgt_path,
            args.left_dim,
            args.right_dim,
            args.num_workers,
            sign_pattern=_parse_sign(args.sign_pattern),
            episode_filter=_parse_episodes(args.episodes),
        )
    
    elif args.command == 'merge':
        if not MERGE_AVAILABLE:
            print("Error: merge_lerobot module not available. Please ensure merge_lerobot.py is accessible.")
            sys.exit(1)
        
        features = None
        if args.features_json:
            with open(args.features_json, 'r', encoding='utf-8') as f:
                features = json.load(f)
        
        merge_lerobot_datasets(
            args.src_paths,
            args.tgt_path,
            args.repo_id,
            args.fps,
            args.robot_type,
            features,
            args.force
        )
    
    elif args.command == 'full':
        print("=" * 60)
        print("Space Mirror Full Pipeline")
        print("=" * 60)
        print()
        
        # Step 1: Create mirrored dataset
        print("Step 1/2: Creating mirrored dataset")
        create_mirror_dataset(
            args.src_path,
            args.mirror_path,
            args.left_dim,
            args.right_dim,
            args.num_workers,
            sign_pattern=_parse_sign(args.sign_pattern),
            episode_filter=_parse_episodes(args.episodes),
        )
        print()
        
        # Step 2: Merge datasets
        print("Step 2/2: Merging datasets")
        if not MERGE_AVAILABLE:
            print("Error: merge_lerobot module not available. Please ensure merge_lerobot.py is accessible.")
            sys.exit(1)
        
        features = None
        if args.features_json:
            with open(args.features_json, 'r', encoding='utf-8') as f:
                features = json.load(f)
        
        merge_lerobot_datasets(
            [args.src_path, args.mirror_path],
            args.merge_path,
            args.repo_id,
            args.fps,
            args.robot_type,
            features,
            args.force
        )
        print()
        print("=" * 60)
        print("✓ All processing complete!")
        print("=" * 60)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

