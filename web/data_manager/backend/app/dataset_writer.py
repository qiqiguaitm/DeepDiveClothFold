"""LeRobot v2.1 episode writer + meta-update helpers.

Extracted from recorder.py so both the teleop backend (FastAPI-driven) and the
autonomy ROS2 recorder node can produce binary-identical dataset bytes.

NO FastAPI / ros_bridge / pydantic dependencies — safe to import from any venv
that has av + pyarrow + zarr.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import time
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import zarr
    _HAS_ZARR = True
    _ZARR_MAJOR = int(zarr.__version__.split(".")[0])
except ImportError:
    _HAS_ZARR = False
    _ZARR_MAJOR = 0

try:
    import numcodecs
    _HAS_NUMCODECS = True
except ImportError:
    _HAS_NUMCODECS = False

from .layout import compound_to_subset_root, new_task_subset_root, today_compound


def _open_depth_zarr(path, h, w):
    """Open a uint16 depth zarr array shape (0, h, w) chunks (1, h, w) with
    blosc/zstd/bitshuffle. Returns a zarr.Array that supports append-via-resize.

    Forces on-disk format zarr_format=2 so depth files are interchangeable
    with teleop's data_manager backend (which is pinned to zarr 2.x).
    """
    if _ZARR_MAJOR >= 3:
        # numcodecs.Blosc is the underlying compressor, works across versions.
        # `compressors=` (plural, list) is the zarr 3 signature.
        comp = numcodecs.Blosc(cname="zstd", clevel=3,
                               shuffle=numcodecs.Blosc.BITSHUFFLE) if _HAS_NUMCODECS else None
        return zarr.create_array(
            store=str(path), shape=(0, h, w), chunks=(1, h, w),
            dtype="uint16", zarr_format=2, compressors=comp, overwrite=True,
        )
    # zarr 2.x — original API.
    try:
        comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    except Exception:
        comp = None
    return zarr.open(str(path), mode="w",
                     shape=(0, h, w), chunks=(1, h, w),
                     dtype="uint16", compressor=comp)


def _append_depth_frame(z, frame):
    """Append one (h, w) uint16 frame to a depth zarr array.

    Hides the zarr 2 vs 3 API split: 2.x has `.append()`; 3.x removed it,
    use `.resize()` + slice assignment instead.
    """
    if _ZARR_MAJOR >= 3:
        t = z.shape[0]
        z.resize((t + 1, z.shape[1], z.shape[2]))
        z[t] = frame
    else:
        z.append(frame[None, :, :])


def _load_depth_flags() -> tuple[str, ...]:
    """Read config/camera_depth_flags.py by probing upward from this file.

    The data_manager backend isn't a child of /config/, so a plain relative
    import won't reach it. Falls back to () if the macro file is missing.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "camera_depth_flags.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "kai0_camera_depth_flags", candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return tuple(mod.DEPTH_CAMERAS)
    return ()


CAMERAS = ("top_head", "hand_left", "hand_right")
DEPTH_CAMERAS = _load_depth_flags()
FPS = 30
WIDTH = 640
HEIGHT = 480

# ── V3 online front-trim (leading-idle trim at record time) ──
# Same semantics/constants as train_scripts/kai/data/build_no_release.py
# (motion_onset + cut = max(0, onset - MARGIN)). Lets the collection pipeline
# emit V3 datasets directly instead of a post-hoc build_no_release pass.
TRIM_ARM_DIMS = list(range(0, 6)) + list(range(7, 13))  # 12 arm dims (exclude grippers 6,13)
TRIM_THR = 3e-3   # rad/frame: sustained mean |Δaction| over arm dims => "moving"
TRIM_WIN = 10     # frames of sustained motion to call it the onset
TRIM_MARGIN = 15  # keep this many frames before onset (lead-in; NOT a full delete)

log = logging.getLogger(__name__)


def task_subset_root(task: str, subset: str) -> Path:
    """`<DATA_ROOT>/<task>/<subset>/<today-v2>` for new episodes (v2 layout)."""
    return new_task_subset_root(task, subset)


def pick_codec() -> tuple[str, str, dict]:
    """Pick video codec — h264 (default, broad compatibility) or av1 (compact).

    KAI0_VIDEO_CODEC=av1 selects libsvtav1 → libaom-av1 → falls back to h264.
    """
    choice = os.environ.get("KAI0_VIDEO_CODEC", "h264").lower()
    avail = set(av.codecs_available)
    if choice == "av1":
        if "libsvtav1" in avail:
            return "av1", "libsvtav1", {"preset": "8", "crf": "32"}
        if "libaom-av1" in avail:
            return "av1", "libaom-av1", {"cpu-used": "8", "crf": "32", "b:v": "0"}
        log.warning("AV1 encoder not found, falling back to libx264")
    return "h264", "libx264", {"preset": "veryfast", "crf": "23"}


class EpisodeWriter:
    """Single-episode disk writer: 3 mp4 containers + 1 parquet buffer.

    Optional depth zarr for cameras listed in DEPTH_CAMERAS (D435 head only
    by default).
    """

    def __init__(self, task: str, subset: str, ep: int, prompt: str,
                 template_id: str, operator: str,
                 front_trim: bool | None = None) -> None:
        self.task = task
        self.subset = subset
        self.ep = ep
        self.prompt = prompt
        self.template_id = template_id
        self.operator = operator

        self.root = task_subset_root(task, subset)
        self.pq_path = self.root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        self.video_paths = {
            cam: self.root / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4"
            for cam in CAMERAS
        }
        self.depth_paths = {
            cam: self.root / "videos" / "chunk-000" / f"{cam}_depth" / f"episode_{ep:06d}.zarr"
            for cam in DEPTH_CAMERAS
        }
        for p in [self.pq_path.parent, *(v.parent for v in self.video_paths.values()),
                  *(d.parent for d in self.depth_paths.values())]:
            p.mkdir(parents=True, exist_ok=True)

        spec_name, codec_name, codec_opts = pick_codec()
        self._spec_name = spec_name
        self._codec_name = codec_name
        self._containers: dict[str, av.container.OutputContainer] = {}
        self._streams: dict[str, av.video.stream.VideoStream] = {}
        for cam, path in self.video_paths.items():
            container = av.open(str(path), mode="w")
            stream = container.add_stream(codec_name, rate=FPS)
            stream.width = WIDTH
            stream.height = HEIGHT
            stream.pix_fmt = "yuv420p"
            stream.options = dict(codec_opts)
            self._containers[cam] = container
            self._streams[cam] = stream

        self._depth_arrays: dict[str, object] = {}
        if _HAS_ZARR:
            for cam, path in self.depth_paths.items():
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                self._depth_arrays[cam] = _open_depth_zarr(path, HEIGHT, WIDTH)
        else:
            log.warning("zarr not installed, depth recording disabled")

        self._rows_state: list[list[float]] = []
        self._rows_action: list[list[float]] = []
        self._rows_ts: list[float] = []
        self._rows_intervention: list[int] = []  # int8: 0=policy, 1=human, -1=N/A
        self._frame_idx = 0   # count of EMITTED (kept) frames → pts + parquet index
        self._t0 = time.time()

        # V3 front-trim: rolling-buffer the leading ticks until motion onset,
        # then flush [onset-MARGIN:] and stream the rest. Zero re-encode,
        # memory bounded to (MARGIN+WIN) frames. Default from KAI0_FRONT_TRIM
        # env (off unless a collection entry script opts in — keeps the
        # autonomy diagnostic recorder un-trimmed).
        if front_trim is None:
            front_trim = os.environ.get("KAI0_FRONT_TRIM", "0") == "1"
        self._front_trim = bool(front_trim)
        self._onset_found = not self._front_trim   # off → everything streams now
        self._buf: list[tuple] = []                # pending (cam_arrs, depth_arrs, s, a, ts, iv)
        self._prev_action: np.ndarray | None = None
        self._run = 0                              # consecutive-moving frame counter

    def write_tick(self, frames: dict[str, np.ndarray],
                   state: list[float], action: list[float], ts: float,
                   depth_frames: dict[str, np.ndarray] | None = None,
                   intervention: int = -1) -> None:
        # Prep all per-tick payloads up front so the front-trim buffer holds
        # final-size, contiguous arrays (cheap to flush later).
        cam_arrs = {cam: self._prep_rgb(frames.get(cam)) for cam in CAMERAS}
        if self._depth_arrays:
            depth_frames = depth_frames or {}
            depth_arrs = {cam: self._prep_depth(depth_frames.get(cam)) for cam in DEPTH_CAMERAS}
        else:
            depth_arrs = {}
        s = [float(x) for x in (list(state)[:14] + [0.0] * max(0, 14 - len(state)))]
        a = [float(x) for x in (list(action)[:14] + [0.0] * max(0, 14 - len(action)))]
        iv = max(-1, min(1, int(intervention)))  # clamp to int8 (clawvla format)

        # Fast path: trim off, or onset already passed → stream immediately.
        if self._onset_found:
            self._emit_tick(cam_arrs, depth_arrs, s, a, ts, iv)
            return

        # ── V3 front-trim: buffer + incremental onset detection ──
        self._buf.append((cam_arrs, depth_arrs, s, a, ts, iv))
        a_np = np.asarray(a, dtype=np.float64)
        if self._prev_action is not None:
            da = float(np.abs(a_np[TRIM_ARM_DIMS] - self._prev_action[TRIM_ARM_DIMS]).mean())
            self._run = self._run + 1 if da > TRIM_THR else 0
        self._prev_action = a_np

        if self._run >= TRIM_WIN:
            # onset reached; the rolling buffer holds exactly [onset-MARGIN : now]
            # (proof: cap=MARGIN+WIN ⇒ buf_start = max(0, onset-MARGIN) = cut).
            self._onset_found = True
            for tk in self._buf:
                self._emit_tick(*tk)
            self._buf = []
            return

        # Cap the rolling window. Dropped frames are provably earlier than any
        # future cut (cut = future_onset - MARGIN > dropped index), so safe.
        if len(self._buf) > TRIM_MARGIN + TRIM_WIN:
            self._buf.pop(0)

    def _emit_tick(self, cam_arrs: dict, depth_arrs: dict,
                   s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Encode one kept tick → mp4 + depth zarr + parquet rows. pts/frame_index
        count only emitted (kept) frames, so trimmed output starts from 0."""
        for cam in CAMERAS:
            arr = cam_arrs.get(cam)
            if arr is None:
                arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = self._frame_idx
            for packet in self._streams[cam].encode(frame):
                self._containers[cam].mux(packet)

        if self._depth_arrays:
            for cam in DEPTH_CAMERAS:
                z = self._depth_arrays.get(cam)
                if z is None:
                    continue
                d = depth_arrs.get(cam)
                if d is None:
                    d = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                _append_depth_frame(z, d)

        self._rows_state.append(s)
        self._rows_action.append(a)
        self._rows_ts.append(float(ts - self._t0))
        self._rows_intervention.append(iv)
        self._frame_idx += 1

    def _prep_rgb(self, arr: np.ndarray | None) -> np.ndarray | None:
        return None if arr is None else self._ensure_size(arr)

    @staticmethod
    def _prep_depth(d: np.ndarray | None) -> np.ndarray | None:
        if d is None or getattr(d, "shape", None) != (HEIGHT, WIDTH):
            return None
        return np.ascontiguousarray(d.astype(np.uint16, copy=False))

    @staticmethod
    def _ensure_size(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == HEIGHT and arr.shape[1] == WIDTH and arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
        from PIL import Image
        img = Image.fromarray(arr).resize((WIDTH, HEIGHT))
        return np.asarray(img, dtype=np.uint8)

    def finalize(self) -> None:
        # Front-trim with no motion onset (degenerate/never-moved episode): keep
        # only the last MARGIN frames — matches build_no_release (cut=len-MARGIN).
        if self._front_trim and not self._onset_found and self._buf:
            for tk in self._buf[-TRIM_MARGIN:]:
                self._emit_tick(*tk)
            self._buf = []
            self._onset_found = True
        for cam, stream in self._streams.items():
            for packet in stream.encode():
                self._containers[cam].mux(packet)
            self._containers[cam].close()
        self._containers.clear()
        self._streams.clear()
        self._write_parquet()

    def abort(self) -> None:
        self._buf = []  # drop any un-flushed front-trim buffer
        for container in self._containers.values():
            try:
                container.close()
            except Exception:
                pass
        self._containers.clear()
        self._streams.clear()
        self._depth_arrays.clear()
        for path in self.video_paths.values():
            path.unlink(missing_ok=True)
        for d in self.depth_paths.values():
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        self.pq_path.unlink(missing_ok=True)

    def _write_parquet(self) -> None:
        n = len(self._rows_state)
        cols = {
            "observation.state": pa.array(self._rows_state, type=pa.list_(pa.float32())),
            "action": pa.array(self._rows_action, type=pa.list_(pa.float32())),
            "timestamp": pa.array(self._rows_ts, type=pa.float32()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "episode_index": pa.array([self.ep] * n, type=pa.int64()),
            "index": pa.array(list(range(n)), type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
        }
        # Only emit intervention column when any tick wrote a non-default value
        # (-1 means "N/A / not applicable", matches clawvla convention for non-DAgger
        # captures). For DAgger episodes intervention rows are 0 (policy) or 1 (human).
        if self._rows_intervention and any(v != -1 for v in self._rows_intervention):
            cols["intervention"] = pa.array(self._rows_intervention, type=pa.int8())
        table = pa.table(cols)
        pq.write_table(table, self.pq_path)

    @property
    def frame_count(self) -> int:
        return self._frame_idx


def write_episode_meta(writer: EpisodeWriter, duration: float,
                       success: bool = True, note: str = "",
                       scene_tags: list[str] | None = None) -> None:
    """Append one record to meta/episodes.jsonl + ensure meta/tasks.jsonl has prompt."""
    meta_dir = writer.root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "episode_id": writer.ep,
        "length": writer.frame_count,
        "duration_s": round(duration, 3),
        "operator": writer.operator,
        "prompt": writer.prompt,
        "template_id": writer.template_id,
        "success": success,
        "note": note,
        "scene_tags": list(scene_tags or []),
        "created_at": time.time(),
    }
    with (meta_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tasks_path = meta_dir / "tasks.jsonl"
    tasks_path.touch()
    existing_prompts = set()
    for ln in tasks_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            existing_prompts.add(json.loads(ln).get("task"))
        except json.JSONDecodeError:
            continue
    if writer.prompt not in existing_prompts:
        with tasks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"task_index": len(existing_prompts), "task": writer.prompt},
                ensure_ascii=False,
            ) + "\n")


def update_info_json(task: str | None, subset: str | None) -> None:
    """Re-aggregate meta/info.json from episodes.jsonl for one (task, subset)."""
    if not task or not subset:
        return
    root = task_subset_root(task, subset)
    info_path = root / "meta" / "info.json"
    ep_log_path = root / "meta" / "episodes.jsonl"
    total_ep = 0
    total_frames = 0
    if ep_log_path.exists():
        for ln in ep_log_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            total_ep += 1
            total_frames += int(d.get("length", 0))

    info = {
        "codebase_version": "v2.1",
        "robot_type": "agilex",
        "total_episodes": total_ep,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_ep * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{total_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "depth_path": "videos/chunk-{episode_chunk:03d}/{video_key}_depth/episode_{episode_index:06d}.zarr",
        "features": features_block(),
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def features_block() -> dict:
    spec_codec = pick_codec()[0]
    img_feat = {
        "dtype": "video",
        "shape": [HEIGHT, WIDTH, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": HEIGHT,
            "video.width": WIDTH,
            "video.codec": spec_codec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    depth_feat = {
        "dtype": "uint16_zarr",
        "shape": [HEIGHT, WIDTH],
        "names": ["height", "width"],
        "info": {
            "store": "zarr.DirectoryStore",
            "compressor": "blosc.zstd:level3:bitshuffle",
            "unit": "millimeter",
            "depth.height": HEIGHT,
            "depth.width": WIDTH,
            "depth.fps": FPS,
        },
    }
    return {
        **{f"observation.images.{cam}": img_feat for cam in CAMERAS},
        **{f"observation.depth.{cam}": depth_feat for cam in DEPTH_CAMERAS},
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "action": {"dtype": "float32", "shape": [14], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def next_episode_id(task: str, subset: str) -> int:
    """Scan data/chunk-000/episode_*.parquet under task_subset_root and return
    max+1 (or 0 if empty). Used by autonomy recorder to auto-pick episode_id
    without needing a UI/state-machine.
    """
    root = task_subset_root(task, subset)
    chunk_dir = root / "data" / "chunk-000"
    if not chunk_dir.exists():
        return 0
    eps = []
    for p in chunk_dir.glob("episode_*.parquet"):
        try:
            eps.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(eps) + 1) if eps else 0
