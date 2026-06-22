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
import queue
import shutil
import threading
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

from .depth_archive import pack_zarr_dir, zip_path_for
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
    cams: tuple[str, ...] = ()
    for parent in here.parents:
        candidate = parent / "config" / "camera_depth_flags.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "kai0_camera_depth_flags", candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            cams = tuple(mod.DEPTH_CAMERAS)
            break
    # Per-run head-depth override: KAI0_HEAD_DEPTH=0 (set by start_dagger_collect.sh
    # for V1/v0 dagger) drops top_head from the recorded set, so we don't subscribe
    # to / allocate a top_head depth zarr that would otherwise fill with zeros once
    # multi_camera stops publishing head depth. Env UNSET → keep the file default
    # (teleop path is unaffected). Only "0" disables; any other value is a no-op.
    if os.environ.get("KAI0_HEAD_DEPTH") == "0":
        cams = tuple(c for c in cams if c != "top_head")
    return cams


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

# ── V3 online tail-trim (trailing post-task idle cap at record time) ──
# Mirrors build_no_release.py::tail_cap_keep_indices: a trailing frame is "idle"
# only when BOTH arm AND gripper are static, so a final gripper release/place is
# NEVER dropped; the long post-completion hold is capped to TAIL_CAP terminal
# settle frames. ONLY the trailing run is touched — interior idle streams as-is
# (no middle thinning), so per-chunk task-motion displacement is unchanged.
TRIM_GRIP_DIMS = [6, 13]   # L/R gripper action dims (excluded from TRIM_ARM_DIMS)
TRIM_GRIP_THR = 0.02       # |Δgrip| above this => gripper acting (grasp/release)
TAIL_CAP = 15              # keep this many trailing-idle frames as terminal settle (~0.5s @30Hz)

log = logging.getLogger(__name__)


def task_subset_root(task: str, subset: str) -> Path:
    """`<DATA_ROOT>/<task>/<subset>/<today-v2>` for new episodes (v2 layout)."""
    return new_task_subset_root(task, subset)


def pick_codec() -> tuple[str, str, dict]:
    """Pick video codec — h264 (default, broad compatibility), av1 (compact),
    or nvenc (GPU hardware H.264, keeps the mp4 encode off the CPU).

    KAI0_VIDEO_CODEC:
      h264   (default) — libx264 veryfast (CPU).
      av1              — libsvtav1 → libaom-av1 → falls back to h264 (CPU).
      nvenc | gpu      — h264_nvenc (GPU). Encodes on KAI0_NVENC_GPU (default
                         '0' = first CUDA-visible device); point it at an *idle*
                         card so the encode steals neither inference CPU cores
                         nor the inference GPU. Falls back to libx264 when the
                         linked PyAV/ffmpeg has no NVENC — e.g. kai0/.venv pins
                         av==13 (no nvenc), while backend/.venv PyAV 17 has it,
                         so the teleop recorder gets GPU encode and the dagger
                         recorder degrades gracefully to libx264.
    """
    choice = os.environ.get("KAI0_VIDEO_CODEC", "h264").lower()
    avail = set(av.codecs_available)
    if choice in ("nvenc", "gpu", "h264_nvenc"):
        if "h264_nvenc" in avail:
            gpu = os.environ.get("KAI0_NVENC_GPU", "0")
            # p4 = balanced preset; vbr+cq for constant-quality ≈ libx264 crf 23.
            return "h264", "h264_nvenc", {
                "preset": "p4", "tune": "ll", "rc": "vbr", "cq": "23",
                "gpu": str(gpu),
            }
        log.warning("h264_nvenc not in this PyAV build, falling back to libx264")
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
                 front_trim: bool | None = None,
                 tail_trim: bool | None = None) -> None:
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

        # Encoder warmup: nvenc opens an encode session on its FIRST encode
        # (~0.3-0.6s per stream). That stall, paid in the capture loop, makes the
        # 30Hz recorder fall behind and drop the first ~0.5s of frames. Pay it HERE
        # in __init__ (recorder.start(), before the capture thread exists) instead.
        # _skip_packets cancels the one frame nvenc buffers from the warmup so the
        # output stays exactly N frames with 0-based PTS. libx264 has no such stall,
        # so warmup is nvenc-only.
        self._skip_packets: dict[str, int] = {}
        self._force_idr: dict[str, int] = {}
        if codec_name == "h264_nvenc" and os.environ.get("KAI0_ENCODER_WARMUP", "1") == "1":
            self._warmup_encoders()

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
        self._rows_intervention: list[int] = []  # int8: 0=policy, 1=human, -1=N/A
        self._frame_idx = 0   # count of EMITTED (kept) frames → drives pts + parquet
                              # index + the frame_index/fps timestamp (all 0-based)

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

        # V3 online tail-trim: hold the trailing idle run (arm AND gripper static)
        # and cap it to TAIL_CAP terminal frames at finalize. Frames that turn out
        # interior (motion resumes) are flushed un-touched. Default follows
        # front_trim (i.e. on for V3 collection) unless KAI0_TAIL_TRIM is set
        # explicitly. Independent of the onset detection above.
        if tail_trim is None:
            tail_trim = os.environ.get(
                "KAI0_TAIL_TRIM", "1" if self._front_trim else "0") == "1"
        self._tail_trim = bool(tail_trim)
        self._tail_buf: list[tuple] = []           # consecutive trailing-idle ticks held back
        self._tail_prev_action: np.ndarray | None = None  # prev STAGED action (Δ for active test)

        # Optional async writer (KAI0_ASYNC_WRITER=1): the capture thread preps +
        # enqueues at 30Hz; this background thread drains the queue and does the
        # heavy front-trim + encode + depth compress, so a slow tick (NVENC/IO
        # stall, GIL contention) never stalls the grab loop → no record-time frame
        # drops. Off by default; sync mode is the legacy path with identical output.
        self._async = os.environ.get("KAI0_ASYNC_WRITER", "0") == "1"
        self._q: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._worker_exc: BaseException | None = None
        self._dropped = 0
        if self._async:
            self._q = queue.Queue(maxsize=512)   # ~17s @30Hz headroom for transient stalls
            self._worker = threading.Thread(target=self._writer_loop,
                                            name=f"epwriter-{task}-{ep}", daemon=True)
            self._worker.start()

    def write_tick(self, frames: dict[str, np.ndarray],
                   state: list[float], action: list[float], ts: float,
                   depth_frames: dict[str, np.ndarray] | None = None,
                   intervention: int = -1) -> None:
        # Prep all per-tick payloads up front. _prep_rgb/_prep_depth copy into
        # contiguous final-size arrays, so the result is self-owned — safe to hand
        # to the async writer thread and frees the bridge's frame buffer now.
        cam_arrs = {cam: self._prep_rgb(frames.get(cam)) for cam in CAMERAS}
        if self._depth_arrays:
            depth_frames = depth_frames or {}
            depth_arrs = {cam: self._prep_depth(depth_frames.get(cam)) for cam in DEPTH_CAMERAS}
        else:
            depth_arrs = {}
        s = [float(x) for x in (list(state)[:14] + [0.0] * max(0, 14 - len(state)))]
        a = [float(x) for x in (list(action)[:14] + [0.0] * max(0, 14 - len(action)))]
        iv = max(-1, min(1, int(intervention)))  # clamp to int8 (clawvla format)
        item = (cam_arrs, depth_arrs, s, a, ts, iv)

        # Async mode: enqueue and return — the heavy front-trim + encode + depth
        # compress run on the writer thread, so a slow tick never stalls the 30Hz
        # grab loop (record-time frame-drop root fix). Sync mode: process inline.
        if self._async:
            if self._worker_exc is not None:        # writer thread died → surface to caller
                raise self._worker_exc
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 30 == 0:
                    log.warning("[async-writer] queue full, dropped %d tick(s) "
                                "(writer can't keep up)", self._dropped)
            return
        self._ingest(*item)

    def _ingest(self, cam_arrs: dict, depth_arrs: dict,
                s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Front-trim onset buffering + staging. Runs on the capture thread (sync
        mode) OR the writer thread (async mode) — never both at once, so the
        front-trim state needs no extra lock."""
        # Fast path: front-trim off, or onset already passed → hand to the
        # tail-trim stage immediately (which is a passthrough when tail_trim off).
        if self._onset_found:
            self._stage_tick(cam_arrs, depth_arrs, s, a, ts, iv)
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
                self._stage_tick(*tk)
            self._buf = []
            return

        # Cap the rolling window. Dropped frames are provably earlier than any
        # future cut (cut = future_onset - MARGIN > dropped index), so safe.
        if len(self._buf) > TRIM_MARGIN + TRIM_WIN:
            self._buf.pop(0)

    def _writer_loop(self) -> None:
        """Async writer thread: drain queued ticks → full front-trim + encode +
        depth pipeline. Exits on the None sentinel, on stop, or on first
        processing error (stored in _worker_exc → surfaced to the caller at the
        next write_tick / at finalize)."""
        while True:
            item = self._q.get()
            try:
                if item is None or self._stop.is_set():
                    return
                self._ingest(*item)
            except BaseException as e:  # noqa: BLE001
                self._worker_exc = e
                log.exception("[async-writer] tick processing failed; stopping writer")
                return
            finally:
                self._q.task_done()

    def _stage_tick(self, cam_arrs: dict, depth_arrs: dict,
                    s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Tail-trim stage between front-trim and encode. Holds a run of trailing
        idle ticks (arm AND gripper static vs the previous staged action); when
        motion resumes the held run is provably interior → flushed un-touched, when
        the episode ends the held run is the post-task hold → capped to TAIL_CAP in
        finalize(). Passthrough (emit immediately) when tail_trim is off."""
        if not self._tail_trim:
            self._emit_tick(cam_arrs, depth_arrs, s, a, ts, iv)
            return
        a_np = np.asarray(a, dtype=np.float64)
        if self._tail_prev_action is None:
            active = True   # first staged frame = anchor (matches offline active[0]=True)
        else:
            d_arm = float(np.abs(a_np[TRIM_ARM_DIMS]
                                 - self._tail_prev_action[TRIM_ARM_DIMS]).mean())
            d_grip = float(np.abs(a_np[TRIM_GRIP_DIMS]
                                  - self._tail_prev_action[TRIM_GRIP_DIMS]).max())
            active = (d_arm > TRIM_THR) or (d_grip > TRIM_GRIP_THR)
        self._tail_prev_action = a_np
        if active:
            # motion resumed → the whole held run was interior idle, keep it all
            for tk in self._tail_buf:
                self._emit_tick(*tk)
            self._tail_buf = []
            self._emit_tick(cam_arrs, depth_arrs, s, a, ts, iv)
        else:
            # might be the trailing post-task hold — hold back until we know
            self._tail_buf.append((cam_arrs, depth_arrs, s, a, ts, iv))

    def _warmup_encoders(self) -> None:
        """Pay the nvenc per-session init (the ~0.3-0.6s/stream first-encode stall)
        at construction time so the capture loop never sees it. Feed one throwaway
        black frame per stream with a NEGATIVE pts (so the real frames' pts 0..N-1
        stay strictly increasing); discard whatever it emits now, and record the one
        frame nvenc keeps buffered (1-frame delay) in _skip_packets so _emit_tick
        drops it when it surfaces on the first real encode — output stays exactly N
        frames, 0-based PTS (verified)."""
        black = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for cam, stream in self._streams.items():
            try:
                frame = av.VideoFrame.from_ndarray(black, format="rgb24")
                frame.pts = -1
                emitted = sum(1 for _ in stream.encode(frame))
                self._skip_packets[cam] = max(0, 1 - emitted)
                # The warmup frame's keyframe packet is skipped, so the FIRST real
                # frame MUST be forced to an IDR keyframe — otherwise the muxed
                # stream has no keyframe and the whole video is undecodable (black).
                self._force_idr[cam] = 1
            except Exception:  # noqa: BLE001
                log.warning("encoder warmup failed for %s, skipping", cam, exc_info=True)
                self._skip_packets[cam] = 0
                self._force_idr[cam] = 0

    def _emit_tick(self, cam_arrs: dict, depth_arrs: dict,
                   s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Encode one kept tick → mp4 + depth zarr + parquet rows. pts/frame_index
        count only emitted (kept) frames, so trimmed output starts from 0 and the
        video PTS is zeroed by construction (first kept frame → pts 0). The parquet
        timestamp is derived as frame_index/fps in _write_parquet (NOT the wall-clock
        ts arg), keeping it aligned with the zeroed PTS after front/tail trim — see
        docs/deployment/training_ops/dataset_trimming_and_pts.md. `ts` is unused now
        (kept in the signature for callers / future diagnostics)."""
        for cam in CAMERAS:
            arr = cam_arrs.get(cam)
            if arr is None:
                arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = self._frame_idx
            if self._force_idr.get(cam, 0) > 0:
                frame.pict_type = av.video.frame.PictureType.I  # first real frame = keyframe
                self._force_idr[cam] -= 1
            for packet in self._streams[cam].encode(frame):
                if self._skip_packets.get(cam, 0) > 0:
                    self._skip_packets[cam] -= 1   # drop the warmup frame's delayed packet
                    continue
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
        # Async: drain all queued ticks (process them) then stop the writer thread
        # BEFORE we touch encoders/buffers on this thread. The recorder already
        # joined its capture thread (no more enqueues arrive), so this is race-free.
        if self._async and self._worker is not None:
            if self._worker_exc is None:
                self._q.put(None)            # FIFO sentinel → drains all real items first
                self._worker.join()
            else:
                self._worker.join(timeout=1.0)
            if self._dropped:
                log.warning("[async-writer] ep=%d: %d tick(s) dropped under backpressure",
                            self.ep, self._dropped)
            if self._worker_exc is not None:
                raise self._worker_exc
        # Front-trim with no motion onset (degenerate/never-moved episode): keep
        # only the last MARGIN frames — matches build_no_release (cut=len-MARGIN).
        if self._front_trim and not self._onset_found and self._buf:
            for tk in self._buf[-TRIM_MARGIN:]:
                self._stage_tick(*tk)
            self._buf = []
            self._onset_found = True
        # Tail-trim: the still-held run is the trailing post-task idle; keep only
        # the first TAIL_CAP terminal-settle frames, drop the rest. Matches
        # build_no_release.tail_cap_keep_indices (keep arange(0, T-(tail-TAIL_CAP))).
        if self._tail_trim and self._tail_buf:
            for tk in self._tail_buf[:TAIL_CAP]:
                self._emit_tick(*tk)
            self._tail_buf = []
        for cam, stream in self._streams.items():
            for packet in stream.encode():
                self._containers[cam].mux(packet)
            self._containers[cam].close()
        self._containers.clear()
        self._streams.clear()
        self._write_parquet()
        # Video alignment self-check (fast, mp4-only) — BEFORE backgrounding the
        # depth pack so a bad video still raises synchronously on save.
        if os.environ.get("KAI0_VALIDATE_TRIM", "0") == "1":
            self._validate_alignment()
        # Pack each depth `.zarr/` dir (~3000 tiny files) into one `.zarr.zip`.
        # This is ~1-2s for an ~800MB episode and dominated the save latency, so it
        # runs in a BACKGROUND daemon thread — the save response (pedal/UI) returns
        # immediately. The `.zarr/` dir is itself a valid depth representation
        # (readers handle both forms), so if the process exits before the pack
        # finishes nothing is lost; it just stays unpacked. KAI0_DEPTH_PACK_SYNC=1
        # forces the old inline behavior (e.g. when an immediate TOS push needs the
        # single-file form).
        dirs = [d for d in self.depth_paths.values() if d.is_dir()]
        if dirs:
            if os.environ.get("KAI0_DEPTH_PACK_SYNC", "0") == "1":
                self._pack_depth(dirs)
            else:
                threading.Thread(target=self._pack_depth, args=(dirs,),
                                 name=f"depthpack-{self.task}-{self.ep}", daemon=True).start()

    @staticmethod
    def _pack_depth(dirs: list) -> None:
        for dpath in dirs:
            try:
                pack_zarr_dir(dpath, remove_dir=True)
            except Exception:  # noqa: BLE001
                log.warning("depth pack failed for %s, keeping .zarr dir", dpath, exc_info=True)

    def _validate_alignment(self) -> None:
        """docs/deployment/training_ops/dataset_trimming_and_pts.md §4 checklist:
        every video's first PTS == 0 and frame count == parquet rows. Structurally
        guaranteed by _emit_tick (pts = frame_index, one row per emitted frame), but
        a record-time spot check catches encoder/mux regressions before training.

        Uses packet DEMUX (no decode): one coded packet per frame, so len(packets)
        == frame count, and min(pts) == first displayed frame's pts. ~10-50× cheaper
        than decoding every frame — important because finalize runs under the
        recorder lock, so a slow check would freeze the backend on save. Gated by
        KAI0_VALIDATE_TRIM."""
        n_rows = pq.read_metadata(self.pq_path).num_rows
        for path in self.video_paths.values():
            with av.open(str(path)) as c:
                ptss = [p.pts for p in c.demux(c.streams.video[0]) if p.pts is not None]
            first_pts = min(ptss) if ptss else None
            if first_pts != 0:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: first pts={first_pts} != 0 "
                    f"(video PTS not zeroed → visual↔action skew)")
            if len(ptss) != n_rows:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: video frames {len(ptss)} != parquet rows {n_rows}")
            # Decode the first frame (cheap, 1 frame): catches a missing keyframe /
            # undecodable stream — the demux count+pts check above passes even when
            # the video is all-black (e.g. the encoder-warmup keyframe got skipped).
            with av.open(str(path)) as c:
                first = next(c.decode(c.streams.video[0]), None)
            if first is None:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: no decodable frame (missing keyframe?)")
            if first.to_ndarray(format="rgb24").mean() < 2.0:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: first frame is black (decode/keyframe broken)")
        log.info("[trim-validate] ep=%d OK: first-pts=0, frames==rows==%d, first-frame decodes",
                 self.ep, n_rows)

    def abort(self) -> None:
        # Async: stop the writer thread fast and discard whatever's still queued.
        if self._async and self._worker is not None:
            self._stop.set()
            try:
                self._q.put_nowait(None)     # wake a blocked get()
            except queue.Full:
                pass
            self._worker.join(timeout=2.0)
        self._buf = []       # drop any un-flushed front-trim buffer
        self._tail_buf = []  # drop any held trailing-idle buffer
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
            zip_path_for(d).unlink(missing_ok=True)  # in case a prior pack ran
        self.pq_path.unlink(missing_ok=True)

    def _write_parquet(self) -> None:
        n = len(self._rows_state)
        cols = {
            "observation.state": pa.array(self._rows_state, type=pa.list_(pa.float32())),
            "action": pa.array(self._rows_action, type=pa.list_(pa.float32())),
            # lerobot-standard timestamp = frame_index / fps (0-based, contiguous) —
            # matches build_no_release. NOT wall-clock: each emitted tick is exactly
            # one video frame (pts = frame_index, first kept frame pts = 0), so
            # frame_index/fps is the only axis that stays aligned with the zeroed
            # video PTS after front/tail trim. Wall-clock here would re-introduce the
            # §2 PTS-style visual↔action skew (invisible to offline MAE). See
            # docs/deployment/training_ops/dataset_trimming_and_pts.md.
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS, type=pa.float32()),
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
                       scene_tags: list[str] | None = None,
                       extra: dict | None = None) -> None:
    """Append one record to meta/episodes.jsonl + ensure meta/tasks.jsonl has prompt.

    `extra` merges additional keys into the record (e.g. terminal-cause labels
    like {"terminal": "intervention", "intervention_frame_index": N} that the
    dagger recorder attaches to policy-rollout episodes)."""
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
    if extra:
        rec.update(extra)
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
        "depth_path": "videos/chunk-{episode_chunk:03d}/{video_key}_depth/episode_{episode_index:06d}.zarr.zip",
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
            "container": "zip",  # packed as one episode_NNNNNN.zarr.zip (ZIP_STORED); unzip to read
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
