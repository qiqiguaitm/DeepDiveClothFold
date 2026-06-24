"""Single-file depth archive: pack a depth `.zarr/` directory (one tiny file per
frame-chunk → ~1.7k files/episode) into one `.zarr.zip` object, and read it back
by extracting to a temp dir on demand.

Why: depth zarr DirectoryStore emits ~1749 tiny files per episode. On object
storage (TOS) that means ~155k objects/day → listing/transfer is the bottleneck
(hot-sync excludes depth entirely for this reason). Training never reads depth;
the only readers are the data_manager viewer + the offline video_publisher replay
node. So we store depth as ONE file per episode and decompress to a temp dir when
a reader actually needs the frames.

Format: ZIP (ZIP_STORED — chunks are already blosc-zstd compressed, so the outer
container adds no recompression CPU) whose root holds the CONTENTS of the `.zarr`
dir (`.zarray`, `.zattrs`, chunk files `0.0.0` …). Extracting to a temp dir T then
`zarr.open(T)` yields the original array.

All readers stay BACKWARD-COMPATIBLE: if the legacy `.zarr/` dir is present it is
used directly; the `.zarr.zip` is preferred when both exist.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ZIP_SUFFIX = ".zip"  # artifact = "<...>.zarr.zip"


def zip_path_for(zarr_dir: Path | str) -> Path:
    """The `.zarr.zip` sibling path for a base `.zarr` dir path."""
    return Path(str(zarr_dir) + ZIP_SUFFIX)


def resolve_depth_artifact(zarr_dir: Path | str) -> Path | None:
    """Given the base `.../episode_X.zarr` path, return the artifact that exists:
    prefer the lossless FFV1 `.mkv` (newest, ~58% smaller), then the packed
    `.zarr.zip`, then the legacy `.zarr/` dir, else None."""
    zarr_dir = Path(zarr_dir)
    mkv = Path(str(zarr_dir).removesuffix(".zarr") + ".mkv")
    if mkv.is_file():
        return mkv
    zp = zip_path_for(zarr_dir)
    if zp.is_file():
        return zp
    if zarr_dir.is_dir():
        return zarr_dir
    return None


def pack_zarr_dir(zarr_dir: Path | str, *, remove_dir: bool = True) -> Path:
    """Pack a `.zarr/` directory into a sibling `.zarr.zip` (ZIP_STORED, contents
    at zip root). Atomic: writes to a .tmp then renames. Returns the zip path.
    On `remove_dir`, deletes the source dir only after the zip is in place."""
    zarr_dir = Path(zarr_dir)
    if not zarr_dir.is_dir():
        raise FileNotFoundError(f"not a zarr dir: {zarr_dir}")
    zp = zip_path_for(zarr_dir)
    tmp = Path(str(zp) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(zarr_dir):
            for name in files:
                fp = Path(root) / name
                arc = fp.relative_to(zarr_dir).as_posix()  # contents at zip root
                zf.write(fp, arc)
    os.replace(tmp, zp)
    if remove_dir:
        shutil.rmtree(zarr_dir, ignore_errors=True)
    return zp


def open_depth_readonly(artifact: Path | str):
    """Open a depth artifact (`.zarr.zip` OR legacy `.zarr/` dir) read-only.

    Returns (zarr_array, tmpdir) where tmpdir is a str to rmtree when done
    (None for the legacy dir path, which is opened in place). Caller MUST
    clean up tmpdir (use `close_depth(tmpdir)`)."""
    import zarr  # lazy: only when a reader actually needs frames

    artifact = Path(artifact)
    if artifact.suffix == ".mkv":  # lossless FFV1 gray16le depth video
        import av
        import numpy as np

        c = av.open(str(artifact))
        frames = [f.to_ndarray() for f in c.decode(video=0)]  # each (H,W) uint16
        c.close()
        return np.stack(frames), None  # (T,H,W) uint16 — supports .shape / arr[i]
    if artifact.is_dir():
        return zarr.open(str(artifact), mode="r"), None
    if artifact.suffix == ZIP_SUFFIX:
        tmp = tempfile.mkdtemp(prefix="kai0_depthz_")
        try:
            with zipfile.ZipFile(artifact) as zf:
                zf.extractall(tmp)
            return zarr.open(tmp, mode="r"), tmp
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    raise FileNotFoundError(f"no depth artifact at {artifact}")


def close_depth(tmpdir: str | None) -> None:
    """Remove the temp dir returned by open_depth_readonly (no-op if None)."""
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
