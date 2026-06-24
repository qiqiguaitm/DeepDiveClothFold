"""Video writing helper (PyAV). Defaults to NVENC, falls back to libx264."""
from __future__ import annotations

import numpy as np


class VideoWriter:
    """Minimal RGB frame sink:  with VideoWriter(path, fps) as w: w.add(frame_hwc_rgb)."""

    def __init__(self, path: str, fps: float = 30.0, codec: str = "h264_nvenc"):
        import av
        self._av = av
        self.container = av.open(str(path), "w")
        try:
            self.stream = self.container.add_stream(codec, rate=int(round(fps)))
        except Exception:
            self.stream = self.container.add_stream("libx264", rate=int(round(fps)))
        self.stream.pix_fmt = "yuv420p"
        self._init = False

    def add(self, frame_rgb: np.ndarray):
        f = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        if not self._init:
            h, w = f.shape[:2]
            self.stream.height, self.stream.width = h - h % 2, w - w % 2
            self._init = True
        f = f[: self.stream.height, : self.stream.width]
        vf = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for pkt in self.stream.encode(vf):
            self.container.mux(pkt)

    def close(self):
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
