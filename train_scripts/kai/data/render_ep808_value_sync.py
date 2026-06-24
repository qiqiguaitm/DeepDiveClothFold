#!/usr/bin/env python3
"""Render a value-sync visualization for one AWBC milestone-value (V2.4 / CRAVE) episode:
top = top_head camera frame, bottom = per-frame `absolute_value` curve with NEGATIVE spans
(advantage<0 真退步, task_index==0) shaded red, plus a moving frame cursor.

Rescued + generalized from the 2026-06-13 inline heredoc /tmp/make_ep808_video.py that
produced temp/ep808_value_sync.mp4 + temp/ep808_preview_neg.png.

Two modes:
  (default) --video : encode the full synced mp4 (every --step-th frame).
  --png             : single-frame preview (replicates ep808_preview_neg.png). By default
                      picks the midpoint of the LONGEST negative span so the red marking is
                      visible; override with --frame N.

Data: a discretized AWBC dataset whose parquet carries absolute_value / absolute_advantage /
task_index (e.g. dagger_all_mvA). task_index==0 ⇔ "Advantage: negative" (真退步).

Usage:
  kai0/.venv/bin/python train_scripts/kai/data/render_ep808_value_sync.py --png
  kai0/.venv/bin/python train_scripts/kai/data/render_ep808_value_sync.py --video
  ... --dataset kai0/data/Task_A/self_built/dagger_all_mvA --ep 805 --frame 1200 --out temp/foo.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image

NEG_RED = "#f4a6a6"     # shaded span = negative (advantage<0)
VAL_BLUE = "#1f4e79"    # value curve
CURSOR_ORANGE = "#d35400"
RENDER_PX = 700


def neg_spans(task_index: np.ndarray):
    """Contiguous [a,b) spans where task_index==0 (negative / 真退步)."""
    neg = (task_index == 0)
    spans, s = [], None
    for i, v in enumerate(neg):
        if v and s is None:
            s = i
        elif (not v) and s is not None:
            spans.append((s, i)); s = None
    if s is not None:
        spans.append((s, len(neg)))
    return spans


def make_fig():
    fig = plt.figure(figsize=(7, 7))
    axv = fig.add_axes([0.02, 0.42, 0.96, 0.56]); axv.axis("off")
    axp = fig.add_axes([0.10, 0.07, 0.86, 0.30])
    return fig, axv, axp


def draw_frame(axv, axp, img, idx, val, adv, ti, spans, n, label):
    axv.clear(); axv.axis("off"); axv.imshow(img)
    axv.set_title(f"{label} (dagger, {n}f)  frame {idx}/{n}", fontsize=11)
    axp.clear()
    for a, b in spans:
        axp.axvspan(a, b, color=NEG_RED, alpha=.55, lw=0)
    axp.plot(np.arange(n), val, color=VAL_BLUE, lw=1.8)
    axp.axvline(idx, color="k", lw=1.2)
    axp.scatter([idx], [val[idx]], color=CURSOR_ORANGE, zorder=5, s=30)
    axp.set_xlim(0, n); axp.set_ylim(0, 1.02)
    axp.set_ylabel("value"); axp.set_xlabel("frame")
    tag = "NEG" if ti[idx] == 0 else "pos"
    axp.set_title(f"V2.4 value (red = negative: advantage<0 真退步)  "
                  f"v={val[idx]:.2f}  adv={adv[idx]:+.3f}  [{tag}]", fontsize=9)


def fig_to_px(canvas):
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
    return np.asarray(Image.fromarray(buf).resize((RENDER_PX, RENDER_PX)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kai0/data/Task_A/self_built/dagger_all_mvA",
                    help="discretized AWBC dataset root (parquet w/ absolute_value/advantage/task_index)")
    ap.add_argument("--ep", type=int, default=805, help="internal episode_index (ep805 == 'ep808' label)")
    ap.add_argument("--label", default="ep808", help="title label")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--png", action="store_true", help="single-frame preview mode")
    ap.add_argument("--video", action="store_true", help="full synced mp4 (default if neither flag)")
    ap.add_argument("--frame", type=int, default=-1, help="png mode: frame idx (-1 = mid of longest neg span)")
    ap.add_argument("--step", type=int, default=4, help="video mode: render every Nth frame")
    ap.add_argument("--fps", type=int, default=15, help="video mode: output fps")
    ap.add_argument("--out", default="", help="output path (default temp/<label>_{value_sync.mp4|preview_neg.png})")
    a = ap.parse_args()

    ds = Path(a.dataset)
    pqf = ds / "data" / "chunk-000" / f"episode_{a.ep:06d}.parquet"
    vid = ds / "videos" / "chunk-000" / a.camera / f"episode_{a.ep:06d}.mp4"
    for p in (pqf, vid):
        if not p.exists():
            raise FileNotFoundError(p)

    t = pq.read_table(pqf, columns=["absolute_value", "absolute_advantage", "task_index"])
    val = np.asarray(t["absolute_value"]); adv = np.asarray(t["absolute_advantage"]); ti = np.asarray(t["task_index"])
    n = len(val)
    spans = neg_spans(ti)
    fig, axv, axp = make_fig()
    canvas = FigureCanvasAgg(fig)

    import av  # imported lazily so --png works even if libav video write is fiddly

    do_png = a.png and not a.video
    if do_png:
        idx = a.frame
        if idx < 0:
            if spans:
                a0, b0 = max(spans, key=lambda s: s[1] - s[0])  # longest neg span → red is visible
                idx = (a0 + b0) // 2
            else:
                idx = n // 2  # no neg span; just middle
        idx = int(np.clip(idx, 0, n - 1))
        # decode just the target frame
        inp = av.open(str(vid)); vs = inp.streams.video[0]
        img = None
        for j, frame in enumerate(inp.decode(vs)):
            if j == idx:
                img = frame.to_ndarray(format="rgb24"); break
        inp.close()
        if img is None:
            raise RuntimeError(f"could not decode frame {idx} from {vid}")
        draw_frame(axv, axp, img, idx, val, adv, ti, spans, n, a.label)
        out = Path(a.out) if a.out else Path(f"temp/{a.label}_preview_neg.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fig_to_px(canvas)).save(out)
        print(f"SAVED {out}  (frame {idx}/{n}, neg-span-mid)  neg%={np.mean(ti==0)*100:.1f}", flush=True)
        return

    # video mode
    out = Path(a.out) if a.out else Path(f"temp/{a.label}_value_sync.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    oc = av.open(str(out), "w")
    ostream = oc.add_stream("libx264", rate=a.fps)
    ostream.width = RENDER_PX; ostream.height = RENDER_PX; ostream.pix_fmt = "yuv420p"; ostream.options = {"crf": "23"}
    inp = av.open(str(vid)); vs = inp.streams.video[0]
    for idx, frame in enumerate(inp.decode(vs)):
        if idx % a.step != 0 or idx >= n:
            continue
        img = frame.to_ndarray(format="rgb24")
        draw_frame(axv, axp, img, idx, val, adv, ti, spans, n, a.label)
        px = fig_to_px(canvas)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(px), format="rgb24")
        for p in ostream.encode(vf):
            oc.mux(p)
    for p in ostream.encode():
        oc.mux(p)
    oc.close(); inp.close()
    print(f"SAVED {out}  (~{n // a.step} frames @ {a.fps}fps)  neg%={np.mean(ti==0)*100:.1f}", flush=True)


if __name__ == "__main__":
    main()
