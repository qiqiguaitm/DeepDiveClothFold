#!/usr/bin/env python3
"""Run a trained Advantage-Estimator (AE) on ONE lerobot episode and visualize its
per-frame absolute_value (+ absolute_advantage) — quick "看效果" preview, no dataset writes.

Reuses SimpleValueEvaluator (kai0/stage_advantage/annotation/evaluator.py), the same engine
Stage-2 eval.py uses. Output: a PNG with K sampled top_head frames (value-annotated) on top
and the value/advantage curves below.

Run (from repo root, local GPU):
  CUDA_VISIBLE_DEVICES=0 kai0/.venv/bin/python train_scripts/kai/data/viz_ae_value_episode.py \
      --src kai0/data/Task_A/vis_base/v3/2026-04-30-v3 --ep 11 \
      --ckpt kai0/checkpoints/ADVANTAGE_TORCH_VIS_AWBC/adv_est_vis_v1/100000 \
      --config ADVANTAGE_TORCH_VIS_AWBC
"""
import argparse, sys
from pathlib import Path
import numpy as np
import av
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/tim/workspace/deepdive_kai0")
sys.path.insert(0, str(REPO / "kai0" / "stage_advantage" / "annotation"))
from evaluator import SimpleValueEvaluator  # noqa: E402

CAMS = ("top_head", "hand_left", "hand_right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="lerobot dataset dir (e.g. .../vis_base/v3/2026-04-30-v3)")
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="AE ckpt step dir (…/adv_est_vis_v1/100000)")
    ap.add_argument("--config", default="ADVANTAGE_TORCH_VIS_AWBC")
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--rel-interval", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--nframes", type=int, default=6, help="how many top_head frames to strip on top")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    src = Path(a.src) if Path(a.src).is_absolute() else REPO / a.src
    ckpt = a.ckpt if Path(a.ckpt).is_absolute() else str(REPO / a.ckpt)
    e6 = f"episode_{a.ep:06d}"
    vids = tuple(src / "videos" / "chunk-000" / f"observation.images.{c}" / f"{e6}.mp4" for c in CAMS)
    pqf = src / "data" / "chunk-000" / f"{e6}.parquet"
    for p in (*vids, pqf):
        if not p.exists():
            sys.exit(f"FATAL missing {p}")

    fi = pq.read_table(pqf)["frame_index"].to_pylist()
    print(f"[ep{a.ep}] {len(fi)} frames; running AE {a.config} @ {ckpt}", flush=True)

    ev = SimpleValueEvaluator(config_name=a.config, ckpt_dir=ckpt, num_workers=16)
    res = ev.evaluate_video_2timesteps_advantages(
        video_paths=vids, prompt=a.prompt, batch_size=a.batch_size, frame_interval=1,
        relative_interval=a.rel_interval, min_frame_index=fi[0], max_frame_index=fi[-1], prefetch=True,
    )
    res = sorted(res, key=lambda r: r["frame_idx"])
    val = np.array([r["absolute_value"] for r in res], dtype=np.float32)
    adv = np.array([r["absolute_advantage"] for r in res], dtype=np.float32)
    n = len(val)
    tag = f"{src.name}_ep{a.ep}"
    np.savez(REPO / "temp" / f"ae_value_{tag}.npz", value=val, adv=adv)

    # sample top_head frames at K evenly-spaced indices
    idxs = np.linspace(0, n - 1, a.nframes).astype(int)
    want = set(int(i) for i in idxs)
    frames = {}
    c = av.open(str(vids[0]))
    for j, fr in enumerate(c.decode(video=0)):
        if j in want:
            frames[j] = fr.to_ndarray(format="rgb24")
        if j >= idxs[-1]:
            break
    c.close()

    mono = float(np.mean(np.diff(val) >= -1e-3)) if n > 1 else 1.0
    fig = plt.figure(figsize=(2.3 * a.nframes, 6.4))
    w = 0.97 / a.nframes
    for k, ix in enumerate(idxs):
        axi = fig.add_axes([0.015 + k * w, 0.52, w * 0.92, 0.44]); axi.axis("off")
        if ix in frames:
            axi.imshow(frames[ix])
        axi.set_title(f"f{ix}  v={val[ix]:.2f}", fontsize=9)
    axp = fig.add_axes([0.06, 0.08, 0.9, 0.36])
    axp.plot(val, color="#1f4e79", lw=2.0, label="absolute_value")
    axp.plot(adv, color="#d62728", lw=1.0, alpha=.7, label="absolute_advantage")
    axp.axhline(0, color="k", lw=.5)
    for ix in idxs:
        axp.axvline(ix, color="gray", ls=":", lw=.7)
    axp.set_xlim(0, n); axp.set_ylim(-0.25, 1.05)
    axp.set_xlabel("frame"); axp.set_ylabel("value / adv")
    axp.legend(loc="upper left", fontsize=9)
    axp.set_title(f"NEW vis-AE (adv_est_vis_v1/100000) value · {tag} ({n}f) · "
                  f"monotone={mono:.0%}  start={val[0]:.2f}  end={val[-1]:.2f}  max={val.max():.2f}",
                  fontsize=10)
    out = a.out or str(REPO / "temp" / f"ae_value_{tag}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"SAVED {out}  n={n} start={val[0]:.3f} end={val[-1]:.3f} max={val.max():.3f} mono={mono:.0%}", flush=True)


if __name__ == "__main__":
    main()
