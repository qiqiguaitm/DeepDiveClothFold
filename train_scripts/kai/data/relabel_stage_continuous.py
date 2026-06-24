#!/usr/bin/env python3
"""Retrofit an existing stage-labeled dataset whose `stage_progress_gt` is a FLAT per-stage
constant ({0.25,0.75}) into the correct per-stage LINEAR-INTERPOLATED 0→1 progress (README
Step-0 spec). Flat per-stage constants make the AE regression target (sp_gt[t]−sp_gt[t-100])
~94% exactly-zero → AE collapses to ≈0 (dead value). Continuous within-stage ramp = dense
target → AE trains through (matches KAI0's manual labels).

Non-destructive: writes a NEW dataset (parquets rewritten, videos/ symlinked to source dataset,
meta copied). t_star (flat→fold boundary) is recovered from the existing flat labels
(first frame with sp_gt>0.5). K=2 stages.

Run: kai0/.venv/bin/python train_scripts/kai/data/relabel_stage_continuous.py \
        --src kai0/data/Task_A/self_built/vis_awbc_merged_stage \
        --dst kai0/data/Task_A/self_built/vis_awbc_merged_stage_interp
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path("/home/tim/workspace/deepdive_kai0")


def interp_sp_gt(old: np.ndarray) -> tuple[np.ndarray, int]:
    """old flat {0.25,0.75} → continuous 0→1; returns (new, t_star)."""
    n = len(old)
    fold = np.where(old > 0.5)[0]
    t_star = int(fold[0]) if len(fold) else n  # all-flat if no fold
    new = np.zeros(n, dtype=np.float32)
    if t_star > 0:
        new[:t_star] = 0.5 * (np.arange(t_star, dtype=np.float32) / t_star)
    if t_star < n:
        lf = n - t_star
        new[t_star:] = 0.5 + 0.5 * (np.arange(lf, dtype=np.float32) / lf)
    return new, t_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src = Path(a.src) if Path(a.src).is_absolute() else REPO / a.src
    dst = Path(a.dst) if Path(a.dst).is_absolute() else REPO / a.dst
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)

    # meta: copy
    shutil.copytree(src / "meta", dst / "meta")
    for extra in ("norm_stats.json", ".kai0_ts_validated"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    # videos: symlink the whole subtree (zero-copy; source already resolves)
    (dst / "videos").symlink_to((src / "videos").resolve())

    pqs = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    t_stars = {}
    n_allflat = 0
    for p in pqs:
        rel = p.relative_to(src / "data")
        out = dst / "data" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        t = pq.read_table(p)
        old = np.asarray(t["stage_progress_gt"]).astype(np.float32)
        new, t_star = interp_sp_gt(old)
        ep = int(p.stem.split("_")[1])
        t_stars[ep] = t_star
        if t_star >= len(old):
            n_allflat += 1
        col = pa.array(new, type=pa.float32())
        idx = t.column_names.index("stage_progress_gt")
        t = t.set_column(idx, "stage_progress_gt", col)
        pq.write_table(t, out)
    (dst / "meta" / "_t_stars_recovered.json").write_text(json.dumps(t_stars))
    print(f"relabeled {len(pqs)} ep → {dst}", flush=True)
    print(f"  all-flat (no fold) ep: {n_allflat}  | t_star median={int(np.median(list(t_stars.values())))}", flush=True)
    # sanity on one ep
    s = pq.read_table(sorted((dst / 'data').glob('chunk-*/episode_*.parquet'))[0])["stage_progress_gt"].to_numpy()
    print(f"  sample ep new sp_gt: start={s[0]:.3f} end={s[-1]:.3f} max={s.max():.3f} n_distinct={len(set(np.round(s,3)))}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
