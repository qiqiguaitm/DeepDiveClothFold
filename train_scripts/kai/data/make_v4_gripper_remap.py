#!/usr/bin/env python3
"""Generate v4 datasets from v3: front/tail-trimmed v3 data + gripper dims (6,13) of
BOTH observation.state and action affine-remapped from the GLOBAL [q01,q99] (over all
in-scope v3 frames, per key/dim) onto the canonical real-robot range [0, 0.07]m
(0-70mm), clamped. Other dims untouched. Videos + depth symlinked (gripper remap does
not touch pixels), meta copied. v3 is read-only.

Layout: .../Task_*/<subset>/v3/<date>-v3  ->  .../Task_*/<subset>/v4/<date>-v4

Usage:
  python3 train_scripts/kai/data/make_v4_gripper_remap.py            # convert all KAI0 v3
  python3 train_scripts/kai/data/make_v4_gripper_remap.py --only base   # only paths containing 'base'
  python3 train_scripts/kai/data/make_v4_gripper_remap.py --dry-run     # compute params + plan, no write
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data1/DATA_IMP/KAI0"
DIMS = [6, 13]
LO, HI = 0.0, 0.07  # canonical real-robot gripper range (meters) = 0-70mm


def find_v3_dirs():
    out = subprocess.check_output(["find", ROOT, "-type", "d", "-name", "20*-v3"]).decode().split()
    return sorted(out)


def _sa(t):
    s = np.array(t.column("observation.state").to_pylist(), dtype=np.float64)
    a = np.array(t.column("action").to_pylist(), dtype=np.float64)
    return s, a


def compute_global_params(dirs):
    """One global affine per (key, dim) from [q01,q99] over ALL in-scope v3 frames."""
    acc = {("state", d): [] for d in DIMS} | {("action", d): [] for d in DIMS}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")):
            s, a = _sa(pq.read_table(f, columns=["observation.state", "action"]))
            for dim in DIMS:
                acc[("state", dim)].append(s[:, dim])
                acc[("action", dim)].append(a[:, dim])
    params = {}
    for k, vals in acc.items():
        v = np.concatenate(vals)
        q01, q99 = float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99))
        span = q99 - q01
        a = (HI - LO) / span if span > 1e-9 else 1.0
        b = LO - a * q01
        params[k] = (q01, q99, a, b)
    return params


def remap(arr, key, params):
    """arr (T,14) -> affine-remap gripper DIMS, clamp [LO,HI]; other dims untouched."""
    arr = arr.copy()
    for dim in DIMS:
        _, _, a, b = params[(key, dim)]
        arr[:, dim] = np.clip(a * arr[:, dim] + b, LO, HI)
    return arr


def v4_path(d):
    parts = d.split("/")
    assert parts[-2] == "v3", f"unexpected layout: {d}"
    parts[-2] = "v4"
    parts[-1] = parts[-1].replace("-v3", "-v4")
    return "/".join(parts)


def convert_dataset(d, params):
    v4 = v4_path(d)
    pq_files = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
    vids = [p for p in glob.glob(f"{d}/videos/**/*", recursive=True) if os.path.isfile(p)]
    print(f"  {d.replace(ROOT + '/', '')} -> v4  ({len(pq_files)} eps, {len(vids)} media)")
    # ---- data: remap gripper dims, preserve schema ----
    for f in pq_files:
        dst = os.path.join(v4, os.path.relpath(f, d))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        t = pq.read_table(f)
        s, a = _sa(t)
        s2 = remap(s, "state", params).astype(np.float32)
        a2 = remap(a, "action", params).astype(np.float32)
        st = t.schema.field("observation.state").type
        at = t.schema.field("action").type
        t = t.set_column(t.schema.get_field_index("observation.state"), "observation.state",
                         pa.array([r.tolist() for r in s2], type=st))
        t = t.set_column(t.schema.get_field_index("action"), "action",
                         pa.array([r.tolist() for r in a2], type=at))
        pq.write_table(t, dst)
    # ---- videos + depth: relative symlinks (pixels unchanged) ----
    for vf in vids:
        dst = os.path.join(v4, os.path.relpath(vf, d))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(vf, os.path.dirname(dst)), dst)
    # ---- meta: copy as-is (shapes unchanged) ----
    os.makedirs(os.path.join(v4, "meta"), exist_ok=True)
    for m in glob.glob(f"{d}/meta/*"):
        shutil.copy2(m, os.path.join(v4, "meta", os.path.basename(m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="only convert v3 dirs whose path contains this substring")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dirs = find_v3_dirs()
    print(f"in-scope v3 dirs: {len(dirs)}")
    print("computing GLOBAL [q01,q99] -> [0,0.07] params over ALL in-scope v3 ...")
    params = compute_global_params(dirs)
    for (key, dim), (q01, q99, a, b) in params.items():
        print(f"  {key}[{dim}]: q01={q01:.5f} q99={q99:.5f} -> a={a:.4f} b={b:+.5f}  "
              f"(q01->{a * q01 + b:.4f}, q99->{a * q99 + b:.4f})")

    todo = [d for d in dirs if args.only in d]
    print(f"\nconverting {len(todo)} dataset(s):")
    if args.dry_run:
        for d in todo:
            print(f"  [dry] {d.replace(ROOT + '/', '')} -> {v4_path(d).replace(ROOT + '/', '')}")
        return
    for d in todo:
        convert_dataset(d, params)
    print("\ndone")


if __name__ == "__main__":
    main()
