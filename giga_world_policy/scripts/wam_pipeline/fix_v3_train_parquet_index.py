#!/usr/bin/env python3
"""Fix stale episode_index/index columns in wam_fold_v3 TRAIN parquets.

ROOT CAUSE (2026-06-15): visrobot01_v3_train parquets carry pre-merge `episode_index`
(e.g. file episode_000100.parquet has episode_index=85) and non-clean global `index`.
Upstream LeRobot __getitem__ reads the per-row episode_index to set delta-query
boundaries -> 100% of action chunks clamp to a single frame (constant target) ->
model learns to emit a constant chunk -> flat MAE@1 ~= MAE@48. The val set was
already rewritten (bug 4); the train set never was. fastwam is immune because it
loads per-episode by filename and windows positionally.

FIX: for episode_{N:06d}.parquet set
  episode_index := N                         (matches filename / episodes.jsonl order)
  index         := cum[N] + frame_index      (clean global cumulative; frame_index already 0..len-1)
All other columns (observation.state, action, timestamp, frame_index, task_index)
are left byte-identical. Atomic write (temp + os.replace). Idempotent & verifiable.

Usage:
  python -m scripts.wam_pipeline.fix_v3_train_parquet_index \
      --root /mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/visrobot01_v3_train \
      [--workers 16] [--verify-only]
"""
import argparse
import json
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def build_cum(root):
    lens = []
    with open(os.path.join(root, "meta", "episodes.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                lens.append(int(json.loads(line)["length"]))
    cum = [0]
    for L in lens:
        cum.append(cum[-1] + L)
    return cum, lens


def ep_num(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def fix_one(path, gstart, expected_len, verify_only):
    t = pq.read_table(path)
    n = t.num_rows
    if n != expected_len:
        return (path, "LEN_MISMATCH", f"rows={n} jsonl={expected_len}")
    N = ep_num(path)
    fi = t.column("frame_index").to_numpy()
    want_ei = N
    want_idx = (gstart + fi).astype(np.int64)
    cur_ei = t.column("episode_index").to_numpy()
    cur_idx = t.column("index").to_numpy()
    ei_ok = bool((cur_ei == want_ei).all())
    idx_ok = bool((cur_idx == want_idx).all())
    if ei_ok and idx_ok:
        return (path, "ALREADY_CLEAN", "")
    if verify_only:
        return (path, "NEEDS_FIX", f"ei {cur_ei[0]}->{want_ei} idx {cur_idx[0]}->{want_idx[0]}")
    # rebuild columns preserving original types
    cols = {}
    for name in t.column_names:
        if name == "episode_index":
            cols[name] = pa.array(np.full(n, N, dtype=np.int64), type=t.schema.field(name).type)
        elif name == "index":
            cols[name] = pa.array(want_idx, type=t.schema.field(name).type)
        else:
            cols[name] = t.column(name)
    new_t = pa.table(cols, schema=t.schema)
    tmp = path + ".tmp"
    pq.write_table(new_t, tmp)
    os.replace(tmp, path)
    return (path, "FIXED", f"ei {cur_ei[0]}->{want_ei} idx {cur_idx[0]}->{want_idx[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cum, lens = build_cum(args.root)
    files = sorted(glob.glob(os.path.join(args.root, "data", "chunk-*", "episode_*.parquet")), key=ep_num)
    print(f"[fix] root={args.root}")
    print(f"[fix] episodes.jsonl={len(lens)} parquets={len(files)} verify_only={args.verify_only}")
    if len(files) != len(lens):
        print(f"[fix] WARNING: parquet count {len(files)} != episodes.jsonl {len(lens)}")

    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for p in files:
            N = ep_num(p)
            tasks.append(ex.submit(fix_one, p, cum[N], lens[N], args.verify_only))
        from collections import Counter
        c = Counter()
        examples = []
        for fut in as_completed(tasks):
            path, status, msg = fut.result()
            c[status] += 1
            if status in ("LEN_MISMATCH",) and len(examples) < 20:
                examples.append((path, status, msg))
            if status in ("FIXED", "NEEDS_FIX") and len([e for e in examples if e[1] == status]) < 3:
                examples.append((path, status, msg))
    print(f"[fix] result: {dict(c)}")
    for path, status, msg in examples:
        print(f"   {status}: {os.path.basename(path)} {msg}")


if __name__ == "__main__":
    main()
