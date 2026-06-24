#!/usr/bin/env python3
"""Phase 2: gripper remap for Task_A v4 (base+dagger).

Compute ONE global [q01,q99] over ALL v3 gripper dims (the old-frame source), then
write canonical 0-70mm gripper into the matching v4 parquet. Sourcing from v3 (not
v4) means native-v3-derived v4 (already remapped earlier) isn't double-remapped.
v3 stays old-frame; only v4 parquet's dims 6,13 (state+action) change. Clamped.
"""
import glob
import os
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data1/DATA_IMP/KAI0/Task_A"
DIMS = [6, 13]
LO, HI = 0.0, 0.07


def v3_dirs():
    out = subprocess.check_output(
        ["bash", "-c", f"ls -d {ROOT}/base/v3/*-v3 {ROOT}/dagger/v3/*-v3 2>/dev/null"]).decode().split()
    return sorted(out)


def _sa(t):
    return (np.array(t.column("observation.state").to_pylist(), dtype=np.float64),
            np.array(t.column("action").to_pylist(), dtype=np.float64))


def main():
    dirs = v3_dirs()
    print(f"v3 datasets: {len(dirs)}")
    # pass 1: global [q01,q99]
    acc = {("state", d): [] for d in DIMS} | {("action", d): [] for d in DIMS}
    for v3 in dirs:
        for f in glob.glob(f"{v3}/data/chunk-*/episode_*.parquet"):
            s, a = _sa(pq.read_table(f, columns=["observation.state", "action"]))
            for d in DIMS:
                acc[("state", d)].append(s[:, d])
                acc[("action", d)].append(a[:, d])
    params = {}
    for k, v in acc.items():
        vv = np.concatenate(v)
        q01, q99 = float(np.quantile(vv, 0.01)), float(np.quantile(vv, 0.99))
        span = q99 - q01
        a = (HI - LO) / span if span > 1e-9 else 1.0
        params[k] = (a, LO - a * q01)
        print(f"  {k[0]}[{k[1]}]: q01={q01:.5f} q99={q99:.5f} -> a={a:.4f}")
    # pass 2: remap v3 -> v4 parquet
    n = 0
    for v3 in dirs:
        v4 = v3.replace("/v3/", "/v4/").replace("-v3", "-v4")
        for f in sorted(glob.glob(f"{v3}/data/chunk-*/episode_*.parquet")):
            dst = os.path.join(v4, os.path.relpath(f, v3))
            if not os.path.exists(os.path.dirname(dst)):
                continue
            t = pq.read_table(f)
            s, a = _sa(t)
            for key, arr in (("state", s), ("action", a)):
                for d in DIMS:
                    aa, bb = params[(key, d)]
                    arr[:, d] = np.clip(aa * arr[:, d] + bb, LO, HI)
            st = t.schema.field("observation.state").type
            at = t.schema.field("action").type
            t = t.set_column(t.schema.get_field_index("observation.state"), "observation.state",
                             pa.array([r.tolist() for r in s.astype(np.float32)], type=st))
            t = t.set_column(t.schema.get_field_index("action"), "action",
                             pa.array([r.tolist() for r in a.astype(np.float32)], type=at))
            pq.write_table(t, dst)
            n += 1
        print(f"  remapped {v4.replace(ROOT + '/', '')}")
    print(f"done: {n} v4 parquet remapped to canonical 0-70mm")


if __name__ == "__main__":
    main()
