#!/usr/bin/env python3
"""5-18 ~ 5-22 右爪 (dim 13) 非线性平滑压缩.

问题:
  这 5 天右爪 master 传感器零点偏移 ~5-10mm. 数据中"完全闭合"档 (0-3mm) 只占 ~50%,
  本该在 (0-3mm) 的帧大量散落在 (3-10mm). 模型学到"右爪闭合 = 留 5mm 缝", 推理时
  抓 T-shirt 薄面料容易松手.

修复:
  对 dim 13 应用 C^∞ 平滑单调压缩 f: [0, 80mm] → [0, 80mm]
    f(R) = R * g(R)
    g(R) = scale_lo + (1 - scale_lo) * sigmoid((R - center) / width)
  参数 (B "中和"): scale_lo=0.25, center=0.020m, width=0.006m
  效果:
    f(0)    = 0          (端点保持)
    f(5)    ≈ 1.5 mm     (压缩)
    f(10)   ≈ 3.7 mm     (满足"3-10mm → 0-3mm 附近"目标, 不过修)
    f(15)   ≈ 7.2 mm
    f(20)   ≈ 12.5 mm
    f(26)   ≈ 21.0 mm    (斜率峰在此, ~1.44×)
    f(30)   ≈ 26.4 mm
    f(40)   ≈ 39.0 mm    (基本 identity)
    f(80)   ≈ 80         (端点保持)
  整个函数 C^∞ 平滑, 不引入时序断点. 选 B 优势: 过渡区斜率峰 1.44× (vs A 默认 1.66×),
  长尾 |ΔR| 长尾放大效应更小; tight% 修复到 ~2-3% 而非过修到 0.5%.

只动 dim 13 (右爪). dim 6 (左爪) 检查显示无显著漂移, 不修改.
state 与 action 同步改写 (relabel 后两者相等).

写入:
  - 文件 in-place 覆盖 (atomic .tmp + os.replace)
  - 原文件 mv 到 /data2/visrobot_backup/datasets/KAI0/Task_A_backup_grip_R_nl/...
  - manifest JSON 记录每个 ep 的变换参数

用法:
  dry-run: python3 rescale_gripper_R_nonlinear.py
  实际改: python3 rescale_gripper_R_nonlinear.py --apply
"""
import argparse
import glob
import json
import os
import shutil
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data1/DATA_IMP/KAI0/Task_A/base"
BACKUP_ROOT = "/data2/visrobot_backup/datasets/KAI0/Task_A_backup_grip_R_nl/base"

DEFAULT_DATES = [
    "2026-05-18-v2",
    "2026-05-19-v2",
    "2026-05-20-v2",
    "2026-05-21-v2",
    "2026-05-22-v2",
]

DIM_R_GRIP = 13

# Smooth compression params (units: meters) — "B 中和" preset
SCALE_LO = 0.25
CENTER_M = 0.020
WIDTH_M  = 0.006


def smooth_compress(r_m: np.ndarray) -> np.ndarray:
    """C^∞ smooth monotonic compression on r in [0, 0.08] meters."""
    # sigmoid blend
    s = 1.0 / (1.0 + np.exp(-(r_m - CENTER_M) / WIDTH_M))
    g = SCALE_LO + (1.0 - SCALE_LO) * s
    out = r_m * g
    # clamp 防浮点越界 (function 单调 f(0)=0, f(0.08)<0.08 严格成立)
    return np.clip(out, 0.0, 0.08).astype(np.float32)


def rewrite_parquet(src: str, dst: str):
    """读 src, 改 state/action dim 13, atomic 写 dst."""
    table = pq.read_table(src)
    schema = table.schema
    cols = {name: table.column(name) for name in schema.names}

    state_flat = cols["observation.state"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    state_arr = state_flat.reshape(-1, 14).astype(np.float32, copy=True)
    state_arr[:, DIM_R_GRIP] = smooth_compress(state_arr[:, DIM_R_GRIP])

    action_flat = cols["action"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    action_arr = action_flat.reshape(-1, 14).astype(np.float32, copy=True)
    action_arr[:, DIM_R_GRIP] = smooth_compress(action_arr[:, DIM_R_GRIP])

    n = state_arr.shape[0]
    offsets = np.arange(0, (n + 1) * 14, 14, dtype=np.int32)
    state_pa = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(state_arr.reshape(-1), type=pa.float32()),
    )
    action_pa = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(action_arr.reshape(-1), type=pa.float32()),
    )

    new_cols = {}
    for name in schema.names:
        if name == "observation.state":
            new_cols[name] = state_pa
        elif name == "action":
            new_cols[name] = action_pa
        else:
            new_cols[name] = cols[name]
    new_table = pa.table(new_cols, schema=schema)

    tmp = dst + ".tmp"
    pq.write_table(new_table, tmp, compression="snappy")
    os.replace(tmp, dst)


def stats_R(date_dir: str, label: str):
    parquets = sorted(glob.glob(f"{date_dir}/data/chunk-000/*.parquet"))
    all_R = []
    for p in parquets:
        t = pq.read_table(p, columns=["observation.state"])
        flat = t.column("observation.state").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        a = flat.reshape(-1, 14)
        all_R.append(a[:, DIM_R_GRIP])
    R = np.concatenate(all_R) * 1000.0   # mm
    bins = [0, 3, 10, 25, 50, 70, 80.5]
    counts = np.histogram(R, bins=bins)[0]
    pct = counts / R.size * 100
    print(f"  {label:<8}  full(0-3): {pct[0]:5.2f}%   tight(3-10): {pct[1]:5.2f}%   "
          f"slight(10-25): {pct[2]:5.2f}%   med(25-50): {pct[3]:5.2f}%   "
          f"open(50-70): {pct[4]:5.2f}%   full_open(70-80): {pct[5]:5.2f}%")
    return pct


def process_date(date: str, apply: bool, manifest: list):
    date_dir = os.path.join(ROOT, date)
    if not os.path.isdir(date_dir):
        print(f"  [skip] {date_dir} not found"); return
    print(f"\n=== {date} ===")
    pct_before = stats_R(date_dir, "BEFORE")

    if not apply:
        # dry-run: 在内存里算 AFTER, 不落盘
        parquets = sorted(glob.glob(f"{date_dir}/data/chunk-000/*.parquet"))
        all_R_new = []
        for p in parquets:
            t = pq.read_table(p, columns=["observation.state"])
            flat = t.column("observation.state").combine_chunks().flatten().to_numpy(zero_copy_only=False)
            a = flat.reshape(-1, 14)
            new_R = smooth_compress(a[:, DIM_R_GRIP].astype(np.float32))
            all_R_new.append(new_R)
        Rn = np.concatenate(all_R_new) * 1000.0
        bins = [0, 3, 10, 25, 50, 70, 80.5]
        counts = np.histogram(Rn, bins=bins)[0]
        pct = counts / Rn.size * 100
        print(f"  AFTER     full(0-3): {pct[0]:5.2f}%   tight(3-10): {pct[1]:5.2f}%   "
              f"slight(10-25): {pct[2]:5.2f}%   med(25-50): {pct[3]:5.2f}%   "
              f"open(50-70): {pct[4]:5.2f}%   full_open(70-80): {pct[5]:5.2f}%")
        print(f"  [dry-run] would rewrite {len(parquets)} parquets")
        return

    # apply: backup → rewrite
    backup_dir = os.path.join(BACKUP_ROOT, date, "data", "chunk-000")
    os.makedirs(backup_dir, exist_ok=True)
    parquets = sorted(glob.glob(f"{date_dir}/data/chunk-000/*.parquet"))
    for i, p in enumerate(parquets):
        fname = os.path.basename(p)
        bk = os.path.join(backup_dir, fname)
        if os.path.exists(bk):
            print(f"  [{i+1:3d}/{len(parquets)}] {fname}  backup exists, re-rewriting")
        else:
            shutil.move(p, bk)
            print(f"  [{i+1:3d}/{len(parquets)}] {fname}  backup → rewrite")
        rewrite_parquet(bk, p)
        manifest.append({
            "date": date, "ep": fname,
            "func": "smooth_compress",
            "params": {"scale_lo": SCALE_LO, "center_m": CENTER_M, "width_m": WIDTH_M},
            "dim": DIM_R_GRIP,
        })

    pct_after = stats_R(date_dir, "AFTER")
    print(f"  Δ full(0-3): {pct_after[0]-pct_before[0]:+.2f}%   "
          f"Δ tight(3-10): {pct_after[1]-pct_before[1]:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="all", help="逗号分隔日期, 或 'all' = 5-18~5-22")
    ap.add_argument("--apply", action="store_true", help="实际改写 (默认 dry-run)")
    args = ap.parse_args()

    dates = DEFAULT_DATES if args.dates == "all" else [d.strip() for d in args.dates.split(",") if d.strip()]
    print(f"Target dates: {dates}")
    print(f"Function: f(R) = R * (scale_lo + (1-scale_lo) * sigmoid((R-{CENTER_M*1000:.0f}mm)/{WIDTH_M*1000:.0f}mm))")
    print(f"          scale_lo = {SCALE_LO}")
    print(f"Reference points: f(5)={smooth_compress(np.float32(0.005))*1000:.2f}mm  "
          f"f(10)={smooth_compress(np.float32(0.010))*1000:.2f}mm  "
          f"f(15)={smooth_compress(np.float32(0.015))*1000:.2f}mm  "
          f"f(20)={smooth_compress(np.float32(0.020))*1000:.2f}mm  "
          f"f(25)={smooth_compress(np.float32(0.025))*1000:.2f}mm  "
          f"f(30)={smooth_compress(np.float32(0.030))*1000:.2f}mm  "
          f"f(40)={smooth_compress(np.float32(0.040))*1000:.2f}mm  "
          f"f(80)={smooth_compress(np.float32(0.080))*1000:.2f}mm")
    print(f"Apply: {args.apply}")

    manifest = []
    for d in dates:
        process_date(d, args.apply, manifest)

    if args.apply:
        os.makedirs(BACKUP_ROOT, exist_ok=True)
        mf = os.path.join(BACKUP_ROOT, f"rescale_R_nl_manifest_{os.getpid()}.json")
        with open(mf, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nmanifest → {mf}")


if __name__ == "__main__":
    main()
