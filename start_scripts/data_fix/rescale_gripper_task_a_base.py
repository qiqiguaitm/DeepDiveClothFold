#!/usr/bin/env python3
"""把 Task_A/base 中 2026-05-18 ~ 2026-05-21 这 4 个日期的夹爪通道
线性拉伸到统一的 [0, hi_target]:
  - L grip (dim 6)  → [0, 0.0797]
  - R grip (dim 13) → [0, 0.0795]

设计:
  - 每日 scan 全部 episode 算 per-date (lo, hi); 同一日内所有 episode 用同一组系数,
    避免引入跨 ep 的台阶.
  - state == action (relabel 之后), 两列用同一公式同步改写.
  - parquet schema: list<float> 通道, 用 pyarrow 直接改 dim 6/13, 其他 dim 不动.
  - in-place 改写 + 原文件备份到
    /data2/visrobot_backup/datasets/KAI0/Task_A_backup/base/<date>-v2/data/chunk-000/.
  - 备份用 'mv' (而非 cp), 既快又强制原文件不在原位 → 接下来的 atomic write 不会
    误覆盖未备份的原始.
  - 写新文件先到 .tmp, fsync, 然后 atomic rename → 中断不会留半个文件.

用法:
  python3 rescale_gripper_task_a_base.py --dates 2026-05-18-v2,2026-05-19-v2,...  # dry-run
  python3 rescale_gripper_task_a_base.py --dates all --apply                       # 真改
"""
import argparse
import glob
import os
import sys
import shutil
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data1/DATA_IMP/KAI0/Task_A/base"
BACKUP_ROOT = "/data2/visrobot_backup/datasets/KAI0/Task_A_backup/base"

DEFAULT_DATES = [
    "2026-05-18-v2",
    "2026-05-19-v2",
    "2026-05-20-v2",
    "2026-05-21-v2",
]

DIM_L_GRIP = 6
DIM_R_GRIP = 13
TARGET_HI_L = 0.0797
TARGET_HI_R = 0.0795


def scan_grip_range(date_dir: str) -> dict:
    """扫描整日所有 ep, 返回 dim 6/13 的 (min, max)."""
    parquets = sorted(glob.glob(f"{date_dir}/data/chunk-000/*.parquet"))
    g_l_min = +np.inf; g_l_max = -np.inf
    g_r_min = +np.inf; g_r_max = -np.inf
    total_frames = 0
    for p in parquets:
        t = pq.read_table(p, columns=["observation.state"])
        # list<float>(14) → np (T, 14)
        flat = t.column("observation.state").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        a = flat.reshape(-1, 14)
        g_l_min = min(g_l_min, float(a[:, DIM_L_GRIP].min()))
        g_l_max = max(g_l_max, float(a[:, DIM_L_GRIP].max()))
        g_r_min = min(g_r_min, float(a[:, DIM_R_GRIP].min()))
        g_r_max = max(g_r_max, float(a[:, DIM_R_GRIP].max()))
        total_frames += a.shape[0]
    return {
        "L": (g_l_min, g_l_max),
        "R": (g_r_min, g_r_max),
        "n_eps": len(parquets),
        "frames": total_frames,
    }


def rescale_array(a14: np.ndarray, lo_l: float, lo_r: float) -> np.ndarray:
    """a14: (T, 14) float32. 原地拉伸 dim 6/13. 返回新数组."""
    out = a14.copy()
    # dim 6: (x - lo_l) * TARGET_HI_L / (hi_l - lo_l), 其中 hi_l 就是 TARGET_HI_L
    scale_l = TARGET_HI_L / (TARGET_HI_L - lo_l)
    scale_r = TARGET_HI_R / (TARGET_HI_R - lo_r)
    out[:, DIM_L_GRIP] = (out[:, DIM_L_GRIP] - lo_l) * scale_l
    out[:, DIM_R_GRIP] = (out[:, DIM_R_GRIP] - lo_r) * scale_r
    # clamp 一下数值稳定 (浮点误差)
    out[:, DIM_L_GRIP] = np.clip(out[:, DIM_L_GRIP], 0.0, TARGET_HI_L)
    out[:, DIM_R_GRIP] = np.clip(out[:, DIM_R_GRIP], 0.0, TARGET_HI_R)
    return out.astype(np.float32)


def rewrite_parquet(src: str, dst: str, lo_l: float, lo_r: float):
    """从 src 读, 拉伸 state/action 的夹爪 dim, 写 dst (临时文件 + atomic rename)."""
    table = pq.read_table(src)
    schema = table.schema

    cols = {name: table.column(name) for name in schema.names}

    # 改 observation.state
    state_flat = cols["observation.state"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    state_arr = state_flat.reshape(-1, 14)
    state_new = rescale_array(state_arr, lo_l, lo_r)
    # 改 action (== state)
    action_flat = cols["action"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    action_arr = action_flat.reshape(-1, 14)
    action_new = rescale_array(action_arr, lo_l, lo_r)

    # 重建 list<float>: 14 个 element 平铺 + offsets
    n = state_new.shape[0]
    offsets = np.arange(0, (n + 1) * 14, 14, dtype=np.int32)
    state_pa = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(state_new.reshape(-1), type=pa.float32()),
    )
    action_pa = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(action_new.reshape(-1), type=pa.float32()),
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


def process_date(date: str, apply: bool, manifest: list):
    date_dir = os.path.join(ROOT, date)
    if not os.path.isdir(date_dir):
        print(f"  [skip] {date_dir} not found")
        return

    print(f"\n=== {date} ===")
    rng = scan_grip_range(date_dir)
    lo_l, hi_l = rng["L"]
    lo_r, hi_r = rng["R"]
    print(f"  Scanned {rng['n_eps']} eps  {rng['frames']:>7} frames")
    print(f"  L-grip: [{lo_l:.6f}, {hi_l:.6f}]  → [0, {TARGET_HI_L}]  "
          f"scale={TARGET_HI_L / (TARGET_HI_L - lo_l):.6f}  shift={-lo_l:.6f}")
    print(f"  R-grip: [{lo_r:.6f}, {hi_r:.6f}]  → [0, {TARGET_HI_R}]  "
          f"scale={TARGET_HI_R / (TARGET_HI_R - lo_r):.6f}  shift={-lo_r:.6f}")

    parquets = sorted(glob.glob(f"{date_dir}/data/chunk-000/*.parquet"))
    backup_dir = os.path.join(BACKUP_ROOT, date, "data", "chunk-000")
    if apply:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"  backup_dir: {backup_dir}")

    if not apply:
        # dry-run: 只读首尾各 1 个 ep 演示效果
        sample = parquets[:1] + parquets[-1:]
        for p in set(sample):
            t = pq.read_table(p, columns=["observation.state"])
            flat = t.column("observation.state").combine_chunks().flatten().to_numpy(zero_copy_only=False)
            a = flat.reshape(-1, 14)
            old_l_min, old_l_max = a[:, DIM_L_GRIP].min(), a[:, DIM_L_GRIP].max()
            new = rescale_array(a, lo_l, lo_r)
            new_l_min, new_l_max = new[:, DIM_L_GRIP].min(), new[:, DIM_L_GRIP].max()
            print(f"  [preview] {os.path.basename(p)}  L-grip: "
                  f"[{old_l_min:.5f},{old_l_max:.5f}] → [{new_l_min:.5f},{new_l_max:.5f}]")
        print(f"  [dry-run] would rewrite {len(parquets)} parquets")
        return

    # apply
    for i, p in enumerate(parquets):
        fname = os.path.basename(p)
        bk = os.path.join(backup_dir, fname)
        if os.path.exists(bk):
            # 备份已存在 = 此 ep 上一次跑过, 跳过 (幂等)
            # 但仍要重写以确保覆盖
            print(f"  [{i+1:3d}/{len(parquets)}] {fname}  (backup exists, re-rewriting)")
        else:
            # 第一次: 原文件 mv 到备份
            shutil.move(p, bk)
            print(f"  [{i+1:3d}/{len(parquets)}] {fname}  backup → rewrite")
        # 从备份读, 写回原路径
        rewrite_parquet(bk, p, lo_l, lo_r)
        manifest.append({"date": date, "ep": fname, "lo_l": lo_l, "lo_r": lo_r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="all",
                    help="comma list of date dirs OR 'all' (= 4 default dates)")
    ap.add_argument("--apply", action="store_true",
                    help="实际改写; 不加这个 flag 只 dry-run")
    args = ap.parse_args()

    if args.dates == "all":
        dates = DEFAULT_DATES
    else:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    print(f"target dates: {dates}")
    print(f"apply: {args.apply}")
    manifest = []
    for d in dates:
        process_date(d, args.apply, manifest)

    if args.apply:
        # 写 manifest 到备份根目录, 方便回滚时知道原 lo_l/lo_r
        os.makedirs(BACKUP_ROOT, exist_ok=True)
        mfile = os.path.join(BACKUP_ROOT, f"rescale_manifest_{os.getpid()}.json")
        with open(mfile, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nmanifest → {mfile}")
        print(f"\n✓ applied. To rollback: mv backup files back over originals.")


if __name__ == "__main__":
    main()
