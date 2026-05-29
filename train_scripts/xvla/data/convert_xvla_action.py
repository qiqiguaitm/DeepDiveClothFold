"""Pre-convert XVLA-Soft-Fold hdf5 action (14D joint) → 20D EE6D (per episode).

Output: {hdf5_dir}/action_ee6d_cache/{stage}/{ep_id}.npy (shape (T, 20))
Reads from hdf5 episode files, FK convert, save numpy cache.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

_FK = None
def get_fk():
    global _FK
    if _FK is None:
        _FK = C_PiperForwardKinematics(0x01)
    return _FK

def j14_to_ee6d(q14: np.ndarray) -> np.ndarray:
    fk = get_fk()
    out = np.zeros(20, dtype=np.float32)
    for arm in range(2):
        q6 = q14[arm*7 : arm*7+6]
        gripper = q14[arm*7 + 6]
        result = fk.CalFK(list(q6.astype(np.float64)))
        ee = result[-1]
        xyz_m = np.array(ee[:3], dtype=np.float32) / 1000.0
        rpy_deg = np.array(ee[3:], dtype=np.float32)
        R = Rotation.from_euler("xyz", np.radians(rpy_deg)).as_matrix()
        rot6d = R[:, :2].T.flatten().astype(np.float32)
        out[arm*10 : arm*10+3] = xyz_m
        out[arm*10+3 : arm*10+9] = rot6d
        out[arm*10+9] = gripper
    return out

def convert_one(hdf5_path: Path, out_dir: Path) -> int:
    out_path = out_dir / (hdf5_path.parent.name + "__" + hdf5_path.stem + ".npy")
    if out_path.exists():
        return -1
    with h5py.File(hdf5_path, "r") as f:
        action = f["action"][:]  # (T, 14)
    T = action.shape[0]
    ee6d = np.zeros((T, 20), dtype=np.float32)
    for i in range(T):
        ee6d[i] = j14_to_ee6d(action[i])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, ee6d)
    return T

def main(in_root, out_dir, workers):
    in_root = Path(in_root)
    out_dir = Path(out_dir)
    hdf5_files = sorted(in_root.rglob("episode_*.hdf5"))
    print(f"{len(hdf5_files)} hdf5 files to convert")
    done, total = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(convert_one, f, out_dir): f for f in hdf5_files}
        for fut in as_completed(futures):
            n = fut.result()
            done += 1
            if n > 0:
                total += n
            if done % 100 == 0:
                print(f"  {done}/{len(hdf5_files)} done, {total} frames cached")
    print(f"DONE: {done} files, {total} frames")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold")
    ap.add_argument("--out_dir", default="/data/shared/ubuntu/workspace/dataset_ee6d/xvla_soft_fold_action_cache")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    main(args.in_root, args.out_dir, args.workers)
