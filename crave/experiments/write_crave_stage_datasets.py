#!/usr/bin/env python
"""把两套 CRAVE 标签写成两个 AE 训练数据集(替换 stage_progress_gt 列)。

基于 advantage_q5(已含 stage_progress_gt 列 + 完整 meta),symlink meta/+videos/,
data/ parquet 逐 ep 把 stage_progress_gt 覆盖为 CRAVE 标签(native-fps)。
缺 CRAVE 标签的 ep 回退保留原(人工)列。
输出:kai0/data/Task_A/self_built/crave_stage_{A,B}/
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/write_crave_stage_datasets.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
SRC = REPO / "kai0/data/Task_A/self_built/advantage_q5"
LAB = REPO / "temp/crave_ae_labels"
OUT = REPO / "kai0/data/Task_A/self_built"
CSQ = 1000


def link_shared(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("meta", "videos"):
        d = dst / sub
        if d.is_symlink() or d.exists():
            continue
        os.symlink((SRC / sub).resolve(), d)


def main():
    for method, dsname in [("anchor", "crave_stage_A"), ("viterbi", "crave_stage_B")]:
        dst = OUT / dsname
        link_shared(dst)
        (dst / "data").mkdir(parents=True, exist_ok=True)
        parts = sorted((SRC / "data").glob("chunk-*/episode_*.parquet"))
        n_ok = n_fb = n_mismatch = 0
        for p in parts:
            e = int(p.stem.split("_")[1])
            df = pd.read_parquet(p)
            lab_f = LAB / method / f"ep{e}.npy"
            if lab_f.exists():
                v = np.load(lab_f)
                if len(v) == len(df):
                    df["stage_progress_gt"] = v.astype(np.float32); n_ok += 1
                else:                                     # 长度不符 → 插值对齐
                    xi = np.linspace(0, 1, len(df)); xa = np.linspace(0, 1, len(v))
                    df["stage_progress_gt"] = np.interp(xi, xa, v).astype(np.float32); n_mismatch += 1
            else:
                n_fb += 1                                  # 无 CRAVE 标签 → 保留原(人工)
            outp = dst / "data" / p.parent.name / p.name
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(outp)
        print(f"[{dsname}] wrote {len(parts)} eps | crave={n_ok} interp={n_mismatch} fallback_manual={n_fb} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
