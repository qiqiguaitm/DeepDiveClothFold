#!/usr/bin/env python
"""把 CRAVE 正式标签写成两个 AE 训练数据集(kai0_base + stage_progress_gt 列)。

正式读出:**完整对称 Viterbi(无 smooth) @30Hz native · joint img⊕proprio**
(见 gen_viterbi30_labels.py,标签在 temp/crave_ae_labels/vit30_sym/)。
  - crave_stage_A = 对称 Viterbi 原值(保留真实再抓回落)
  - crave_stage_B = cummax(A) 单调投影(progress-only,只进不退)
**统一以 kai0_base 为底**(干净、不带旧 AE 输出列),symlink meta/+videos/,
data/ parquet 逐 ep 加 stage_progress_gt(native-30Hz,逐帧对齐)。
输出:kai0/data/Task_A/self_built/crave_stage_{A,B}/
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/write_crave_stage_datasets.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
SRC = REPO / "kai0/data/Task_A/kai0_base"
LAB = REPO / "temp/crave_ae_labels/final"   # 收口读出:双锚 Viterbi(无 smooth · 无 norm01),见 gen_anchored_labels.py
OUT = REPO / "kai0/data/Task_A/self_built"
CSQ = 1000


def link_shared(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("meta", "videos"):
        d = dst / sub
        if d.is_symlink() or d.exists():
            continue
        os.symlink((SRC / sub).resolve(), d)


def clip01(v):   # 双锚标签本身即真实 0→1,不再 per-ep norm01(norm01 会掩盖 raw 达顶失败);只做安全裁剪
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def main():
    only = os.environ.get("CRAVE_ONLY")   # 'A'|'B'|None(both) — 只重建一个数据集(避开正在训练的另一个)
    # (proj) 从双锚 Viterbi 标签派生:A=对称原值, B=cummax 单调;双锚已保证 0→1,不 norm01
    jobs = [("A", "crave_stage_A", lambda v: clip01(v)),
            ("B", "crave_stage_B", lambda v: clip01(np.maximum.accumulate(v)))]
    if only:
        jobs = [j for j in jobs if j[0] == only]
    for tag, dsname, proj in jobs:
        dst = OUT / dsname
        link_shared(dst)
        (dst / "data").mkdir(parents=True, exist_ok=True)
        parts = sorted((SRC / "data").glob("chunk-*/episode_*.parquet"))
        n_ok = n_fb = n_mismatch = 0
        for p in parts:
            e = int(p.stem.split("_")[1])
            df = pd.read_parquet(p)
            # 只保留标准 lerobot 列 → schema 逐 ep 一致 (kai0_base 个别 ep 如 ep104 残留
            # prediction/model_prediction 列 → HF CastError → cluster 崩, 见 lerobot schema 教训)
            _STD = ["observation.state", "action", "timestamp", "frame_index",
                    "episode_index", "index", "task_index"]
            df = df[[c for c in _STD if c in df.columns]]
            lab_f = LAB / f"ep{e}.npy"
            if lab_f.exists():
                v = proj(np.load(lab_f).astype(np.float32))   # A:原值 / B:cummax
                if len(v) == len(df):
                    df["stage_progress_gt"] = v.astype(np.float32); n_ok += 1
                else:                                     # 长度不符 → 插值对齐
                    xi = np.linspace(0, 1, len(df)); xa = np.linspace(0, 1, len(v))
                    df["stage_progress_gt"] = np.interp(xi, xa, v).astype(np.float32); n_mismatch += 1
            else:
                df["stage_progress_gt"] = np.linspace(0, 1, len(df)).astype(np.float32)  # 无标签 → 线性兜底(全量已覆盖, 不触发)
                n_fb += 1
            outp = dst / "data" / p.parent.name / p.name
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(outp)
        print(f"[{dsname}] wrote {len(parts)} eps | crave={n_ok} interp={n_mismatch} fallback_manual={n_fb} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
