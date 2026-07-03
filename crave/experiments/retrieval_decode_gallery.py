#!/usr/bin/env python
"""最佳解码器(检索=最近真实帧)的全分辨率画廊,跨叠衣进度展示效果 + top-3 稳健性。

每行一个 held-out query(按 Pord 从早到晚),列 = [真实 query | 检索 top1(最佳解码) | top2 | top3],
全部来自不同 episode,标注 cos。用来直观判断"最佳解码图效果如何"。
Run: REPO=/home/tim/workspace/deepdive_kai0 PYTHONPATH=crave/src:lmwm/src \
  /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/retrieval_decode_gallery.py
输出: crave/docs/visualization/decoder_benchmark/retrieval_decode_gallery.png
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from crave.render import setup_mpl
from lmwm.retrieval_decoder import LatentRetrievalDecoder

plt = setup_mpl()
REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FEAT = REPO / "temp/crave_full_dinov3h"
DATA = REPO / "kai0/data/Task_A/kai0_base"
OUT = REPO / "crave/docs/visualization/decoder_benchmark"
rng = np.random.RandomState(3)
NROW = 8


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    rdec = LatentRetrievalDecoder(feature_dir=FEAT, dataset_root=DATA, device="cuda", res=None)
    mz = np.load(FEAT / "milestones_uniform_dinov3h.npz")
    proto = l2(mz["C"].astype(np.float32))
    pord = mz["Pord"].astype(np.float32)

    # sample frames, assign to nearest milestone -> progress, pick NROW spanning early->late
    samp = rng.choice(len(rdec.feat), 4000, replace=False)
    prog = pord[(rdec.feat[samp] @ proto.T).argmax(1)]
    qs = [samp[np.argmin(np.abs(prog - t))] for t in np.linspace(0.03, 0.95, NROW)]

    fig, axs = plt.subplots(NROW, 4, figsize=(12, 3 * NROW))
    col_titles = ["真实 query", "检索 top1 (最佳解码)", "top2", "top3"]
    for r, g in enumerate(qs):
        qp = rdec.feat[g:g + 1]
        idx, cos = rdec.retrieve(qp, topk=12, exclude_episode=int(rdec.E[g]))
        # keep top-3 from distinct episodes
        picks, eps = [], set()
        for j in range(idx.shape[1]):
            gi = int(idx[0, j]); ep = int(rdec.E[gi])
            if ep not in eps:
                picks.append((gi, float(cos[0, j]))); eps.add(ep)
            if len(picks) == 3:
                break
        p = pord[(rdec.feat[g] @ proto.T).argmax()]
        imgs = [(rdec.frame(int(g)), f"进度≈{p:.2f}")] + [(rdec.frame(gi), f"cos={c:.2f}") for gi, c in picks]
        for c, (im, cap) in enumerate(imgs):
            ax = axs[r, c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(cap, fontsize=10)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"#{r + 1}", fontsize=11, rotation=0, ha="right", va="center")
    fig.suptitle("最佳解码器 = 检索(最近真实帧)· 全分辨率 · 跨叠衣进度(早→晚)· 均取自不同 episode",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT / "retrieval_decode_gallery.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("SAVED", out)


if __name__ == "__main__":
    main()
