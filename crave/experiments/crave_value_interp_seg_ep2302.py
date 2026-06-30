"""§3.2 痛点② 配图:CRAVE value 升降可解释(ep2302,0:06–1:17 段)。
铺开过程 → value 上升(POSITIVE);将衣服拎起 → value 下降(NEGATIVE);平稳 → NORMAL。
CRAVE 的升/降与动作对齐 → 可解释。

数据:temp/crave_interp_ep2302_30hz_decoded/_cache.npz(cv=CRAVE value 30Hz, ccls=三档分类);
相机:kai0_base ep2302 视频。纯 CPU。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_value_interp_seg_ep2302.py
输出: crave/docs/visualization/crave_value_interp_seg_ep2302.png
"""
from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np

from crave.config import REPO
from crave.render import setup_mpl

plt = setup_mpl()

CACHE = REPO / "temp/crave_interp_ep2302_30hz_decoded/_cache.npz"
BASE = REPO / "kai0/data/Task_A/kai0_base"
EP = 2302; F0, F1 = 180, 2310            # 0:06–1:17 @30Hz
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}
NAME = {1: "POSITIVE(上升)", 0: "NORMAL(平稳)", -1: "NEGATIVE(下降)"}
ACT = {1: "铺开 → value 上升", 0: "平稳保持 → value 平台", -1: "拎起 → value 下降"}


def longest_run(mask):
    best_s, best_e = 0, 0; i = 0; n = len(mask)
    while i < n:
        if not mask[i]: i += 1; continue
        j = i
        while j < n and mask[j]: j += 1
        if (j - i) > (best_e - best_s): best_s, best_e = i, j
        i = j
    return best_s, best_e


def main():
    z = np.load(CACHE, allow_pickle=True)
    cv = z["cv"].astype(float); ccls = z["ccls"].astype(int)
    seg = slice(F0, F1); cvw = cv[seg]; clw = ccls[seg]; x = (np.arange(F0, F1)) / 30.0
    # 每类取最长连续段,均匀采 3 帧(全局帧号)
    picks = {}
    for c in (1, 0, -1):
        s, e = longest_run((clw == c))
        if e - s < 3: s, e = 0, len(clw)
        idx = np.linspace(s, e - 1, 3).astype(int) + F0
        picks[c] = idx.tolist()
    allidx = sorted(set(sum(picks.values(), [])))
    print(f"ep{EP} 段 [{F0},{F1}] picks: " + " | ".join(f"{NAME[c]}:{picks[c]}" for c in (1, 0, -1)), flush=True)

    # 解码相机帧
    cs = json.load(open(BASE / "meta/info.json"))["chunks_size"]
    vid = BASE / "videos" / f"chunk-{EP//cs:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
    want = set(allidx); frames = {}
    c = av.open(str(vid))
    for i, fr in enumerate(c.decode(video=0)):
        if i in want:
            frames[i] = cv2.resize(fr.to_ndarray(format="rgb24"), (224, 224))
            if len(frames) == len(want): break
    c.close()

    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.35, 0.24, 1, 1, 1], hspace=0.45, wspace=0.06)
    axv = fig.add_subplot(gs[0, :]); axv.plot(x, cvw, color="#333", lw=0.7, alpha=.35)
    for c2 in (-1, 0, 1):
        m = clw == c2; axv.scatter(x[m], cvw[m], s=7, c=[RGB[c2]])
    axv.set_xlim(x[0], x[-1]); axv.set_ylabel("CRAVE value"); axv.grid(alpha=.2); axv.tick_params(labelsize=8)
    axv.set_title(f"CRAVE value 升降可解释 · kai0_base ep{EP}(0:06–1:17):铺开→上升 / 拎起→下降,升降与动作对齐", fontsize=12.5, fontweight="bold")
    # 标注 pos / neg 段
    for c2, lab in ((1, "铺开 → 上升"), (-1, "拎起 → 下降")):
        s, e = longest_run((clw == c2)); xs, xe = x[s], x[min(e, len(x) - 1)]
        axv.axvspan(xs, xe, color=RGB[c2], alpha=.10)
        axv.annotate(lab, xy=((xs + xe) / 2, cvw[s:e].mean() if e > s else cvw[s]), fontsize=9.5,
                     color=tuple(np.array(RGB[c2]) * 0.8), ha="center", fontweight="bold")
    axs = fig.add_subplot(gs[1, :]); axs.imshow(np.array([RGB[c] for c in clw])[None], aspect="auto", extent=[x[0], x[-1], 0, 1])
    axs.set_yticks([]); axs.set_xlabel("秒", fontsize=9); axs.set_ylabel("分类条", fontsize=8, rotation=0, ha="right", va="center"); axs.tick_params(labelsize=8)

    for r, c2 in enumerate((1, 0, -1)):
        for j, gi in enumerate(picks[c2]):
            ax = fig.add_subplot(gs[2 + r, j])
            ax.imshow(frames.get(gi, np.zeros((224, 224, 3), np.uint8))); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(tuple(RGB[c2])); sp.set_linewidth(2.5)
            if j == 0:
                ax.set_ylabel(NAME[c2], fontsize=11, color=tuple(np.array(RGB[c2]) * 0.85), fontweight="bold")
            ax.set_title(f"{gi/30:.0f}s", fontsize=8)
        # 行右侧动作说明
        fig.text(0.91, [0.40, 0.255, 0.11][r], ACT[c2], fontsize=10.5, color=tuple(np.array(RGB[c2]) * 0.85), va="center", fontweight="bold", rotation=0)
    fig.suptitle("", y=1)
    out = REPO / "crave/docs/visualization/crave_value_interp_seg_ep2302.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
