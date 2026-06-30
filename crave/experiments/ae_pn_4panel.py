"""痛点① 插图(纯 KAI0-AE,无 CRAVE):同一 AE episode 的四联图——
  ① absolute_value 曲线  ② relative_value(relative_advantage)曲线
  ③ positive/negative 二档分类条(sign,无 deadband)  ④ positive/normal/negative 三档分类条(deadband)
直观看:AE 的 relative_value 抖动 → 正负评价逐帧横跳(二档全是红绿交错;三档加 deadband 仍churn)。
只读 advantage_q5 parquet,无需 CRAVE 管线 / 无需 GPU。
Run: PY crave/experiments/ae_pn_4panel.py --eps 2302,2238
输出: crave/docs/visualization/cross_dataset/ae_pn_4panel_ep{E}.png
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from crave.config import REPO
from crave.render import setup_mpl

plt = setup_mpl()

Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
COL3 = np.array([[214, 39, 40], [210, 210, 210], [44, 160, 44]]) / 255.0   # neg / normal / pos
COL2 = np.array([[214, 39, 40], [44, 160, 44]]) / 255.0                     # neg / pos


def cls3(adv, th):
    c = np.ones(len(adv), int); c[adv > th] = 2; c[adv < -th] = 0; return c   # 0 neg /1 normal /2 pos


def render(ep, th):
    cands = glob.glob(str(Q5 / "data" / "**" / f"episode_{ep:06d}.parquet"), recursive=True)
    if not cands:
        print(f"ep{ep} not in advantage_q5, skip", flush=True); return
    df = pd.read_parquet(cands[0])
    av = df["absolute_value"].to_numpy().astype(float)
    ra = np.clip(df["relative_advantage"].to_numpy().astype(float), -1, 1)   # AE 直出的 relative value
    n = len(av); x = np.arange(n)
    c2 = (ra >= 0).astype(int)                                                # 二档:1 pos / 0 neg
    c3 = cls3(ra, th)                                                         # 三档
    p2 = (np.mean(c2 == 0) * 100, np.mean(c2 == 1) * 100)                     # neg, pos
    p3 = (np.mean(c3 == 0) * 100, np.mean(c3 == 1) * 100, np.mean(c3 == 2) * 100)
    # 正负翻转次数(逐帧 churn 强度)
    flips = int(np.sum(np.abs(np.diff(c2))))
    print(f"ep{ep} {n}f | 二档 neg/pos={p2[0]:.0f}/{p2[1]:.0f}%  三档 neg/normal/pos={p3[0]:.0f}/{p3[1]:.0f}/{p3[2]:.0f}%  正负翻转 {flips} 次", flush=True)

    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.2, 0.34, 0.34], hspace=0.5)
    axv = fig.add_subplot(gs[0]); axv.plot(x, av, color="#d62728", lw=1.6); axv.axhline(0, color="k", lw=.4)
    axv.set_ylabel("absolute_value"); axv.set_xlim(0, n); axv.grid(alpha=.25)
    axv.set_title(f"KAI0-AE(监督) · ep{ep}({n}f≈{n/30:.0f}s)· absolute_value / relative_value 及其 P/N 分类", fontsize=12, fontweight="bold")
    axv.text(0.012, 0.9, "① absolute_value(AE 输出的进度值)", transform=axv.transAxes, fontsize=9, color="#8a3b3b", va="top")

    axr = fig.add_subplot(gs[1], sharex=axv); axr.plot(x, ra, color="#1f77b4", lw=1.0); axr.axhline(0, color="k", lw=.5)
    axr.axhline(th, color="gray", ls=":", lw=.7); axr.axhline(-th, color="gray", ls=":", lw=.7)
    axr.fill_between(x, 0, ra, where=ra < 0, color="#d62728", alpha=.14); axr.fill_between(x, 0, ra, where=ra > 0, color="#2ca02c", alpha=.10)
    axr.set_ylabel("relative_value"); axr.grid(alpha=.2)
    axr.text(0.012, 0.93, f"② relative_value(AE 直出,P/N 依据)—— 抖动剧烈,正负翻转 {flips} 次", transform=axr.transAxes, fontsize=9, color="#1a5276", va="top")

    ax2 = fig.add_subplot(gs[2], sharex=axv); ax2.imshow(COL2[c2][None], aspect="auto", extent=[0, n, 0, 1]); ax2.set_yticks([])
    ax2.set_ylabel("③ 二档\npos/neg", fontsize=8.5, rotation=0, ha="right", va="center")
    ax2.set_title(f"③ positive/negative 二档(sign,无 deadband):neg/pos = {p2[0]:.0f}/{p2[1]:.0f}%  —— 红绿满屏交错 = 正负极不稳定", fontsize=9.5, color="#7a2a2a", pad=3)

    ax3 = fig.add_subplot(gs[3], sharex=axv); ax3.imshow(COL3[c3][None], aspect="auto", extent=[0, n, 0, 1]); ax3.set_yticks([])
    ax3.set_ylabel("④ 三档\npos/norm/neg", fontsize=8.5, rotation=0, ha="right", va="center"); ax3.set_xlabel("frame")
    ax3.set_title(f"④ positive/normal/negative 三档(deadband={th}):neg/normal/pos = {p3[0]:.0f}/{p3[1]:.0f}/{p3[2]:.0f}%  —— 加缓冲仍 churn", fontsize=9.5, color="#1a5276", pad=3)

    fig.legend(handles=[Patch(color=COL3[2], label="上升 positive"), Patch(color=COL3[1], label="平台 normal"), Patch(color=COL3[0], label="退步 negative")],
               loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.975, "成功 episode,但 AE 的 relative_value 抖动 → 二档/三档分类条都正负逐帧横跳(痛点①)", ha="center", fontsize=9.5, color="#555", style="italic")
    out = REPO / f"crave/docs/visualization/cross_dataset/ae_pn_4panel_ep{ep}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); print("SAVED", out, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eps", default="2302,2238"); ap.add_argument("--th", type=float, default=0.02)
    a = ap.parse_args()
    for ep in [int(x) for x in a.eps.split(",")]:
        render(ep, a.th)


if __name__ == "__main__":
    main()
