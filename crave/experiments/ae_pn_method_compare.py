"""P/N 评价方法对比图(KAI0-AE):同一 absolute_value,对比两种分类条 —
  (a) 旧/错:Δabsolute_value 差分派生的 advantage → P/N
  (b) 对:AE 模型直出的 relative_advantage → P/N
只读 advantage_q5 parquet(无需 CRAVE 管线)。Run: PY crave/experiments/ae_pn_method_compare.py --ep 2303
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import advantage
plt = setup_mpl()

ap = argparse.ArgumentParser(); ap.add_argument("--ep", type=int, default=2303); ap.add_argument("--th", type=float, default=0.02); ap.add_argument("--w", type=int, default=50)
a = ap.parse_args()
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
cands = glob.glob(str(Q5 / "data" / "**" / f"episode_{a.ep:06d}.parquet"), recursive=True)
if not cands:
    a.ep = 2302; cands = glob.glob(str(Q5 / "data" / "**" / f"episode_{a.ep:06d}.parquet"), recursive=True)
df = pd.read_parquet(cands[0]); print(f"ep{a.ep}: {cands[0]}", flush=True)
av = df["absolute_value"].to_numpy().astype(float)
ra = np.clip(df["relative_advantage"].to_numpy().astype(float), -1, 1)
ad = np.clip(advantage(av, a.w), -1, 1)   # 差分派生
n = len(av); x = np.arange(n)


def cls(adv, th):
    c = np.ones(len(adv), int)  # 1=normal
    c[adv > th] = 2; c[adv < -th] = 0  # 2=pos, 0=neg
    return c


COL = np.array([[214, 39, 40], [210, 210, 210], [44, 160, 44]]) / 255.0  # neg/normal/pos
cd, cr = cls(ad, a.th), cls(ra, a.th)
def pct(c): return (np.mean(c == 0) * 100, np.mean(c == 1) * 100, np.mean(c == 2) * 100)
pd_, pr_ = pct(cd), pct(cr)

fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(5, 1, height_ratios=[1.3, 1.0, 0.28, 1.0, 0.28], hspace=0.45)
axv = fig.add_subplot(gs[0]); axv.plot(x, av, color="#d62728", lw=1.6); axv.set_ylabel("absolute_value")
axv.set_xlim(0, n); axv.grid(alpha=.25); axv.axhline(0, color="k", lw=.4)
axv.set_title(f"KAI0-AE · ep{a.ep}({n}f)· 同一 absolute_value,两种 P/N 评价方法对比(deadband={a.th})", fontsize=12, fontweight="bold")

axd = fig.add_subplot(gs[1], sharex=axv); axd.plot(x, ad, color="#8a6d3b", lw=1.1); axd.axhline(0, color="k", lw=.5)
axd.axhline(a.th, color="gray", ls=":", lw=.7); axd.axhline(-a.th, color="gray", ls=":", lw=.7)
axd.fill_between(x, 0, ad, where=ad < 0, color="#d62728", alpha=.12); axd.set_ylabel("advantage")
axd.set_title(f"(a) 旧/错:Δabsolute_value 差分派生  →  neg/normal/pos = {pd_[0]:.0f}/{pd_[1]:.0f}/{pd_[2]:.0f}%", fontsize=10, color="#8a3b3b")
axd.grid(alpha=.2)
axbd = fig.add_subplot(gs[2], sharex=axv); axbd.imshow(COL[cd][None], aspect="auto", extent=[0, n, 0, 1]); axbd.set_yticks([]); axbd.set_ylabel("分类条", fontsize=9, rotation=0, ha="right", va="center")

axr = fig.add_subplot(gs[3], sharex=axv); axr.plot(x, ra, color="#1f77b4", lw=1.1); axr.axhline(0, color="k", lw=.5)
axr.axhline(a.th, color="gray", ls=":", lw=.7); axr.axhline(-a.th, color="gray", ls=":", lw=.7)
axr.fill_between(x, 0, ra, where=ra < 0, color="#1f77b4", alpha=.12); axr.set_ylabel("relative_advantage")
axr.set_title(f"(b) 正确:AE 模型直出 relative_advantage  →  neg/normal/pos = {pr_[0]:.0f}/{pr_[1]:.0f}/{pr_[2]:.0f}%", fontsize=10, color="#1a5276")
axr.grid(alpha=.2)
axbr = fig.add_subplot(gs[4], sharex=axv); axbr.imshow(COL[cr][None], aspect="auto", extent=[0, n, 0, 1]); axbr.set_yticks([]); axbr.set_xlabel("frame"); axbr.set_ylabel("分类条", fontsize=9, rotation=0, ha="right", va="center")

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL[2], label="上升 pos"), Patch(color=COL[1], label="平台 normal"), Patch(color=COL[0], label="退步 neg")],
           loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
corr = np.corrcoef(ad, ra)[0, 1]
fig.text(0.5, 0.965, f"两法逐帧 corr={corr:.2f} —— 聚合占比相近但分类条逐帧明显不同;relative_advantage 更弥散(真实模型输出)", ha="center", fontsize=9.5, color="#555", style="italic")
out = REPO / f"crave/docs/visualization/cross_dataset/ae_pn_method_compare_ep{a.ep}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
print(f"差分法 neg/normal/pos={pd_[0]:.0f}/{pd_[1]:.0f}/{pd_[2]:.0f}  直出法={pr_[0]:.0f}/{pr_[1]:.0f}/{pr_[2]:.0f}  corr={corr:.2f}", flush=True)
