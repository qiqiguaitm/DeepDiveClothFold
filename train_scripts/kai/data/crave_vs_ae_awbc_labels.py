"""AWBC 数据处理层面对比: CRAVE 产的 advantage 标签 vs kai0-AE 产的 AWBC 标签(全 1117 dagger ep)。
CRAVE value = temp/mv_value_full(零标签零训练已算); AE = A_smooth800_dagger_all_awbc(absolute_advantage+task_index)。
看: ① CRAVE 天然 neg 信号够不够(AWBC 要负优势下权坏转移) ② 离散标签与 AE 的一致度 ③ 成本。
"""
import glob, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
MV = REPO / "temp/mv_value_full"; AW = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc"
csAW = json.load(open(AW / "meta/info.json"))["chunks_size"]; W = 50


def adv(v, w=W):
    a = np.zeros(len(v))
    for i in range(len(v)): a[i] = v[min(i + w, len(v) - 1)] - v[i]
    return np.clip(a, -1, 1)


eps = sorted(int(p.stem[2:]) for p in MV.glob("ep*.npy"))
crave_adv_all, ae_adv_all, ae_ti_all = [], [], []
for e in eps:
    cv = np.load(MV / f"ep{e}.npy").astype(float)
    pq = AW / "data" / f"chunk-{e//csAW:03d}" / f"episode_{e:06d}.parquet"
    if not pq.exists(): continue
    d = pd.read_parquet(pq, columns=["absolute_advantage", "task_index"])
    aa = d["absolute_advantage"].to_numpy().astype(float); ti = d["task_index"].to_numpy()
    n = min(len(cv), len(aa))
    crave_adv_all.append(adv(cv[:n])); ae_adv_all.append(aa[:n]); ae_ti_all.append(ti[:n])
ca = np.concatenate(crave_adv_all); aea = np.concatenate(ae_adv_all); aeti = np.concatenate(ae_ti_all)
N = len(ca)
ae_neg = float((aeti == 0).mean())   # AWBC 实际 neg 比例(task_index==0)
print(f"全数据 {len(eps)}ep {N}帧", flush=True)
print(f"AE(现 AWBC 标签): neg(task_index=0) 占比 {ae_neg:.1%}; AE absolute_advantage<0 占比 {(aea<0).mean():.1%}", flush=True)
print(f"CRAVE advantage(零训练): <0 占比 {(ca<0).mean():.1%}; 均值{ca.mean():.3f} std{ca.std():.3f}", flush=True)

# 分位匹配: 把 CRAVE advantage 按 AE 的 neg 比例阈值化 → CRAVE task_index(单变量对照, 锁 neg 比例)
thr = np.quantile(ca, ae_neg); crave_ti = (ca > thr).astype(int)
agree = float((crave_ti == aeti).mean())
# 在 AE 标 neg 的帧里, CRAVE 也标 neg 的比例(neg 信号是否对得上)
both_neg = float(((crave_ti == 0) & (aeti == 0)).sum() / max(1, (aeti == 0).sum()))
print(f"分位匹配({ae_neg:.0%} neg)后 CRAVE-task_index vs AE-task_index: 一致 {agree:.1%}; "
      f"AE 标 neg 的帧 CRAVE 也 neg 占 {both_neg:.1%}", flush=True)

# 天然 neg 够不够(不分位匹配, CRAVE 自己的负优势)
print(f"→ CRAVE 天然 neg {(ca<0).mean():.0%} vs AWBC 需要 ~{ae_neg:.0%}: "
      f"{'天然够' if (ca<0).mean()>=ae_neg*0.8 else '天然偏少, 需分位匹配补'}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].hist(ca, bins=60, alpha=.6, color="#2ca02c", label=f"CRAVE adv (neg {(ca<0).mean():.0%})", density=True)
ax[0].hist(aea, bins=60, alpha=.5, color="#d62728", label=f"AE absolute_advantage (neg {(aea<0).mean():.0%})", density=True)
ax[0].axvline(0, color="k", lw=.7); ax[0].axvline(thr, color="#2ca02c", ls="--", lw=1, label=f"CRAVE 分位阈 {thr:.2f}")
ax[0].set_title(f"advantage 分布(全 {len(eps)} dagger ep)", fontsize=10); ax[0].set_xlabel("advantage"); ax[0].legend(fontsize=8); ax[0].set_xlim(-1, 1)
labels = ["一致", "AE=pos\nCRAVE=neg", "AE=neg\nCRAVE=pos"]
vals = [agree, float(((crave_ti == 0) & (aeti == 1)).mean()), float(((crave_ti == 1) & (aeti == 0)).mean())]
ax[1].bar(labels, vals, color=["#2ca02c", "#ff7f0e", "#d62728"], alpha=.8)
for i, v in enumerate(vals): ax[1].text(i, v + .01, f"{v:.0%}", ha="center", fontsize=9)
ax[1].set_title(f"离散标签一致性(分位匹配 {ae_neg:.0%} neg)", fontsize=10); ax[1].set_ylabel("帧占比"); ax[1].set_ylim(0, 1)
fig.suptitle("AWBC 数据处理: CRAVE(零训练) vs kai0-AE(监督) 的 advantage 标签对比", fontsize=12)
fig.tight_layout(); out = REPO / "docs/visualization/cross_episode_recurrence_value/crave_vs_ae_awbc_labels.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); print("AWBC_LABELS_DONE", flush=True)
