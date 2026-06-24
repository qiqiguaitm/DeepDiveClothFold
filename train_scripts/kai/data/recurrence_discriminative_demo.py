#!/usr/bin/env python
"""鉴别性实验(demo 数据版, 替代 rollout): 证明 V_milestone 是状态触发而非时间驱动。
① 拼接测试: 两条 held-out episode 首尾拼接 = 人工"两轮叠衣" → windowed value 应在
   拼接点回落归零再爬升(可控复现 autonomy rollout 的多轮结构)。
② 倒放测试: 同一 episode 帧序倒放 → value 曲线应镜像反转(纯图像状态函数, 与帧索引无关)。
数据: smooth800(vis demo); milestone 用 V0 50-ep 协议(held-out eps 不在挖掘集内)。
用法: python recurrence_discriminative_demo.py
产物: temp/recurrence_v0/discriminative_demo.png
"""
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROBE = REPO / "temp/recurrence_v0/embeddings.npz"          # 挖掘集 = V0 的 50 ep
CACHE = REPO / "temp/tcc_smooth800/feat_cache"               # 全量特征缓存 (集群提取)
DSMETA = REPO / "kai0/data/Task_A/self_built/A_new_smooth_800/base/meta/episodes.jsonl"
W = 90   # windowed 窗口: 90 帧@3Hz = 30s

# ---- milestone 挖掘 (V0 精确协议, seed 确定性) ----
z = np.load(PROBE)
feats, ep_ids, tnorm = z["feats"], z["ep_ids"], z["tnorm"]
mined_eps = set(ep_ids.tolist())
km = KMeans(n_clusters=48, n_init=4, random_state=0).fit(feats)
n_ep = len(mined_eps)
cov = np.array([len(set(ep_ids[km.labels_ == c].tolist())) / n_ep for c in range(48)])
tpos = np.array([tnorm[km.labels_ == c].mean() for c in range(48)])
ms = sorted(np.argsort(cov)[-10:].tolist(), key=lambda c: tpos[c])
idx = {c: i + 1 for i, c in enumerate(ms)}
print("milestones:", [(int(c), f"{cov[c]:.0%}") for c in ms])

# ---- held-out episodes (不在挖掘 50 ep 内, 有缓存) ----
all_eps = [json.loads(l)["episode_index"] for l in open(DSMETA)]
held = [e for e in all_eps if e not in mined_eps and (CACHE / f"ep{e}.npz").exists()]
rng = np.random.RandomState(7)
pick = sorted(rng.choice(held, 4, replace=False).tolist())
print("held-out eps:", pick)

def level(ep):
    """逐帧 milestone 等级 raw(t) (0=非milestone簇)"""
    f = np.load(CACHE / f"ep{ep}.npz")["f"]
    lab = km.predict(f)
    return np.array([idx.get(c, 0) for c in lab], dtype=float)

def vwin(raw):
    return np.array([raw[max(0, i - W):i + 1].max() for i in range(len(raw))]) / len(ms)

def vmono(raw):
    return np.maximum.accumulate(raw) / len(ms)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(15, 7))

# ① 拼接测试 (两组)
for col, (a, b) in enumerate([(pick[0], pick[1]), (pick[2], pick[3])]):
    ra, rb = level(a), level(b)
    raw = np.concatenate([ra, rb])
    v = vwin(raw)
    ax = axes[0, col]
    x = np.arange(len(raw))
    ax.plot(x, v, "-", color="#9467bd", lw=1.5, label="V_milestone windowed")
    ax.plot(x, vmono(raw), "--", color="#9467bd", lw=0.9, alpha=.5, label="cummax (对照: 时间累计式)")
    ax.axvline(len(ra), color="r", lw=1.5, ls=":", label=f"拼接点 (ep{a}结束→ep{b}开始)")
    drop = v[len(ra) - 1] - v[min(len(ra) + W, len(v) - 1) - W // 2]
    ax.set_title(f"拼接: ep{a}+ep{b} — windowed 在边界回落 {v[len(ra)-1]:.1f}→{v[len(ra)+W//3]:.1f}", fontsize=9)
    ax.set_ylim(-0.05, 1.1); ax.grid(alpha=.3); ax.legend(fontsize=7)

# ② 倒放测试 (两条)
for col, ep in enumerate(pick[:2]):
    r = level(ep)
    vf, vr = vwin(r), vwin(r[::-1])
    ax = axes[1, col]
    x = np.arange(len(r))
    ax.plot(x, vf, "-", color="#2ca02c", lw=1.5, label="正放 value")
    ax.plot(x, vr, "-", color="#d62728", lw=1.5, alpha=.8, label="倒放 value")
    ax.plot(x, vf[::-1], "k:", lw=1, alpha=.6, label="正放的镜像 (预期=倒放)")
    from scipy.stats import pearsonr
    r_mirror = pearsonr(vr, vf[::-1])[0]
    ax.set_title(f"倒放: ep{ep} — corr(倒放, 正放镜像)={r_mirror:.2f}", fontsize=9)
    ax.set_ylim(-0.05, 1.1); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_xlabel("frame (3Hz)", fontsize=8)

fig.suptitle("discriminative test on held-out vis demos: V_milestone 是图像状态的函数, 非时间驱动", fontsize=11)
fig.tight_layout()
out = REPO / "temp/recurrence_v0/discriminative_demo.png"
fig.savefig(out, dpi=120)
print("plot ->", out)
