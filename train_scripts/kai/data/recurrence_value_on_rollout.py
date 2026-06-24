#!/usr/bin/env python
"""鉴别性实验: demo(smooth800)挖出的 recurrence milestone, 跨数据集应用到真机
autonomy rollout, 与 pi0-AE / ViVa 的 value 曲线同图对比。
核心问题: 状态触发的 V_milestone 在 rollout 的失败/重试段是否 plateau/回落
(时间线性的信号做不到), 即 recurrence value 是否真"懂状态"。

用法: kai0/.venv/bin/python train_scripts/kai/data/recurrence_value_on_rollout.py
产物: temp/recurrence_v0/rollout_value_compare.png + autonomy_milestone_value.npy
"""
import json
from pathlib import Path
import numpy as np
import torch

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROBE = REPO / "temp/recurrence_v0"           # smooth800 V0 产物 (mining 源)
ROLL = REPO / "temp/autonomy"                  # 真机 rollout
OUT = PROBE
K, TOP_M, SEED, STRIDE = 48, 10, 0, 10

# ---- 1) 重建 smooth800 KMeans (确定性) + milestones ----
z = np.load(PROBE / "embeddings.npz")
feats, ep_ids, tnorm = z["feats"], z["ep_ids"], z["tnorm"]
from sklearn.cluster import KMeans
km = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit(feats)
lab = km.labels_
n_ep = len(set(ep_ids.tolist()))
cov = np.array([len(set(ep_ids[lab == c].tolist())) / n_ep for c in range(K)])
tpos = np.array([tnorm[lab == c].mean() for c in range(K)])
ms = sorted(np.argsort(cov)[-TOP_M:], key=lambda c: tpos[c])
print("milestones:", [(int(c), f"{cov[c]:.0%}", f"t={tpos[c]:.2f}") for c in ms])

# ---- 2) autonomy ep0 特征 (DINOv2, 同探针协议) ----
cache = OUT / "autonomy_ep0_feats.npz"
if cache.exists():
    fa = np.load(cache)["f"]
else:
    import av
    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    imgs = []
    c = av.open(str(ROLL / "videos/chunk-000/top_head/episode_000000.mp4"))
    for i, f in enumerate(c.decode(video=0)):
        if i % STRIDE:
            continue
        h, w = f.height, f.width
        s = 224 / min(h, w)
        g = f.reformat(width=round(w * s), height=round(h * s), format="rgb24")
        img = g.to_ndarray(format="rgb24")
        hh, ww = img.shape[:2]
        imgs.append(img[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])
    c.close()
    out_f = []
    with torch.no_grad():
        for b in range(0, len(imgs), 64):
            px = proc(images=imgs[b:b + 64], return_tensors="pt").to(dev)
            cls = enc(**px).last_hidden_state[:, 0]
            out_f.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    fa = np.concatenate(out_f)
    np.savez_compressed(cache, f=fa)
print("autonomy feats", fa.shape)

# ---- 3) 跨数据集分配簇 + V_milestone(带回退: 离开 milestone 簇集合一段时间则按
#         "当前可见最高里程碑" 维持; 重试造成回到早期簇时, 用滑动窗口内的 milestone 重估) ----
la = km.predict(fa)
def milestone_value(labels, ms, mode="cummax"):
    idx = {c: i + 1 for i, c in enumerate(ms)}
    raw = np.array([idx.get(c, 0) for c in labels], dtype=float)   # 当前帧命中的 milestone 序号
    if mode == "cummax":      # 单调: 已通过最高 milestone (demo 语义)
        v = np.maximum.accumulate(raw)
    else:                      # windowed: 最近 W 帧内命中的最高 milestone (允许回落 → 反映重试)
        W = 90                # 30s @3Hz
        v = np.array([raw[max(0, i - W):i + 1].max() for i in range(len(raw))])
    return v / len(ms)
v_mono = milestone_value(la, ms, "cummax")
v_win = milestone_value(la, ms, "windowed")
np.save(OUT / "autonomy_milestone_value.npy", v_win)

# ---- 4) 对比图: V_milestone vs pi0-AE vs ViVa(已有 npy, 30Hz → 对齐 3Hz) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
T = len(fa)
x = np.arange(T) * STRIDE                                  # 原始 30fps 帧号
def load30(p, invert=False):
    v = np.load(p).astype(float)
    v = v[::STRIDE][:T]
    if invert: v = 1 - v
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    return v
pi0 = load30(REPO / "temp/autonomy_pi0ae.npy")
viva = load30(REPO / "temp/autonomy_official.npy", invert=True)
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
axes[0].plot(x, v_win, "m-", lw=1.2, label="V_milestone windowed (recurrence, zero-train, mined on smooth800 demos)")
axes[0].plot(x, v_mono, "m--", lw=.8, alpha=.5, label="V_milestone cummax (monotone)")
axes[1].plot(x, pi0, "g-", lw=1, label="pi0-AE absolute_value (supervised)")
axes[2].plot(x, viva, "b-", lw=1, label="ViVa-official 1-value")
for ax in axes:
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3); ax.set_ylim(-0.05, 1.1)
axes[-1].set_xlabel("frame (30fps)")
fig.suptitle("autonomy rollout: recurrence-milestone value (cross-dataset, zero-train) vs learned values", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "rollout_value_compare.png", dpi=120)
print("plot ->", OUT / "rollout_value_compare.png")

# 简单量化: 三者两两 Pearson
from scipy.stats import pearsonr
print("corr(V_milestone_win, pi0ae) =", round(pearsonr(v_win, pi0)[0], 3))
print("corr(V_milestone_win, viva)  =", round(pearsonr(v_win, viva)[0], 3))
print("corr(pi0ae, viva)            =", round(pearsonr(pi0, viva)[0], 3))
