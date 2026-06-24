#!/usr/bin/env python
"""TCC 应用场景①验证: 用 smooth800 域 TCC head 的对齐-进度读出处理 autonomy rollout,
检验 §4.4.10 猜想——TCC 嵌入能否消除聚类主线的"高位误吸"(f400 团布/f3100 空桌
被误配到高进度, 图39 已证 TCC 区分始/终态)。
对照: raw armmask⊕proprio 最近邻读出(无 TCC)在同 rollout 上的高位误吸。
产物: temp/tcc_rollout/{disambig.png, result.json}
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/self_built/A_new_smooth_800/base"
CACHE = REPO / "temp/tcc_smooth800_armmask/feat_cache"
OUT = REPO / "temp/tcc_rollout"; OUT.mkdir(parents=True, exist_ok=True)
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)

# smooth800 head 训练时的协议: pool 打乱(seed0, 排除660), 前200做train, PMU/PSD取这200
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
pool = np.random.RandomState(0).permutation([e for e in all_eps if e != 660]).tolist()
TRAIN = pool[:200]

def load_raw_demo(e):
    img = np.load(CACHE / f"ep{e}.npz")["f"]; n = len(img)
    st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)

print("[disambig] loading smooth800 refs ...")
RAW = {}
for e in TRAIN:
    try: RAW[e] = load_raw_demo(e)
    except Exception: pass
TRAIN = [e for e in TRAIN if e in RAW]
P = np.concatenate([RAW[e][1] for e in TRAIN]); MU, SD = P.mean(0), P.std(0) + 1e-8
def feat(img, prp):
    p = (prp - MU) / SD; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)
FEAT = {e: feat(*RAW[e]) for e in TRAIN}

class Head(nn.Module):
    def __init__(self, din=412, dh=256, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(),
                                 nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)
head = Head(); head.load_state_dict(torch.load(REPO / "temp/tcc_v3_smooth800/tcc_head_v3.pt")); head.eval()
def hemb(x):
    with torch.no_grad(): z = head(torch.from_numpy(x)).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
def nrm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

# autonomy rollout 特征
img = np.load(REPO / "temp/tcc_autonomy_armmask/feat_cache/ep0.npz")["f"]; n = len(img)
st = np.stack(pd.read_parquet(REPO / "temp/autonomy/data/chunk-000/episode_000000.parquet",
                              columns=["observation.state"])["observation.state"].to_numpy())
st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
fA = feat(img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1))

REFS = TRAIN[:30]
def readout(emb_fn):
    REs = [emb_fn(FEAT[r]) for r in REFS]
    RTs = [np.arange(len(z)) / max(1, len(z) - 1) for z in REs]
    z = emb_fn(fA)
    preds = [RTs[k][(z @ REs[k].T).argmax(1)] for k in range(len(REFS))]
    return median_filter(np.median(np.stack(preds), 0), size=9)
v_tcc = readout(hemb)
v_raw = readout(nrm)

# f400≈idx40, f3100≈idx310 (空桌/团布高位误吸区, §4.4.10)
def hi_at(v, idx, w=8):
    return float(np.max(v[max(0, idx - w):idx + w]))
probe = {"f400_idx40(bunched)": (40,), "f3100_idx310(empty)": (310,)}
res = {}
for name, (idx,) in probe.items():
    res[name] = {"raw_hi": hi_at(v_raw, idx), "tcc_hi": hi_at(v_tcc, idx)}
    print(f"  {name}: raw高位={res[name]['raw_hi']:.2f}  TCC={res[name]['tcc_hi']:.2f}")

fig, ax = plt.subplots(figsize=(13, 4.2))
x = np.arange(n) * 10
ax.plot(x, v_raw, "-", color="#d62728", lw=1.3, label="raw armmask⊕proprio NN (no TCC)")
ax.plot(x, v_tcc, "-", color="#2ca02c", lw=1.8, label="TCC v3 align-progress")
for b in (3000, 5000):
    ax.axvline(b, color="gray", ls=":", lw=1.0)
for nm, (idx,) in probe.items():
    ax.annotate(nm.split("_")[0], (idx * 10, 1.02), fontsize=8, color="purple", ha="center")
    ax.axvline(idx * 10, color="purple", ls="--", lw=0.8, alpha=.5)
ax.set_xlabel("rollout frame (30Hz)"); ax.set_ylabel("progress value")
ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=9); ax.grid(alpha=.3)
ax.set_title("App①: TCC align-progress vs raw-feature NN on autonomy rollout — does TCC fix high-position mis-suck at bunched/empty frames?", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "disambig.png", dpi=125)
json.dump(res, open(OUT / "result.json", "w"), indent=2)
print(f"[disambig] -> {OUT}/")
