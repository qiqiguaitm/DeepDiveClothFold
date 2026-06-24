#!/usr/bin/env python
"""两线合流实验 (§4.4.5 段内插值升级路径落地):
V_hybrid(t) = milestone 校准阶梯给绝对锚位 + TCC v3 对齐-进度做段内插值
  - 阶梯: V2 配方 (P_k 首入中位, 置信门控, 循环型忽略) -> 锚 A(t), 下一锚 N(t)
  - 插值: p_tcc(t) = TCC head 逐参考 argmax 中位数读出 (refs=30)
  - V_hybrid(t) = cummax( clip(p_tcc(t), A(t), N(t)) )
评测 (kai0 held-out 50 GT eps): MAE/Pearson/tau vs uniform / calibrated / TCC-only
附: 拼接判别测试 (两条 held-out ep 拼接, p_tcc 应在边界回落重爬)
产物: temp/v2_tcc_hybrid/{eval.json, hybrid_curves.png, concat_test.png}
"""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.cluster import KMeans
from scipy.stats import kendalltau, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_advantage"
CACHE = REPO / "temp/tcc_kai0_armmask/feat_cache"
OUT = REPO / "temp/v2_tcc_hybrid"
OUT.mkdir(parents=True, exist_ok=True)
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)
np.random.seed(0); random.seed(0); torch.manual_seed(0)

zp = np.load(REPO / "temp/recurrence_v0_kai0/embeddings.npz")
EVAL = sorted(set(zp["ep_ids"].tolist()))
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
pool = np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()
TRAIN500 = pool[:500]   # 挖掘集 (milestone)
TRAIN200 = pool[:200]   # TCC v3 训练时所用 (PMU/PSD 协议一致)

def load_raw(e):
    img = np.load(CACHE / f"ep{e}.npz")["f"]
    n = len(img)
    st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)

print("[hybrid] loading features ...")
RAW = {}
for e in set(TRAIN500 + EVAL):
    try:
        RAW[e] = load_raw(e)
    except Exception:
        pass
TRAIN500 = [e for e in TRAIN500 if e in RAW]
TRAIN200 = [e for e in TRAIN200 if e in RAW]
EVALu = [e for e in EVAL if e in RAW]

# 挖掘特征 (PMU/PSD over 500, 与 §2.11 协议一致)
P500 = np.concatenate([RAW[e][1] for e in TRAIN500])
MU5, SD5 = P500.mean(0), P500.std(0) + 1e-8
def feat5(e):
    p = (RAW[e][1] - MU5) / SD5; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = RAW[e][0] / (np.linalg.norm(RAW[e][0], axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)
# TCC 特征 (PMU/PSD over 200, 与 tcc_v3 协议一致)
P200 = np.concatenate([RAW[e][1] for e in TRAIN200])
MU2, SD2 = P200.mean(0), P200.std(0) + 1e-8
def feat2(e):
    p = (RAW[e][1] - MU2) / SD2; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = RAW[e][0] / (np.linalg.norm(RAW[e][0], axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)

GT = {}
for e in EVALu:
    g = pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                        columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    GT[e] = g[np.minimum(np.arange(len(RAW[e][0])) * 10, len(g) - 1)]

# ---- milestone 挖掘 (V2 配方: k96/M20 + P_k + 循环三票) ----
print("[hybrid] mining milestones (500 ep, k=96) ...")
F5 = np.concatenate([feat5(e) for e in TRAIN500])
E5 = np.concatenate([np.full(len(RAW[e][0]), e) for e in TRAIN500])
T5 = np.concatenate([np.arange(len(RAW[e][0])) / max(1, len(RAW[e][0]) - 1) for e in TRAIN500])
km = KMeans(n_clusters=96, n_init=2, random_state=0).fit(F5)
lab = km.labels_
n_ep = len(TRAIN500)
cov = np.array([len(set(E5[lab == c].tolist())) / n_ep for c in range(96)])
tpos = np.array([T5[lab == c].mean() for c in range(96)])
ms = sorted(np.argsort(cov)[-20:].tolist(), key=lambda c: tpos[c])

def gated_runs(e, c):
    m = np.where(E5 == e)[0]; hit = m[lab[m] == c]
    runs = []; s = None; prev = None
    for i in hit:
        if prev is None or i != prev + 1:
            if s is not None: runs.append((s, prev))
            s = i
        prev = i
    if s is not None: runs.append((s, prev))
    return [r for r in runs if r[1] - r[0] >= 1]

from sklearn.mixture import GaussianMixture
Pk, cyc = {}, {}
for c in ms:
    fe, nr, re_, vis, starts = [], [], 0, 0, []
    for e in TRAIN500:
        rs = gated_runs(e, c)
        if not rs: continue
        vis += 1; fe.append(T5[rs[0][0]]); nr.append(len(rs)); starts += [T5[r[0]] for r in rs]
        if len(rs) >= 2:
            m = np.where(E5 == e)[0]
            for (a1, b1), (a2, b2) in zip(rs[:-1], rs[1:]):
                if any(x in ms and x != c for x in lab[[j for j in m if b1 < j < a2]]):
                    re_ += 1; break
    Pk[c] = float(np.median(fe)) if fe else tpos[c]
    v1 = np.mean(nr) > 1.5 if nr else False
    X = np.array(starts).reshape(-1, 1)
    v2 = False
    if len(X) >= 10:
        v2 = (GaussianMixture(1, random_state=0).fit(X).bic(X) -
              GaussianMixture(2, random_state=0).fit(X).bic(X)) > 10
    v3 = (re_ / max(1, vis)) > 0.2
    cyc[c] = int(v1) + int(v2) + int(v3) >= 2
abs_ms = [c for c in ms if not cyc[c]]
abs_sorted = sorted(abs_ms, key=lambda c: Pk[c])
print(f"  absolute milestones: {len(abs_sorted)}/20, P_k range {Pk[abs_sorted[0]]:.2f}-{Pk[abs_sorted[-1]]:.2f}")

# ---- eval ep 的阶梯 (门控 + 忽略循环型, E3 最优) ----
C = km.cluster_centers_
msset = set(abs_ms)
def staircase(e):
    f = feat5(e)
    D = np.linalg.norm(f[:, None, :] - C[None], axis=2)
    l = D.argmin(1); ds = np.sort(D, axis=1); mg = ds[:, 0] / ds[:, 1]
    A = np.zeros(len(f)); N = np.ones(len(f))
    cur = 0.0; seen = set()
    for j in range(len(f)):
        c = l[j]
        dwell = (j + 1 < len(f) and l[j + 1] == c) or (j > 0 and l[j - 1] == c)
        if c in msset and (dwell or mg[j] <= 0.8) and c not in seen:
            seen.add(c); cur = max(cur, Pk[c])
        A[j] = cur
        nxt = [Pk[a] for a in abs_sorted if Pk[a] > cur + 1e-9]
        N[j] = nxt[0] if nxt else 1.0
    return A, N

# ---- TCC v3 head + 逐参考中位数读出 ----
class Head(nn.Module):
    def __init__(self, din=412, dh=256, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(),
                                 nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)
head = Head(); head.load_state_dict(torch.load(REPO / "temp/tcc_v3_kai0/tcc_head_v3.pt")); head.eval()
def hemb(x):
    with torch.no_grad():
        z = head(torch.from_numpy(x)).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
REFS = TRAIN200[:30]
REs = [hemb(feat2(r)) for r in REFS]
RTs = [np.arange(len(z)) / max(1, len(z) - 1) for z in REs]
def tcc_progress(f2):
    z = hemb(f2)
    preds = [RTs[k][(z @ REs[k].T).argmax(1)] for k in range(len(REFS))]
    return np.median(np.stack(preds), 0)

# ---- 评测四种 value ----
print("[hybrid] evaluating ...")
res = {m: ([], [], []) for m in ("uniform", "calibrated", "tcc_only", "hybrid")}
curves = {}
for e in EVALu:
    g = GT[e]
    if g.std() < 1e-6: continue
    A, N = staircase(e)
    p = tcc_progress(feat2(e))
    vh = np.maximum.accumulate(np.clip(p, A, N))
    vu_steps = (A > 0).astype(float)  # placeholder
    # uniform 阶梯: 用与 A 相同的首入事件序列, 等步长
    # 重算: 首入次数 / len(abs_sorted)
    k_seen = np.searchsorted(np.sort([Pk[a] for a in abs_sorted]), A, side="right")
    vu = k_seen / max(1, len(abs_sorted))
    vals = {"uniform": vu, "calibrated": A, "tcc_only": np.maximum.accumulate(p), "hybrid": vh}
    curves[e] = (g, A, p, vh)
    for m, v in vals.items():
        res[m][0].append(kendalltau(v, g)[0])
        res[m][1].append(pearsonr(v, g)[0])
        res[m][2].append(np.abs(v - g).mean())
summary = {}
for m, (t, r, ma) in res.items():
    summary[m] = dict(tau=float(np.nanmean(t)), r=float(np.nanmean(r)), mae=float(np.nanmean(ma)))
    print(f"  {m:<11} tau={summary[m]['tau']:.3f} Pearson={summary[m]['r']:.3f} MAE={summary[m]['mae']:.3f}")
json.dump(summary, open(OUT / "eval.json", "w"), indent=2)

# ---- 拼接判别测试: 两条 held-out 拼接, p_tcc 与 hybrid 应回落重爬 ----
ea, eb = EVALu[0], EVALu[1]
fcat = np.concatenate([feat2(ea), feat2(eb)])
pcat = tcc_progress(fcat)
nA = len(feat2(ea))
seg1, seg2 = pcat[:nA], pcat[nA:]
drop = float(seg1[-5:].mean() - seg2[:5].mean())
t1 = kendalltau(seg2, np.arange(len(seg2)))[0]
print(f"  concat: boundary drop={drop:.2f}, second-segment re-climb tau={t1:.2f}")

# ---- 图 ----
fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
show = [e for e in curves][:3]
for ax, e in zip(axes[:3], show):
    g, A, p, vh = curves[e]
    x = np.arange(len(g)) / 3
    ax.plot(x, g, "k-", lw=2, label="GT")
    ax.step(x, A, "-", color="#d62728", lw=1.2, where="post", label="V2 staircase (anchors)")
    ax.plot(x, p, ":", color="#1f77b4", lw=1, alpha=.7, label="TCC progress (raw)")
    ax.plot(x, vh, "-", color="#2ca02c", lw=1.8, label="HYBRID")
    ax.set_title(f"ep{e} (held-out)", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=.3); ax.set_ylim(-0.05, 1.1)
ax = axes[3]
xc = np.arange(len(pcat)) / 3
ax.plot(xc, pcat, "-", color="#1f77b4", lw=1.2)
ax.axvline(nA / 3, color="r", ls=":", lw=1.5, label=f"splice (drop={drop:.2f})")
ax.set_title(f"concat test ep{ea}+ep{eb}: TCC progress resets", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.suptitle("HYBRID value: V2 calibrated anchors + TCC v3 within-segment interpolation (kai0 held-out)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "hybrid_curves.png", dpi=125)
print(f"[hybrid] outputs -> {OUT}/")
