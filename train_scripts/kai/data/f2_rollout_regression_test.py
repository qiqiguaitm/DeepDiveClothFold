#!/usr/bin/env python
"""F2 (§4.4.6 排期): 真机 3 轮叠衣 rollout 上验证退步回落规则 + 修订判据 (§4.4.7③ rollout 终判)。
挖掘 = smooth800 500ep img⊕proprio k96/M20 (vis 同本体), P_k 首入中位 + 循环三票;
V2.1 规则: 绝对 anchor 门控首入 -> V=max(V,P_k); 已见 anchor 置信重入 (P_k<=V-0.15,
驻留>=3帧=1s) -> V 回落至该 anchor 的 P_k (§4.4.6 规则②③)。
预期: V 在 ~3000/~5000 帧两个轮次边界回落重爬; ΔV<0 集中在边界 (判据② 负标签语义审计)。
产物: temp/f2_rollout/{f2_curves.png, result.json}
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/self_built/A_new_smooth_800/base"
CACHE = REPO / "temp/tcc_smooth800_armmask/feat_cache"
OUT = REPO / "temp/f2_rollout"
OUT.mkdir(parents=True, exist_ok=True)
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)

# ---- smooth800 挖掘 (与图32协议一致: 全池打乱取500) ----
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
mined = sorted(np.random.RandomState(0).permutation([e for e in all_eps if e != 660])[:500].tolist())
def load_raw(ds, cache, e):
    img = np.load(cache / f"ep{e}.npz")["f"]
    n = len(img)
    st = np.stack(pd.read_parquet(ds / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)
print("[f2] loading mining features ...")
IMG, PRP, E, T = [], [], [], []
for e in mined:
    try:
        i, p = load_raw(DS, CACHE, e)
    except Exception:
        continue
    IMG.append(i); PRP.append(p)
    E.append(np.full(len(i), e)); T.append(np.arange(len(i)) / max(1, len(i) - 1))
IMG = np.concatenate(IMG); PRP = np.concatenate(PRP); E = np.concatenate(E); T = np.concatenate(T)
MU, SD = PRP.mean(0), PRP.std(0) + 1e-8
def mkfeat(img, prp):
    p = (prp - MU) / SD; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)
F = mkfeat(IMG, PRP)
n_ep = len(set(E.tolist()))
print(f"[f2] mining {n_ep} eps, {len(F)} frames; KMeans k=96 ...")
km = KMeans(n_clusters=96, n_init=2, random_state=0).fit(F)
lab = km.labels_
cov = np.array([len(set(E[lab == c].tolist())) / n_ep for c in range(96)])
tpos = np.array([T[lab == c].mean() for c in range(96)])
ms = sorted(np.argsort(cov)[-20:].tolist(), key=lambda c: tpos[c])

def gated_runs_idx(idx_arr):
    runs = []; s = None; prev = None
    for i in idx_arr:
        if prev is None or i != prev + 1:
            if s is not None: runs.append((s, prev))
            s = i
        prev = i
    if s is not None: runs.append((s, prev))
    return [r for r in runs if r[1] - r[0] >= 1]

Pk, cyc = {}, {}
for c in ms:
    fe, nr, re_, vis, starts = [], [], 0, 0, []
    for e in set(E.tolist()):
        m = np.where(E == e)[0]
        rs = gated_runs_idx(m[lab[m] == c].tolist())
        if not rs: continue
        vis += 1; fe.append(T[rs[0][0]]); nr.append(len(rs)); starts += [T[r[0]] for r in rs]
        if len(rs) >= 2:
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
anchors = sorted([c for c in ms if not cyc[c]], key=lambda c: Pk[c])
print(f"[f2] absolute anchors: {len(anchors)}/20  P {Pk[anchors[0]]:.2f}-{Pk[anchors[-1]]:.2f}; cyclic: {[f'c{c}' for c in ms if cyc[c]]}")

# ---- autonomy rollout 帧分配 ----
AUTO = REPO / "temp/autonomy"
img = np.load(REPO / "temp/tcc_autonomy_armmask/feat_cache/ep0.npz")["f"]
n = len(img)
st = np.stack(pd.read_parquet(AUTO / "data/chunk-000/episode_000000.parquet",
                              columns=["observation.state"])["observation.state"].to_numpy())
st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
prp = np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)
fA = mkfeat(img, prp)
D = np.linalg.norm(fA[:, None, :] - km.cluster_centers_[None], axis=2)
lA = D.argmin(1); ds_ = np.sort(D, axis=1); mg = ds_[:, 0] / ds_[:, 1]
aset = set(anchors)

# ---- V_mono vs V2.1 (退步回落) ----
DELTA = 0.15
def values():
    vm = np.zeros(n); vr = np.zeros(n)
    cm = 0.0; cr = 0.0
    seen = set()
    events = []
    j = 0
    while j < n:
        c = lA[j]
        # run 长度
        k = j
        while k + 1 < n and lA[k + 1] == c:
            k += 1
        runlen = k - j + 1
        conf = runlen >= 2 or mg[j] <= 0.8
        if c in aset and conf:
            if c not in seen:
                seen.add(c)
                cm = max(cm, Pk[c]); cr = max(cr, Pk[c])
            else:
                cm = max(cm, Pk[c])
                if Pk[c] <= cr - DELTA and runlen >= 3:        # 退步: 已见 anchor 置信重入
                    events.append((j, cr, Pk[c]))
                    cr = Pk[c]
                else:
                    cr = max(cr, Pk[c])
        vm[j:k + 1] = cm; vr[j:k + 1] = cr
        j = k + 1
    return vm, vr, events
vm, vr, events = values()
print(f"[f2] regression events: {[(int(j), f'{a:.2f}->{b:.2f}') for j, a, b in events]}")

# ---- 判据②: ΔV<0 帧位置 ----
H = 5
dvr = vr[H:] - vr[:-H]
neg_idx = np.where(dvr < -1e-6)[0]
print(f"[f2] ΔV<0 frames: {len(neg_idx)}/{len(dvr)} ({len(neg_idx)/len(dvr):.1%}), at 3Hz idx {neg_idx[:20].tolist()}")

# pi0-AE 对照
ae = np.load(REPO / "temp/autonomy_pi0ae.npy")
ae3 = ae[np.minimum(np.arange(n) * 10, len(ae) - 1)] if len(ae) >= n * 10 - 9 else ae[:n]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
x = np.arange(n) * 10  # 原始帧号
for b in (3000, 5000):
    for ax in axes:
        ax.axvline(b, color="r", ls=":", lw=1.2, alpha=.7)
axes[0].plot(x, vm, "-", color="#d62728", lw=1.4, label="V monotone (cummax, 现行)")
axes[0].plot(x, vr, "-", color="#2ca02c", lw=1.8, label="V2.1 退步回落")
for j, a, b in events:
    axes[0].annotate(f"{a:.2f}→{b:.2f}", (j * 10, b), fontsize=7, color="#2ca02c",
                     xytext=(j * 10, b - 0.12))
axes[0].set_ylabel("V"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[0].set_title("real-robot 3-round folding rollout: V2.1 regression rule vs monotone (red dotted = round boundaries ~3000/~5000)", fontsize=10)
axes[1].plot(x[:-H], dvr, "-", color="#9467bd", lw=0.8)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_ylabel("ΔV (50frames)"); axes[1].grid(alpha=.3)
axes[1].set_title("advantage layer: negative ΔV should concentrate at round boundaries (semantic audit)", fontsize=9)
ae_n = (ae3 - ae3.min()) / (np.ptp(ae3) + 1e-9)
axes[2].plot(x, ae_n, "-", color="#1f77b4", lw=1.0)
axes[2].set_ylabel("pi0-AE value (norm)"); axes[2].set_xlabel("rollout frame (30Hz)")
axes[2].grid(alpha=.3)
axes[2].set_title("supervised pi0-AE reference (3-segment structure)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "f2_curves.png", dpi=125)
json.dump({"events": [(int(j * 10), float(a), float(b)) for j, a, b in events],
           "neg_frac": float(len(neg_idx) / len(dvr)),
           "neg_idx30hz": (neg_idx[:50] * 10).tolist(),
           "anchors": len(anchors)}, open(OUT / "result.json", "w"), indent=2)
print(f"[f2] -> {OUT}/")
