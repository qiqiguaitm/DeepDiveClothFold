#!/usr/bin/env python
"""ep808 段间连续化对比: 原离散 CRAVE 阶梯 vs 两种连续化
 方法1 TCC: dagger 挖掘集训 frozen-feature TCC 头(cycle-consistency)→ 逐帧对齐-进度读出
 方法2 IDW: 当前帧到最近 3 个 milestone 簇的反平方距离加权 P_k 插值(用户提议)
挖掘/离散值与 crave_vs_ae_ep808.py 逐字一致。输出 docs/.../continuize_ep808_compare.png
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib, torch, torch.nn as nn
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt, os
from sklearn.cluster import KMeans
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
ARM = REPO / "temp/tcc_smooth800_dagger_armmask/feat_cache"
RAW = REPO / "temp/tcc_smooth800_dagger_raw/feat_cache"
EP, MINE_N = 808, 500
csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
np.random.seed(0); torch.manual_seed(0)

def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//csDS:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]
def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n
def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)

rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
mined = sorted(np.random.RandomState(0).permutation(all_eps)[:min(MINE_N, len(all_eps))].tolist())
if EP not in mined: mined = sorted(mined + [EP])
Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)

A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
for e in mined:
    aa, rr, st, n = loadep(e); g = emb(aa, rr, st)
    A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
    SP.append(g[:2]); EP_.append(g[-2:])
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
G = emb(A, R, S)
km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
Pstart = {}
for e in sorted(set(E.tolist())):
    m = np.where(E == e)[0][:3]; nnz = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nnz]))
cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(96)])
bk = np.linspace(0, 1, 11); sel = []
for b in range(10):
    inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
    if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
sel = sorted(set(sel), key=lambda c: tpos[c])
def gr(idx):
    o = []; s = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s is not None: o.append((s, pv))
            s = i
        pv = i
    if s is not None: o.append((s, pv))
    return [x for x in o if x[1] - x[0] >= 1]
Pk = {}
for c in sel:
    fe = []
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
        if rs: fe.append(T[rs[0][0]])
    Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pk_ord = np.array([Pk[c] for c in order])
startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]
print(f"mined {N} eps, {len(order)} milestones, Pk {Pk_ord.min():.2f}-{Pk_ord.max():.2f}", flush=True)

def dpHB(emit, lam=8.0):
    pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]
def med(arr, w):
    h = w // 2; return np.array([np.median(arr[max(0, j - h):j + h + 1]) for j in range(len(arr))])

# ---- 离散 CRAVE (原阶梯, 逐字同源) ----
aa, rr, st, n = loadep(EP); Fq = emb(aa, rr, st); nq = len(Fq)
d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
em = np.full((nq, NB), 1e3)
for ci in range(len(order)):
    for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
tn = np.arange(nq) / nq
em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
V_disc = med(dpHB(em), 9)

# ---- 方法2: IDW 距离插值 (最近3 milestone 簇, 反平方加权) ----
K = 3
idx_near = np.argsort(d, axis=1)[:, :K]
V_idw = np.zeros(nq)
for j in range(nq):
    dk = d[j, idx_near[j]]; w = 1.0 / (dk ** 2 + 1e-6); V_idw[j] = (w * Pk_ord[idx_near[j]]).sum() / w.sum()
V_idw = med(V_idw, 9)

# ---- 方法1: TCC frozen-feature 头 + 对齐-进度读出 ----
class Head(nn.Module):
    def __init__(s, din):
        super().__init__(); s.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))
    def forward(s, x): return s.net(x)
head = Head(G.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
GT_ = {e: G[E == e] for e in mined}
TRAIN = mined
for step in range(1000):
    bes = list(np.random.choice(TRAIN, 8, replace=False)); embs, idxs, lens = [], [], []
    for e in bes:
        f = GT_[e]; m = len(f); ix = np.sort(np.random.choice(m, size=32, replace=m < 32))
        embs.append(head(torch.from_numpy(f[ix]).float())); idxs.append(torch.from_numpy(ix).long()); lens.append(m)
    loss = compute_tcc_loss(embs=torch.stack(embs), idxs=torch.stack(idxs), seq_lens=torch.tensor(lens),
        stochastic_matching=False, normalize_embeddings=True, loss_type="regression_mse", similarity_type="l2",
        num_cycles=20, cycle_length=2, temperature=0.1, label_smoothing=0.1, variance_lambda=0.001,
        huber_delta=0.1, normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 200 == 0: print(f"  tcc step {step+1} loss {float(loss):.4f}", flush=True)
head.eval()
def hemb(x):
    with torch.no_grad(): z = head(torch.from_numpy(x).float()).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
REFS = [e for e in mined if e != EP][:30]
REs = [hemb(GT_[e]) for e in REFS]; RTs = [np.arange(len(z)) / max(1, len(z) - 1) for z in REs]
zq = hemb(Fq); preds = [RTs[k][(zq @ REs[k].T).argmax(1)] for k in range(len(REFS))]
V_tcc = med(np.median(np.stack(preds), 0), 9)

# ---- 对比图 (3Hz) ----
x = np.arange(nq) * 10  # 还原到原始帧坐标
def stats(v):
    dv = v[5:] - v[:-5]
    return np.mean(np.diff(v) >= -1e-6), np.mean(np.abs(dv) > 1e-3)  # 单调率, advantage 非零密度
m_d, a_d = stats(V_disc); m_i, a_i = stats(V_idw); m_t, a_t = stats(V_tcc)
fig, ax = plt.subplots(2, 1, figsize=(13, 7), height_ratios=[1.4, 1], sharex=True)
ax[0].step(x, V_disc, where="post", color="#1f77b4", lw=2.4, label=f"原离散 CRAVE 阶梯 (单调{m_d:.0%}, adv密度{a_d:.0%})")
ax[0].plot(x, V_idw, color="#ff7f0e", lw=1.8, label=f"连续①IDW 距离插值 (最近3簇, 单调{m_i:.0%}, adv密度{a_i:.0%})")
ax[0].plot(x, V_tcc, color="#2ca02c", lw=1.8, label=f"连续②TCC 对齐-进度 (单调{m_t:.0%}, adv密度{a_t:.0%})")
ax[0].set_ylabel("value"); ax[0].grid(alpha=.25); ax[0].legend(fontsize=9, loc="upper left")
ax[0].set_title(f"ep808 (dagger {nq*10}f) 段间连续化对比: 离散阶梯 vs IDW距离 vs TCC", fontsize=12)
# advantage 层 (ΔV over 50f@30Hz ≈ 5f@3Hz)
def advv(v): a = v[5:] - v[:-5]; return np.concatenate([a, np.full(5, a[-1])])
ax[1].plot(x, advv(V_disc), color="#1f77b4", lw=1.4, label="离散 ΔV (尖峰串)")
ax[1].plot(x, advv(V_idw), color="#ff7f0e", lw=1.2, label="IDW ΔV")
ax[1].plot(x, advv(V_tcc), color="#2ca02c", lw=1.2, label="TCC ΔV")
ax[1].axhline(0, color="k", lw=.6); ax[1].set_ylabel("advantage (ΔV)"); ax[1].set_xlabel("frame"); ax[1].grid(alpha=.25); ax[1].legend(fontsize=8)
ax[1].set_title("advantage 层: 离散=稀疏尖峰(平台ΔV≡0), 连续化=每帧密集梯度", fontsize=10.5)
out = REPO / "docs/visualization/cross_episode_recurrence_value/continuize_ep808_compare.png"
fig.tight_layout(); fig.savefig(out, dpi=120); print("SAVED", out, flush=True)
print(f"adv密度: 离散{a_d:.0%} IDW{a_i:.0%} TCC{a_t:.0%}", flush=True)
np.savez(REPO / "temp/_continuize_ep808.npz", V_disc=V_disc, V_idw=V_idw, V_tcc=V_tcc, x=x)
print("DONE", flush=True)
