#!/usr/bin/env python
"""kai0_base 随机长 episode 上对比: 离散 CRAVE 阶梯 vs TCC 连续化 vs pi0-AE 监督连续 value。
(IDW 已弃, 末值离1太远=frozen距离噪声+首尾混淆)
 - CRAVE 特征/离散值: temp/crave_kai0bd/feat_cache + hdf5_v24_eval.build_model (逐字 V2.4)
 - TCC 连续: 同特征训 frozen-feature TCC 头(cycle-consistency)→ 逐帧对齐-进度读出
 - AE 连续: advantage_q5 ep absolute_value (AE 对 kai0_base 的 Stage-2 输出, 同一 ep)
输出: docs/.../tcc_vs_ae_kai0base_ep{TEST}.png
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib, torch, torch.nn as nn
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdf5_v24_eval import build_model, loadep, mkp
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0); torch.manual_seed(0)

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = REPO / "temp/crave_kai0bd/feat_cache"
BASE = REPO / "kai0/data/Task_A/kai0_base"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
TEST = 2047; W = 50
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]
eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
print(f"挖矿 {len(eps)} eps; test=kai0_base ep{TEST}", flush=True)

# ---- 离散 CRAVE ----
value, Pord = build_model(FC, eps, eps)
aa, rr, st, n = loadep(FC, TEST); v3 = value(aa, rr, st)
NF = len(pd.read_parquet(BASE / "data" / f"chunk-{TEST//csB:03d}" / f"episode_{TEST:06d}.parquet", columns=["frame_index"]))
crave = np.repeat(v3, 10)[:NF]
if len(crave) < NF: crave = np.concatenate([crave, np.full(NF - len(crave), crave[-1])])

# ---- emb (复刻 build_model 内部, 同 PMU/PSD) 供 TCC ----
Sall = [loadep(FC, e)[2] for e in eps]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
def emb(a_, r_, s_):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(s_) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)
Gd = {}
for e in eps:
    a_, r_, s_, _ = loadep(FC, e); Gd[e] = emb(a_, r_, s_).astype(np.float32)
DIN = Gd[eps[0]].shape[1]

# ---- TCC frozen-feature 头 ----
class Head(nn.Module):
    def __init__(s, din):
        super().__init__(); s.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))
    def forward(s, x): return s.net(x)
head = Head(DIN); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
for step in range(1200):
    bes = list(np.random.choice(eps, 8, replace=False)); embs, idxs, lens = [], [], []
    for e in bes:
        f = Gd[e]; m = len(f); ix = np.sort(np.random.choice(m, size=32, replace=m < 32))
        embs.append(head(torch.from_numpy(f[ix]))); idxs.append(torch.from_numpy(ix).long()); lens.append(m)
    loss = compute_tcc_loss(embs=torch.stack(embs), idxs=torch.stack(idxs), seq_lens=torch.tensor(lens),
        stochastic_matching=False, normalize_embeddings=True, loss_type="regression_mse", similarity_type="l2",
        num_cycles=20, cycle_length=2, temperature=0.1, label_smoothing=0.1, variance_lambda=0.001, huber_delta=0.1, normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 300 == 0: print(f"  tcc step {step+1} loss {float(loss):.4f}", flush=True)
head.eval()
def hemb(x):
    with torch.no_grad(): z = head(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
def med(arr, w=9):
    h = w // 2; return np.array([np.median(arr[max(0, j - h):j + h + 1]) for j in range(len(arr))])
REFS = [e for e in eps if e != TEST][:30]
REs = [hemb(Gd[e]) for e in REFS]; RTs = [np.arange(len(z)) / max(1, len(z) - 1) for z in REs]
zq = hemb(emb(aa, rr, st)); preds = [RTs[k][(zq @ REs[k].T).argmax(1)] for k in range(len(REFS))]
v_tcc3 = med(np.median(np.stack(preds), 0))
tcc = np.repeat(v_tcc3, 10)[:NF]
if len(tcc) < NF: tcc = np.concatenate([tcc, np.full(NF - len(tcc), tcc[-1])])

# ---- AE 连续 ----
dQ = pd.read_parquet(Q5 / "data" / f"chunk-{TEST//csQ:03d}" / f"episode_{TEST:06d}.parquet")
ae = dQ["absolute_value"].to_numpy().astype(float)
NF = min(NF, len(ae), len(crave), len(tcc)); crave, tcc, ae = crave[:NF], tcc[:NF], ae[:NF]
x = np.arange(NF)
def advv(v, w=W):
    a = np.array([v[min(i + w, len(v) - 1)] - v[i] for i in range(len(v))]); return np.clip(a, -1, 1)
def mono(v): return np.mean(np.diff(v) >= -1e-6)
def aden(v): a = advv(v); return np.mean(np.abs(a) > 1e-3)
r_ct = pearsonr(crave, tcc)[0]; r_ca = pearsonr(crave, ae)[0]; r_ta = pearsonr(tcc, ae)[0]
print(f"end CRAVE{crave[-1]:.2f} TCC{tcc[-1]:.2f} AE{ae[-1]:.2f}", flush=True)
print(f"mono CRAVE{mono(crave):.0%} TCC{mono(tcc):.0%} AE{mono(ae):.0%}", flush=True)
print(f"adv密度 CRAVE{aden(crave):.0%} TCC{aden(tcc):.0%} AE{aden(ae):.0%}", flush=True)
print(f"corr CRAVE-TCC{r_ct:.2f} CRAVE-AE{r_ca:.2f} TCC-AE{r_ta:.2f}", flush=True)

fig, ax = plt.subplots(2, 1, figsize=(13, 7), height_ratios=[1.4, 1], sharex=True)
ax[0].step(x, crave, where="post", color="#1f77b4", lw=2.2, label=f"离散 CRAVE 阶梯 (end{crave[-1]:.2f} 单调{mono(crave):.0%} adv密度{aden(crave):.0%})")
ax[0].plot(x, tcc, color="#2ca02c", lw=1.9, label=f"连续 TCC 对齐-进度 (end{tcc[-1]:.2f} 单调{mono(tcc):.0%} adv密度{aden(tcc):.0%})")
ax[0].plot(x, ae, color="#d62728", lw=1.6, alpha=.85, label=f"pi0-AE 监督连续 (end{ae[-1]:.2f} max{ae.max():.2f} 单调{mono(ae):.0%} adv密度{aden(ae):.0%})")
ax[0].axhline(1, color="#999", ls=":", lw=1); ax[0].axhline(0, color="k", lw=.5)
ax[0].set_ylabel("value"); ax[0].grid(alpha=.25); ax[0].legend(fontsize=9, loc="upper left")
ax[0].set_title(f"kai0_base ep{TEST} ({NF}f≈{NF/30:.0f}s): 离散CRAVE vs TCC连续 vs pi0-AE监督 — value 对比", fontsize=12)
ax[1].plot(x, advv(crave), color="#1f77b4", lw=1.3, label="CRAVE ΔV")
ax[1].plot(x, advv(tcc), color="#2ca02c", lw=1.3, label="TCC ΔV")
ax[1].plot(x, advv(ae), color="#d62728", lw=1.1, alpha=.8, label="AE ΔV")
ax[1].axhline(0, color="k", lw=.6); ax[1].set_ylabel(f"advantage (n vs n+{W})"); ax[1].set_xlabel("frame"); ax[1].grid(alpha=.25); ax[1].legend(fontsize=8)
ax[1].set_title(f"advantage 层: 离散稀疏尖峰 vs TCC/AE 密集; corr(CRAVE,TCC)={r_ct:.2f} corr(TCC,AE)={r_ta:.2f}", fontsize=10.5)
out = REPO / f"docs/visualization/cross_episode_recurrence_value/tcc_vs_ae_kai0base_ep{TEST}.png"
fig.tight_layout(); fig.savefig(out, dpi=120); print("SAVED", out, flush=True)
np.savez(REPO / f"temp/_tcc_ae_kai0base_ep{TEST}.npz", crave=crave, tcc=tcc, ae=ae, x=x)
print("DONE", flush=True)
