"""ep2047 (kai0_base) 真 30Hz 解算: 用 stride-1 特征(temp/_ep2047_30hz)逐帧算
CRAVE / frozen-TCC, AE 原生 30Hz。对照之前 3Hz-repeat 版。输出图 + sync 视频数据。
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib, torch, torch.nn as nn
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdf5_v24_eval import build_model, loadep, mkp
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0); torch.manual_seed(0)
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = REPO / "temp/crave_kai0bd/feat_cache"          # 3Hz 挖矿集
EP30 = REPO / "temp/_ep2047_30hz"                     # ep2047 stride-1
BASE = REPO / "kai0/data/Task_A/kai0_base"; Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
TEST = 2047; W = 50
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]; csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]
eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))

# CRAVE 模型(3Hz 挖矿, 逐字 V2.4)
value, _ = build_model(FC, eps, eps)
# 30Hz ep2047 特征
d30 = np.load(EP30 / f"ep{TEST}.npz"); a30, r30, s30 = d30["armmask"], d30["raw"], d30["state"]
n30 = min(len(a30), len(r30), len(s30)); a30, r30, s30 = a30[:n30], r30[:n30], s30[:n30]
print(f"ep{TEST} 30Hz frames: {n30}", flush=True)
crave30 = value(a30, r30, s30)   # value() 内部按输入帧数解算 → 30Hz

# emb 复刻供 TCC
Sall = [loadep(FC, e)[2] for e in eps]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
def emb(a_, r_, s_):
    an = a_/np.linalg.norm(a_, axis=1, keepdims=True); rn = r_/np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(s_)-PMU)/PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1).astype(np.float32)
Gd = {e: emb(*loadep(FC, e)[:3]) for e in eps}
class Head(nn.Module):
    def __init__(s, din): super().__init__(); s.net = nn.Sequential(nn.Linear(din,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,128))
    def forward(s, x): return s.net(x)
head = Head(Gd[eps[0]].shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
for step in range(1200):
    bes = list(np.random.choice(eps, 8, replace=False)); embs, idxs, lens = [], [], []
    for e in bes:
        f = Gd[e]; m = len(f); ix = np.sort(np.random.choice(m, size=32, replace=m < 32))
        embs.append(head(torch.from_numpy(f[ix]))); idxs.append(torch.from_numpy(ix).long()); lens.append(m)
    loss = compute_tcc_loss(embs=torch.stack(embs), idxs=torch.stack(idxs), seq_lens=torch.tensor(lens),
        stochastic_matching=False, normalize_embeddings=True, loss_type="regression_mse", similarity_type="l2",
        num_cycles=20, cycle_length=2, temperature=0.1, label_smoothing=0.1, variance_lambda=0.001, huber_delta=0.1, normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step()
head.eval()
def hemb(x):
    with torch.no_grad(): z = head(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()
    return z/(np.linalg.norm(z, axis=1, keepdims=True)+1e-9)
def med(a, w=27):  # 30Hz → 27 帧≈0.9s 中值
    h = w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
REFS = [e for e in eps if e != TEST][:30]; REs = [hemb(Gd[e]) for e in REFS]; RTs = [np.arange(len(z))/max(1,len(z)-1) for z in REs]
zq = hemb(emb(a30, r30, s30)); preds = [RTs[k][(zq@REs[k].T).argmax(1)] for k in range(len(REFS))]
tcc30 = med(np.median(np.stack(preds), 0))

# AE 原生 30Hz
ae = pd.read_parquet(Q5/"data"/f"chunk-{TEST//csQ:03d}"/f"episode_{TEST:06d}.parquet")["absolute_value"].to_numpy().astype(float)
NF = min(n30, len(ae)); crave30, tcc30, ae = crave30[:NF], tcc30[:NF], ae[:NF]
x = np.arange(NF)
def mono(v): return np.mean(np.diff(v) >= -1e-6)
def aden(v): a = np.array([v[min(i+W,len(v)-1)]-v[i] for i in range(len(v))]); return np.mean(np.abs(np.clip(a,-1,1))>1e-3)
np.savez(REPO/"temp/_solve_ep2047_30hz.npz", crave=crave30, tcc=tcc30, ae=ae, x=x)
print(f"30Hz end CRAVE{crave30[-1]:.2f} TCC{tcc30[-1]:.2f} AE{ae[-1]:.2f}", flush=True)
print(f"30Hz mono CRAVE{mono(crave30):.0%} TCC{mono(tcc30):.0%} AE{mono(ae):.0%}", flush=True)

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(x, crave30, color="#1f77b4", lw=1.8, label=f"CRAVE 30Hz逐帧 (end{crave30[-1]:.2f} 单调{mono(crave30):.0%})")
ax.plot(x, tcc30, color="#2ca02c", lw=1.8, label=f"TCC 30Hz逐帧 (end{tcc30[-1]:.2f} 单调{mono(tcc30):.0%})")
ax.plot(x, ae, color="#d62728", lw=1.4, alpha=.85, label=f"pi0-AE 原生30Hz (end{ae[-1]:.2f} 单调{mono(ae):.0%})")
ax.axhline(1, color="#999", ls=":", lw=1); ax.set_xlim(0, NF); ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("frame (30Hz)"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.legend(fontsize=9, loc="upper left")
ax.set_title(f"kai0_base ep{TEST} 真30Hz 逐帧解算 (stride-1, 非3Hz-repeat): CRAVE / TCC / pi0-AE", fontsize=12)
out = REPO/"docs/visualization/cross_episode_recurrence_value/solve_ep2047_30hz.png"
fig.tight_layout(); fig.savefig(out, dpi=125); print("SAVED", out, flush=True); print("DONE", flush=True)
