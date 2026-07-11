#!/usr/bin/env python
"""kai0_base ep2238: 在线因果 GRU 推理 value  vs  KAI0-AE 监督 value —— 同步对齐视频 + 双分类条。

目的(用户请求): 用【和 §5 一样的在线推理方式】(因果 GRU 蒸馏 polyline teacher, 严格未见 ep2238)
在同一条 ep2238 上跑出逐帧 progress, 与 KAI0-AE 的 value 并排, 各配一条分类条, 直观对比 P/N 分类的噪声程度。

分类条约定(与 report crave_vs_ae 一致, 各用其部署时真实 P/N 信号):
  - 在线 GRU : value=progress(绿) ; P/N = three(advantage(progress, W=50))  —— CRAVE 无相对头, Δvalue 派生
  - KAI0-AE  : value=absolute_value(红) ; P/N = three(relative_advantage 直出)  —— AE 相对头原生输出
噪声量化: 分类信号的 正负翻转次数(flips) + neg 占比。成功 episode 理应几乎无 neg。

ep2238 严格held-out: 训练 eps 显式排除 2238。特征/PCA/proprio 标准化全用训练集拟合, 2238 仅前向。
Run: PYTHONPATH=src /home/tim/miniconda3/envs/srpo/bin/python experiments/render_online_vs_ae_ep2238.py
Out: temp/online_vs_ae/online_vs_ae_ep2238.mp4 (+ _preview.png)
"""
import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DEV = "cuda:0"
rng = np.random.RandomState(0)
torch.manual_seed(0)
CAP = 1000
FPS = 30.0
CSQ = 1000
EP = 2238
W = 50
EPS = 0.02
KAI = REPO / "kai0/data/Task_A/kai0_base"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
OUT = REPO / "lmvla/crave/temp/online_vs_ae"
OUT.mkdir(parents=True, exist_ok=True)
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}
BGR = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}
NAME = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cc(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def advantage(v, w=W):
    return np.array([v[min(i + w, len(v) - 1)] - v[i] for i in range(len(v))])


def three(a):
    return np.where(a > EPS, 1, np.where(a < -EPS, -1, 0))


def flips(sig):
    s = np.sign(np.where(np.abs(sig) > EPS, sig, 0.0))
    s = s[s != 0]
    return int((np.diff(s) != 0).sum()) if len(s) > 1 else 0


def daw(F, C, P, lam):  # 双锚 Viterbi → polyline(去阶梯)  —— 与 render_kai_online_gru.py 一致
    sC = l2(F[:3].mean(0)[None])[0]; eC = l2(F[-3:].mean(0)[None])[0]
    C2 = np.vstack([C, sC, eC]); Pp = np.concatenate([P, [0.0], [1.0]])
    bins = np.unique(np.concatenate([[0.0], Pp, [1.0]])); nb = len(bins)
    cb = [int(np.searchsorted(bins, v)) for v in Pp]; pen = lam * np.abs(bins[:, None] - bins[None])
    de = np.linalg.norm(F[:, None] - C2[None], axis=2); em = np.full((len(F), nb), 1e3)
    for ti in range(len(Pp)):
        em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(F), nb), int)
    for j in range(1, len(F)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = nb - 1; path = np.zeros(len(F), int); path[-1] = si
    for j in range(len(F) - 2, -1, -1):
        si = BP[j + 1][si]; path[j] = si
    step = bins[path]; segs = []; a = 0
    for t in range(1, len(step)):
        if step[t] != step[t - 1]:
            segs.append((a, t - 1, step[t - 1])); a = t
    segs.append((a, len(step) - 1, step[-1])); reps = []
    for i0, i1, val in segs:
        cand = [ti for ti in range(len(Pp)) if abs(Pp[ti] - val) < 1e-9]; fr = np.arange(i0, i1 + 1); bd = 1e18; bf = i0
        for ti in cand:
            dd = np.linalg.norm(F[fr] - C2[ti], axis=1); k = int(dd.argmin())
            if dd[k] < bd:
                bd = dd[k]; bf = fr[k]
        reps.append((bf, float(val)))
    if reps[0][0] != 0:
        reps = [(0, float(step[0]))] + reps
    if reps[-1][0] != len(step) - 1:
        reps = reps + [(len(step) - 1, float(step[-1]))]
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps]); keep = np.concatenate([[True], np.diff(rf) > 0])
    return np.interp(np.arange(len(step)), rf[keep], rv[keep]).astype(np.float32)


# ══════ 1. 加载 bank + 训练集(排除 2238)+ 2238 特征 ══════
print("加载 kai base bank...", flush=True); t0 = time.time()
d = REPO / "lmvla/crave/data/kai_dinov3base"; idx = np.load(d / "index.npz"); E = idx["E"]; FR = idx["FR"]
feat = np.zeros((len(E), 768), np.float16)
for sh in sorted(d.glob("shard_*.npz")):
    s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
all_eps = sorted(np.unique(E).tolist())
pool = [e for e in all_eps if e != EP]                      # 严格排除 held-out ep2238
train_eps = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
assert EP not in train_eps, "ep2238 必须 held-out"
keepT = np.isin(E, train_eps); Et = E[keepT]; FRt = FR[keepT]; featT = feat[keepT]
print(f"  train {len(train_eps)} eps {len(Et)} frames (2238 held-out); PCA...", flush=True)
pca = PCA(128, random_state=0).fit(l2(featT[rng.choice(len(featT), min(20000, len(featT)), replace=False)].astype(np.float32)))


def img128(fh):
    return l2((l2(fh.astype(np.float32)) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T)


IMG = img128(featT)


def read_pos(e, order_fr):
    st = np.stack(pd.read_parquet(KAI / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                  columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    return st[np.minimum(order_fr, len(st) - 1)]


print(f"  [{time.time()-t0:.0f}s] 读训练集 proprio...", flush=True)
POS = np.zeros((len(Et), 14), np.float32)
for e in train_eps:
    m = np.where(Et == e)[0]; o = m[np.argsort(FRt[m])]; POS[o] = read_pos(e, FRt[m][np.argsort(FRt[m])])
SMU = POS.mean(0); SSD = POS.std(0) + 1e-6
JOINT = np.concatenate([IMG, l2((POS - SMU) / SSD)], 1).astype(np.float32)
D = JOINT.shape[1]; NC = len(train_eps)
T = np.zeros(len(Et), np.float32)
for e in train_eps:
    m = np.where(Et == e)[0]; o = m[np.argsort(FRt[m])]; T[o] = np.linspace(0, 1, len(o))

print(f"  [{time.time()-t0:.0f}s] BayesianGMM on {D}D...", flush=True)
bg = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                             max_iter=120, random_state=0).fit(JOINT[rng.choice(len(JOINT), min(80000, len(JOINT)), replace=False)])
labs = bg.predict(JOINT); C = []; P = []
for k in range(40):
    m = labs == k
    if m.sum() < 20:
        continue
    if len(set(Et[m].tolist())) / NC >= 0.5:
        C.append(JOINT[m].mean(0)); P.append(float(np.median(T[m])))
C = l2(np.array(C, np.float32)); P = np.array(P); lam = 16.0 * FPS / 3.0
print(f"  [{time.time()-t0:.0f}s] M={len(C)} milestones; polyline teacher...", flush=True)
DATA = []
for e in train_eps:
    m = np.where(Et == e)[0]; o = m[np.argsort(FRt[m])]; f = JOINT[o]; t = T[o]
    DATA.append((f, daw(f, C, P, lam), t))
tc = np.nanmean([cc(v, t) for f, v, t in DATA]); print(f"  teacher-vs-T corr={tc:.3f} ({time.time()-t0:.0f}s)", flush=True)


class G(nn.Module):
    def __init__(s, h=256, L=2):
        super().__init__(); s.g = nn.GRU(D, h, L, batch_first=True)
        s.head = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(s, x, ln):
        p = nn.utils.rnn.pack_padded_sequence(x, ln.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.g(p); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return torch.sigmoid(s.head(o)).squeeze(-1)


def batches(pl, bs=24):
    ix = sorted(range(len(pl)), key=lambda i: len(pl[i][0]))
    for kk in range(0, len(ix), bs):
        gp = [pl[i] for i in ix[kk:kk + bs]]; L = max(len(a[0]) for a in gp); B = len(gp)
        X = np.zeros((B, L, D), np.float32); Y = np.zeros((B, L), np.float32); M = np.zeros((B, L), np.float32); ln = np.zeros(B, int)
        for b, (f, v, t) in enumerate(gp):
            nn_ = len(f); X[b, :nn_] = f; Y[b, :nn_] = v; M[b, :nn_] = 1; ln[b] = nn_
        yield torch.tensor(X, device=DEV), torch.tensor(Y, device=DEV), torch.tensor(M, device=DEV), torch.tensor(ln)


net = G().to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
print("训练因果 GRU(蒸馏 polyline teacher, 2238 未见)...", flush=True)
for epn in range(40):
    net.train(); pl = DATA[:]; rng.shuffle(pl)
    for X, Y, M, ln in batches(pl):
        pr = net(X, ln); loss = (((pr - Y) ** 2) * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
net.eval()


@torch.no_grad()
def pred(f, warm=20):
    fw = np.concatenate([f[:1].repeat(warm, 0), f])
    return net(torch.tensor(fw[None], device=DEV), torch.tensor([len(fw)]))[0].cpu().numpy()[warm:]


# ══════ 2. held-out ep2238 因果推理 ══════
mE = np.where(E == EP)[0]; oE = mE[np.argsort(FR[mE])]; frE = FR[mE][np.argsort(FR[mE])]
IMG_E = img128(feat[oE]); POS_E = read_pos(EP, frE)
JOINT_E = np.concatenate([IMG_E, l2((POS_E - SMU) / SSD)], 1).astype(np.float32)
gru = pred(JOINT_E)                       # 逐帧 progress(严格因果 + warmup)
nB = len(gru)
print(f"  ep2238 held-out: {nB} bank frames, gru end={gru[-1]:.2f} ({time.time()-t0:.0f}s)", flush=True)

# ══════ 3. KAI0-AE value(现成 Stage-2 输出) ══════
dQ = pd.read_parquet(Q5 / f"data/chunk-{EP // CSQ:03d}/episode_{EP:06d}.parquet",
                     columns=["absolute_value", "relative_advantage"])
ae = dQ["absolute_value"].to_numpy().astype(float)
ra = np.clip(dQ["relative_advantage"].to_numpy().astype(float), -1, 1)
n = min(nB, len(ae)); gru = gru[:n]; ae = ae[:n]; ra = ra[:n]; frE = frE[:n]

# 分类信号(各按部署真实 P/N)
gru_sig = np.clip(advantage(gru, W), -1, 1); gcls = three(gru_sig)   # CRAVE: Δvalue 派生
acls = three(ra)                                                    # AE: relative_advantage 直出
fg = {c: (gcls == c).mean() for c in (1, 0, -1)}; fa = {c: (acls == c).mean() for c in (1, 0, -1)}
gflip = flips(gru_sig); aflip = flips(ra)
print(f"  GRU  neg{fg[-1]:.0%} flips={gflip}  |  AE neg{fa[-1]:.0%} flips={aflip}", flush=True)

# ══════ 4. 背景面板: 双 value 曲线 + 两条分类条 ══════
x = frE / 30.0
gstrip = np.array([RGB[c] for c in gcls])[None]; astrip = np.array([RGB[c] for c in acls])[None]
PFIG = plt.figure(figsize=(9.5, 5.0), dpi=100)
gs = PFIG.add_gridspec(3, 1, height_ratios=[1, 0.20, 0.20], hspace=0.42)
axv = PFIG.add_subplot(gs[0])
axv.plot(x, gru, color="#2ca02c", lw=2.0, label=f"online GRU progress (causal, end {gru[-1]:.2f})")
axv.plot(x, ae, color="#d62728", lw=1.3, alpha=.85, label=f"KAI0-AE value (end {ae[-1]:.2f}, max {ae.max():.2f})")
axv.axhline(1, color="#2ca02c", ls=":", lw=1, alpha=.5); axv.axhline(0, color="k", lw=.5)
axv.set_ylim(min(ae.min(), -0.05), 1.12); axv.set_xlim(0, x[-1]); axv.set_ylabel("value", fontsize=9); axv.tick_params(labelsize=7)
axv.legend(fontsize=8.5, loc="center left"); axv.grid(alpha=.22)
axv.set_title("ep2238 (success fold) — online GRU smooth monotone 0->1  vs  KAI0-AE jittery + end-fold false drop", fontsize=9.5)
axg = PFIG.add_subplot(gs[1]); axg.imshow(gstrip, aspect="auto", extent=[0, x[-1], 0, 1]); axg.set_yticks([]); axg.set_xlim(0, x[-1]); axg.tick_params(labelsize=6)
axg.set_ylabel("online GRU", fontsize=7.5, rotation=0, ha="right", va="center")
axg.set_title(f"online GRU P/N (Δprogress): neg {fg[-1]:.0%} · sign-flips {gflip}", fontsize=8.5, color="#2ca02c")
axa = PFIG.add_subplot(gs[2]); axa.imshow(astrip, aspect="auto", extent=[0, x[-1], 0, 1]); axa.set_yticks([]); axa.set_xlim(0, x[-1]); axa.set_xlabel("seconds", fontsize=8); axa.tick_params(labelsize=6)
axa.set_ylabel("KAI0-AE", fontsize=7.5, rotation=0, ha="right", va="center")
axa.set_title(f"KAI0-AE P/N (native relative_advantage): neg {fa[-1]:.0%} · sign-flips {aflip}", fontsize=8.5, color="#d62728")
PFIG.suptitle("Online causal GRU inference  vs  KAI0-AE supervised — value + P/N bars (POS green / NORMAL gray / NEG red)", fontsize=10.5)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a):
    bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh


def xpx(m, sec):
    x0, x1, _, _, xl, xh, _, _ = m; return int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp))


def yp(m, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m; return int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp))


def ysp(m):
    _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))


MV, MG, MA = pm(axv), pm(axg), pm(axa); plt.close(PFIG)
span = (ysp(MV)[0], ysp(MA)[1])

# ══════ 5. 逐帧渲染(相机左 + 面板右, 双游标) ══════
vid = KAI / f"videos/chunk-{EP // 1000:03d}/observation.images.top_head/episode_{EP:06d}.mp4"
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
omp4 = str(OUT / f"online_vs_ae_ep{EP}.mp4"); mid = n // 2
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"
stv.options = {"preset": "veryfast", "crf": "23"}
cobj = av.open(str(vid)); i = 0
for fr in cobj.decode(video=0):
    if i >= n:
        break
    panel = PANEL.copy(); sec = x[i]
    px = xpx(MV, sec); cv2.line(panel, (px, span[0]), (px, span[1]), (40, 40, 40), 1)
    cv2.circle(panel, (xpx(MV, sec), yp(MV, float(gru[i]))), 6, BGR[gcls[i]], -1); cv2.circle(panel, (xpx(MV, sec), yp(MV, float(gru[i]))), 6, (30, 100, 30), 1)
    cv2.circle(panel, (xpx(MV, sec), yp(MV, float(ae[i]))), 6, BGR[acls[i]], -1); cv2.circle(panel, (xpx(MV, sec), yp(MV, float(ae[i]))), 6, (120, 20, 20), 1)
    cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (330, 80), (0, 0, 0), -1)
    cv2.putText(cam2, f"ep{EP} {i}/{n} (held-out)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"GRU: {NAME[gcls[i]]}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BGR[gcls[i]][::-1], 2, cv2.LINE_AA)
    cv2.putText(cam2, f"AE : {NAME[acls[i]]}", (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BGR[acls[i]][::-1], 2, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
    if i == mid:
        cv2.imwrite(omp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
        oc.mux(pkt)
    i += 1
    if i % 1500 == 0:
        print(f"  vid {i}/{n}", flush=True)
cobj.close()
for pkt in stv.encode():
    oc.mux(pkt)
oc.close()
print(f"SAVED {omp4} {i}f | GRU neg{fg[-1]:.0%}/flips{gflip}  AE neg{fa[-1]:.0%}/flips{aflip}", flush=True)
print("ONLINE_VS_AE_EP2238_DONE", flush=True)
