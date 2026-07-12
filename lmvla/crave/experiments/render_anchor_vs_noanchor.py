#!/usr/bin/env python
"""§2.4 双锚配图: 同一批 ep 上【带双锚 vs 不带双锚】的 Viterbi 读出对比(一图内).

关键洞察: 很多成功 demo 的 raw(不带双锚)末端 *到不了* 顶(末帧最近的 milestone 只有 ~0.6),
于是 per-ep norm01 会把这段低平台【硬拉到 1】= 假登顶 + 整条被抬高、丢掉真实的末端登顶过程;
带双锚强制末帧=1 → DP 沿高 milestone(0.6→0.83→0.96)真实登顶, 中段不被抬高。
选 ep = raw 末端偏低(anchor 帮助最大)的代表性 ep, 让优势可见。

三条线: 灰=不带双锚 raw / 橙=raw+per-ep norm01 / 绿=带双锚(钉 0->1)。
首次运行算 milestone 并缓存到 temp/anchor_cache.npz; 之后秒级复算(改选图/画法不必重跑 GMM)。
Run: PYTHONPATH=src /home/tim/miniconda3/envs/srpo/bin/python experiments/render_anchor_vs_noanchor.py
Out: temp/anchor_vs_noanchor.png
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
rng = np.random.RandomState(0)
CAP = 1000
FPS = 30.0
CSQ = 1000
KAI = REPO / "kai0/data/Task_A/kai0_base"
CACHE = REPO / "lmvla/crave/temp/anchor_cache.npz"


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cc(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def vstep(F, C, P, lam, anchored):
    """Viterbi milestone step path. anchored=True: 加起/终点锚并强制首=0/末=1; False: 自由端点。"""
    if anchored:
        sC = l2(F[:3].mean(0)[None])[0]; eC = l2(F[-3:].mean(0)[None])[0]
        C2 = np.vstack([C, sC, eC]); Pp = np.concatenate([P, [0.0], [1.0]])
    else:
        C2 = C; Pp = P.copy()
    bins = np.unique(np.concatenate([[0.0], Pp, [1.0]])); nb = len(bins)
    cb = [int(np.searchsorted(bins, v)) for v in Pp]; pen = lam * np.abs(bins[:, None] - bins[None])
    de = np.linalg.norm(F[:, None] - C2[None], axis=2); em = np.full((len(F), nb), 1e3)
    for ti in range(len(Pp)):
        em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    if anchored:
        cost = np.full(nb, 1e9); cost[0] = em[0, 0]
    else:
        cost = em[0].copy()
    BP = np.zeros((len(F), nb), int)
    for j in range(1, len(F)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = (nb - 1) if anchored else int(cost.argmin())
    path = np.zeros(len(F), int); path[-1] = si
    for j in range(len(F) - 2, -1, -1):
        si = BP[j + 1][si]; path[j] = si
    return bins[path]


# ══════ milestone discovery(带缓存) ══════
if CACHE.exists():
    print(f"加载缓存 {CACHE.name} ...", flush=True); t0 = time.time()
    z = np.load(CACHE); E = z["E"]; FR = z["FR"]; JOINT = z["JOINT"].astype(np.float32)
    C = z["C"]; P = z["P"]; lam = float(z["lam"])
else:
    from sklearn.decomposition import PCA
    from sklearn.mixture import BayesianGaussianMixture
    print("加载 kai base bank...", flush=True); t0 = time.time()
    d = REPO / "lmvla/crave/data/kai_dinov3base"; idx = np.load(d / "index.npz"); E = idx["E"]; FR = idx["FR"]
    feat = np.zeros((len(E), 768), np.float16)
    for sh in sorted(d.glob("shard_*.npz")):
        s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
    eps0 = sorted(np.unique(E).tolist())
    if len(eps0) > CAP:
        eps0 = [eps0[i] for i in sorted(rng.choice(len(eps0), CAP, replace=False))]
    keep = np.isin(E, eps0); E = E[keep]; FR = FR[keep]; feat = feat[keep]
    print(f"  {len(eps0)} eps {len(E)} frames; PCA...", flush=True)
    pca = PCA(128, random_state=0).fit(l2(feat[rng.choice(len(feat), min(20000, len(feat)), replace=False)].astype(np.float32)))
    IMG = l2((l2(feat.astype(np.float32)) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T)
    print(f"  [{time.time()-t0:.0f}s] proprio...", flush=True)
    POS = np.zeros((len(E), 14), np.float32)
    for e in eps0:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; fr = FR[m][np.argsort(FR[m])]
        st = np.stack(pd.read_parquet(KAI / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                      columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
        POS[o] = st[np.minimum(fr, len(st) - 1)]
    SMU = POS.mean(0); SSD = POS.std(0) + 1e-6
    JOINT = np.concatenate([IMG, l2((POS - SMU) / SSD)], 1).astype(np.float32)
    NC = len(eps0)
    Tt = np.zeros(len(E), np.float32)
    for e in eps0:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; Tt[o] = np.linspace(0, 1, len(o))
    print(f"  [{time.time()-t0:.0f}s] BayesianGMM...", flush=True)
    bg = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                                 max_iter=120, random_state=0).fit(JOINT[rng.choice(len(JOINT), min(80000, len(JOINT)), replace=False)])
    labs = bg.predict(JOINT); Cl = []; Pl = []
    for k in range(40):
        m = labs == k
        if m.sum() < 20:
            continue
        if len(set(E[m].tolist())) / NC >= 0.5:
            Cl.append(JOINT[m].mean(0)); Pl.append(float(np.median(Tt[m])))
    C = l2(np.array(Cl, np.float32)); P = np.array(Pl); lam = 16.0 * FPS / 3.0
    np.savez_compressed(CACHE, E=E, FR=FR, JOINT=JOINT.astype(np.float16), C=C, P=P, lam=lam)
    print(f"  cached -> {CACHE.name}", flush=True)

eps = sorted(np.unique(E).tolist())
T = np.zeros(len(E), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; T[o] = np.linspace(0, 1, len(o))
print(f"  M={len(C)} milestones, sorted P={np.round(np.sort(P),2)} ({time.time()-t0:.0f}s)", flush=True)

# ══════ 对全 1000 条 ep 算 双锚/norm01 与监督 progress_gt 的 corr ══════
import json
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]
lens = {e: int((E == e).sum()) for e in eps}
rows = []   # (e, corr_anchor, corr_norm01, n, curves)
for e in eps:
    if lens[e] < 300:
        continue
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; F = JOINT[o]; fr = FR[m][np.argsort(FR[m])]
    try:
        gt = pd.read_parquet(Q5 / f"data/chunk-{e // csQ:03d}/episode_{e:06d}.parquet", columns=["progress_gt"])["progress_gt"].to_numpy().astype(float)
    except Exception:
        continue
    gt = gt[np.minimum(fr, len(gt) - 1)]
    raw = vstep(F, C, P, lam, False); nrm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9); anc = vstep(F, C, P, lam, True)
    ca, cn = cc(anc, gt), cc(nrm, gt)
    if np.isnan(ca) or np.isnan(cn):
        continue
    rows.append((e, ca, cn, lens[e], gt, raw, nrm, anc))
CA = np.array([r[1] for r in rows]); CN = np.array([r[2] for r in rows])
print(f"  n={len(rows)} eps vs progress_gt | double-anchor mean {CA.mean():.3f}/med {np.median(CA):.3f} | "
      f"norm01 mean {CN.mean():.3f}/med {np.median(CN):.3f} | anchor赢 {np.mean(CA>CN)*100:.0f}% | mean gain {(CA-CN).mean():+.3f}", flush=True)

# 失败案例: norm01 corr 崩(cn 低)而双锚仍高 → 取 2 条最长的
fail = [r for r in rows if r[2] < 0.55 and r[1] > 0.8 and r[3] > 400]
fail = sorted(fail, key=lambda r: -r[3])[:2]
if len(fail) < 2:
    fail = sorted(rows, key=lambda r: -(r[1] - r[2]))[:2]
print(f"  fail-case eps {[r[0] for r in fail]} (norm01 corr {[round(r[2],2) for r in fail]} vs anchor {[round(r[1],2) for r in fail]})", flush=True)

# ══════ 画图: 2 条失败案例曲线(norm01 崩 / 双锚稳) ══════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, r in zip(axes, fail):
    e, ca, cn, n, gt, raw, nrm, anc = r
    ax.plot(gt, color="#333", lw=1.9, label="supervised progress_gt")
    ax.plot(nrm, color="#d98b00", lw=1.6, alpha=.9, label=f"no-anchor + per-ep norm01 (corr {cn:.2f})")
    ax.plot(anc, color="#2ca02c", lw=2.4, label=f"double-anchor (corr {ca:.2f})")
    ax.set_title(f"fail case · kai ep{e}", fontsize=11); ax.set_ylim(-.05, 1.06); ax.grid(alpha=.25)
    ax.set_xlabel("frame", fontsize=9); ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle(f"Double-anchor vs no-anchor(+per-ep norm01): fail cases where norm01 collapses "
             f"(small raw range -> noise blown into square wave), double-anchor stays faithful to GT  ·  "
             f"overall mean corr vs GT: anchor {CA.mean():.3f} vs norm01 {CN.mean():.3f}", fontsize=9.2)
fig.tight_layout()
outp = REPO / "lmvla/crave/temp/anchor_vs_noanchor.png"; outp.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outp, dpi=115, bbox_inches="tight")
print(f"SAVED {outp} ({time.time()-t0:.0f}s)", flush=True)
print("ANCHOR_VS_NOANCHOR_DONE", flush=True)
