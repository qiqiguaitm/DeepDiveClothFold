#!/usr/bin/env python
"""普适性验证: BGMM(coverage 永久关) + 时间注入 w + 浓度 γ, 一套参数跑遍 LIBERO-40(+kai0)。
coverage 的两件活儿改由普适机制接管:
  ① 杀杂簇  = min_members(20) + mode_split(时间多峰拆)
  ② 跨ep一致+铺满全程 = 时间注入 F=[l2(img128) ⊕ w·t](平坦尾段也按时间切, 簇天然跨ep)
评估换成"覆盖均匀性"(不止末段):
  H   = 归一化覆盖熵 -Σ s·log s / log M ∈[0,1] (1=完全均匀, 对M归一可跨任务比)
  mx  = max 段占比 (大空洞暴露, 末段60%这种直接现形)
  M   = 涌现 milestone 数
  rec = 每 milestone 平均跨 episode 覆盖率 (证明时间注入不用 coverage 也跨ep)
用法: srpo python ablate_bgmm_universal.py --w 1.0 --gamma 1.0 [--cov 0.0] [--with_kai0]
"""
import os, sys, glob, argparse, time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

ROOT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
KAI  = "/vePFS/tim/workspace/deepdive_kai0/lmvla/crave/data/kai_dinov3base"

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def mode_split(Tc, nbins=30):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / (h.sum() + 1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0,i-1)] and hs[i] >= hs[min(nbins-1,i+1)] and hs[i] >= 0.10*hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p]-c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6*min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    if len(final) <= 1: return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = [c[a+int(np.argmin(hs[a:b+1]))] for a, b in zip(final[:-1], final[1:])]
    edges = [0.0]+cuts+[1.0]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (Tc >= lo) & (Tc < hi)
        if m.sum() >= 5: out.append((float(np.median(Tc[m])), m))
    return out if out else [(float(np.median(Tc)), np.ones(len(Tc), bool))]

def seg_metrics(path):
    """path: 每帧 milestone 索引(cummax 单调). 返回 (M, H, maxfrac)."""
    b = np.concatenate([[0], np.where(np.diff(path) != 0)[0]+1, [len(path)]])
    fr = np.diff(b).astype(float)/len(path); fr = fr[fr > 0]
    M = len(fr)
    if M <= 1: return M, 0.0, float(fr.max()) if len(fr) else 1.0
    H = float(-(fr*np.log(fr)).sum()/np.log(M))
    return M, H, float(fr.max())

def discover(F, Tv, Ev, offs, teps, cfg):
    """F已含时间注入. 返回 有序 (C, vals) 里程碑."""
    NC = len(teps)
    if cfg["km"]:
        K = 8
        km = KMeans(K, n_init=3, random_state=0).fit(F)
        tpos = np.array([Tv[km.labels_ == c].mean() if (km.labels_ == c).any() else 9 for c in range(K)])
        order = np.argsort(tpos); return km.cluster_centers_[order].astype(np.float32), np.sort(tpos), km.labels_
    bg = BayesianGaussianMixture(n_components=40, covariance_type="diag",
                                 weight_concentration_prior=cfg["gamma"], n_init=cfg["ninit"],
                                 max_iter=cfg["maxiter"], random_state=0).fit(F)
    labs = bg.predict(F)
    cand = np.array([F[labs == k].mean(0) for k in range(40) if (labs == k).sum() >= 20], np.float32)
    if len(cand) == 0: return None, None, None
    asg = np.linalg.norm(F[:, None]-cand[None], axis=2).argmin(1)
    tv = []
    for ki in range(len(cand)):
        mk = asg == ki
        if mk.sum() < 20: continue
        for mv, sub in mode_split(Tv[mk]):
            idx = np.where(mk)[0][sub]
            cov = len(set(Ev[idx].tolist()))/NC
            if cov >= cfg["cov"]:                       # cov 默认 0.0 = 全留
                tv.append((float(np.median(Tv[idx])), F[idx].mean(0), cov))
    if not tv: return None, None, None
    tv.sort(key=lambda t: t[0])
    return np.array([t[1] for t in tv], np.float32), np.array([t[0] for t in tv]), None

def run_task(gd, cfg):
    teps = list(gd)
    IMG = np.concatenate([gd[e] for e in teps])
    pca = PCA(128, random_state=0).fit(IMG); img_f = l2(pca.transform(IMG))
    Tv = np.concatenate([np.linspace(0, 1, len(gd[e])) for e in teps]).astype(np.float32)
    Ev = np.concatenate([np.full(len(gd[e]), e) for e in teps])
    lens = [len(gd[e]) for e in teps]; offs = np.cumsum([0]+lens)
    F = np.concatenate([img_f, cfg["w"]*Tv[:, None]], 1).astype(np.float32)   # 时间注入
    C, vals, _ = discover(F, Tv, Ev, offs, teps, cfg)
    if C is None or len(C) < 1: return None
    # 每 milestone 跨-ep 覆盖(诊断: 时间注入是否天然跨ep)
    asg_all = np.linalg.norm(F[:, None]-C[None], axis=2).argmin(1)
    rec = np.mean([len(set(Ev[asg_all == ki].tolist()))/len(teps) for ki in range(len(C))])
    out = []
    for i in range(len(teps)):
        Fq = F[offs[i]:offs[i+1]]; n = len(Fq)
        raw = np.linalg.norm(Fq[:, None]-C[None], axis=2).argmin(1)
        w = 5; sm = raw.copy()
        for j in range(n): sm[j] = int(np.median(raw[max(0, j-w):j+w+1]))
        path = np.maximum.accumulate(sm)
        out.append(seg_metrics(path))
    Ms, Hs, mxs = zip(*out)
    return int(np.median(Ms)), float(np.median(Hs)), float(np.median(mxs)), float(rec)

def load_libero():
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    meta = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar])
    ep2task = meta.groupby("episode_index")["task_index"].first().to_dict()
    from collections import defaultdict
    task_eps = defaultdict(list)
    for e, t in ep2task.items():
        if os.path.exists(f"{FEAT}/ep{e}.npz"): task_eps[t].append(e)
    tasks = []
    for t, eps in sorted(task_eps.items()):
        if len(eps) < 5: continue
        gd = {e: np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in eps}
        tasks.append((f"libero{t}", gd))
    return tasks

def load_kai0(cap=60):
    idx = np.load(f"{KAI}/index.npz"); E = idx["E"]; FR = idx["FR"]
    feat = np.zeros((len(E), 768), np.float16)
    for sh in sorted(glob.glob(f"{KAI}/shard_*.npz")):
        s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s.files else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
    eps = sorted(np.unique(E).tolist())[:cap]
    gd = {}
    for e in eps:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; gd[e] = feat[o].astype(np.float32)
    return [("kai0", gd)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--cov", type=float, default=0.0)          # 默认关
    ap.add_argument("--km", action="store_true")               # 对照: 固定K=8
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--ninit", type=int, default=4)
    ap.add_argument("--with_kai0", action="store_true")
    a = ap.parse_args()
    cfg = dict(w=a.w, gamma=a.gamma, cov=a.cov, km=a.km, maxiter=a.maxiter, ninit=a.ninit)
    tag = f"{'KM8' if a.km else f'BGMM g{a.gamma}'} w{a.w} cov{a.cov}"
    t0 = time.time()
    tasks = load_libero()
    if a.with_kai0: tasks += load_kai0()
    print(f"[{tag}] {len(tasks)} tasks loaded ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for name, gd in tasks:
        r = run_task(gd, cfg)
        if r is None: print(f"  [{tag}] {name}: EMPTY", flush=True); rows.append((0, 0.0, 1.0, 0.0)); continue
        M, H, mx, rec = r; rows.append((M, H, mx, rec))
        print(f"  [{tag}] {name}: M={M} H={H:.2f} maxfrac={mx:.2f} recur={rec:.2f}", flush=True)
    Ms = [r[0] for r in rows]; Hs = [r[1] for r in rows]; mxs = [r[2] for r in rows]; recs = [r[3] for r in rows]
    print(f"[SUMMARY {tag}] tasks={len(rows)} | M med={int(np.median(Ms))} | "
          f"H med={np.median(Hs):.3f} min(worst)={np.min(Hs):.3f} | "
          f"maxfrac med={np.median(mxs):.3f} max(worst)={np.max(mxs):.3f} | recur med={np.median(recs):.2f} "
          f"| bigGap(maxfrac>0.4) {np.mean(np.array(mxs)>0.4)*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {tag}", flush=True)

if __name__ == "__main__":
    main()
