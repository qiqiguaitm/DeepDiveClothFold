#!/usr/bin/env python
"""受控消融: 在 libero_10 (10 任务) 上, 从旧版 baseline 逐项开启组件, 看各自对分割质量的边际贡献。
组件 3 轴(baseline→finalarch):
  F1 特征: img-only(768) → img128⊕proprio(1:1)
  F2 聚类: KMeans(时间序) → BayesianGMM+per-mode coverage
  F3 读出+建对: argmin+中值+cummax+丢末段 → 双锚Viterbi+self-loop
config: baseline / +proprio(仅F1) / +bgmm(仅F2) / +dualvit(仅F3) / full(全开)
指标(每任务, 中位汇总): M(里程碑数) · last_seg_frac(末段占比) · prog_corr(里程碑值 vs 时间 Spearman)
用法: srpo python ablate_libero_seg.py --config full
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = "/home/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
MIN_COV = 0.50; LAM = 16.0
# libero_10 (eval task 0-9) 的描述关键片段
LIBERO10 = ["alphabet soup and the tomato sauce in the basket", "cream cheese box and the butter",
            "turn on the stove and put the moka", "black bowl in the bottom drawer",
            "white mug on the left plate and", "book and place it in the back",
            "white mug on the plate and put the chocolate", "alphabet soup and the cream cheese",
            "both moka pots on the stove", "yellow and white mug in the microwave"]

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def mode_split(Tc, nbins=30):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / (h.sum() + 1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0, i-1)] and hs[i] >= hs[min(nbins-1, i+1)] and hs[i] >= 0.10 * hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p] - c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6 * min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    if len(final) <= 1: return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = [c[a + int(np.argmin(hs[a:b+1]))] for a, b in zip(final[:-1], final[1:])]
    edges = [0.0] + cuts + [1.0]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (Tc >= lo) & (Tc < hi)
        if m.sum() >= 5: out.append((float(np.median(Tc[m])), m))
    return out if out else [(float(np.median(Tc)), np.ones(len(Tc), bool))]

def viterbi_dp(emit, vals, lam=LAM):
    nb = len(vals); pen = lam * np.abs(vals[:, None] - vals[None])
    cost = np.full(nb, 1e9); cost[0] = emit[0, 0]; BP = np.zeros((len(emit), nb), int)
    for j in range(1, len(emit)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = emit[j] + tr[np.arange(nb), kk]; BP[j] = kk
    cost[nb - 1] -= 2; s = int(cost.argmin()); path = np.zeros(len(emit), int); path[-1] = s
    for j in range(len(emit) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return path

def run_task(gd, proprio, cfg):
    """gd: {e:[N,768]}, proprio: {e:[N,8]}. 返回 per-episode (M, last_seg_frac, prog_corr)."""
    teps = list(gd)
    use_prop = cfg in ("proprio", "full")
    use_bgmm = cfg in ("bgmm", "full")
    use_dual = cfg in ("dualvit", "full")
    # ---- 特征 ----
    IMG = np.concatenate([gd[e] for e in teps])
    if use_prop:
        pca = PCA(128, random_state=0).fit(IMG); img_f = l2(pca.transform(IMG))
        ST = np.concatenate([proprio[e] for e in teps]).astype(np.float32)
        ST = (ST - ST.mean(0)) / (ST.std(0) + 1e-8)
        F = np.concatenate([img_f, l2(ST)], 1)
    else:
        F = l2(IMG)
    Fn = l2(F)
    Tv = np.concatenate([np.linspace(0, 1, len(gd[e])) for e in teps])
    Ev = np.concatenate([np.full(len(gd[e]), e) for e in teps]); NC = len(teps)
    lens = [len(gd[e]) for e in teps]; offs = np.cumsum([0] + lens)
    # ---- 聚类 → 有序质心 C + 值 vals(时间序) ----
    if use_bgmm:
        bg = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                                     n_init=1, max_iter=150, random_state=0).fit(Fn)
        labs = bg.predict(Fn)
        cand = np.array([Fn[labs == k].mean(0) for k in range(40) if (labs == k).sum() >= 20], np.float32)
        if len(cand) == 0: return []
        asg = np.linalg.norm(Fn[:, None] - cand[None], axis=2).argmin(1)
        tv = []
        for ki in range(len(cand)):
            mk = asg == ki
            if mk.sum() < 20: continue
            for mv, sub in mode_split(Tv[mk]):
                if len(set(Ev[mk][sub].tolist())) / NC >= MIN_COV: tv.append((float(np.median(Tv[mk][sub])), cand[ki]))
        if not tv: return []
        tv.sort(key=lambda t: t[0]); vals = np.array([t[0] for t in tv]); C = np.array([t[1] for t in tv], np.float32)
    else:
        K = int(np.clip(round(0.55 * np.sqrt(len(Fn))), 8, 40))
        km = KMeans(K, n_init=3, random_state=0).fit(Fn); cen = km.cluster_centers_
        tpos = np.array([Tv[km.labels_ == c].mean() if (km.labels_ == c).any() else 9 for c in range(K)])
        # 覆盖筛(与旧p1一致: KMeans无coverage, 保留全部按时间序)
        order = np.argsort(tpos); C = cen[order].astype(np.float32); vals = np.sort(tpos)
    # 端点锚(仅 dual 用)
    s_anc = l2(np.mean([Fn[offs[i]:offs[i]+3].mean(0) for i in range(len(teps))], 0))
    e_anc = l2(np.mean([Fn[offs[i+1]-3:offs[i+1]].mean(0) for i in range(len(teps))], 0))
    out = []
    for i, e in enumerate(teps):
        Fq = Fn[offs[i]:offs[i+1]]; n = len(Fq)
        if use_dual:
            Cc = np.concatenate([s_anc[None], C, e_anc[None]], 0); vv = np.concatenate([[0.0], vals, [1.0]])
            o = np.argsort(vv); Cc = Cc[o]; vv = vv[o]
            emit = np.linalg.norm(Fq[:, None] - Cc[None], axis=2)
            path = viterbi_dp(emit, vv); pv = vv[path]
        else:  # argmin + 中值平滑 + cummax
            raw = np.linalg.norm(Fq[:, None] - C[None], axis=2).argmin(1)
            w = 5; sm = raw.copy()
            for j in range(n): sm[j] = int(np.median(raw[max(0, j-w):j+w+1]))
            path = np.maximum.accumulate(sm); pv = vals[np.clip(path, 0, len(vals)-1)]
        M = len(set(path.tolist()))
        fl = np.where(path == path[-1])[0][0]; last_frac = (n - fl) / n
        pc = spearmanr(pv, np.arange(n)).correlation
        out.append((M, last_frac, pc if pc == pc else 0))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["baseline", "proprio", "bgmm", "dualvit", "full"])
    args = ap.parse_args()
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    meta = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar])
    ep2task = meta.groupby("episode_index")["task_index"].first().to_dict()
    tdf = pd.read_parquet(f"{ROOT}/meta/tasks.parquet")  # index=name, col task_index
    name2idx = {n: int(r["task_index"]) for n, r in tdf.iterrows()}
    lib10 = []
    for frag in LIBERO10:
        hit = [v for n, v in name2idx.items() if frag.lower() in n.lower()]
        lib10.append(hit[0] if hit else None)
    from collections import defaultdict
    task_eps = defaultdict(list)
    for e, t in ep2task.items():
        if os.path.exists(f"{FEAT}/ep{e}.npz"): task_eps[t].append(e)
    want = set(e for t in lib10 if t is not None for e in task_eps[t])
    state_ep = {}
    for p in dpar:
        df = pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"])
        df = df[df.episode_index.isin(want)]
        for e, g in df.groupby("episode_index"):
            state_ep[e] = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())
    rows = []
    for ti, midx in enumerate(lib10):
        if midx is None: continue
        teps = task_eps[midx]
        gd = {e: np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in teps}
        prop = {}
        for e in teps:
            n = len(gd[e]); st = state_ep[e][::2][:n]
            if len(st) < n: st = np.concatenate([st, np.repeat(st[-1:], n - len(st), 0)])
            prop[e] = st
        res = run_task(gd, prop, args.config)
        if not res: rows.append((ti, 0, 1.0, 0)); continue
        Ms = [r[0] for r in res]; lf = [r[1] for r in res]; pc = [r[2] for r in res]
        rows.append((ti, int(np.median(Ms)), float(np.median(lf)), float(np.median(pc))))
        print(f"  [{args.config}] libero10-task{ti}(midx{midx}): M={int(np.median(Ms))} last_frac={np.median(lf):.2f} prog_corr={np.median(pc):.2f}", flush=True)
    Ms = [r[1] for r in rows]; lf = [r[2] for r in rows]; pc = [r[3] for r in rows]
    print(f"[SUMMARY {args.config}] 10任务中位: M={int(np.median(Ms))} last_frac={np.median(lf):.3f} prog_corr={np.median(pc):.3f}", flush=True)

if __name__ == "__main__":
    main()
