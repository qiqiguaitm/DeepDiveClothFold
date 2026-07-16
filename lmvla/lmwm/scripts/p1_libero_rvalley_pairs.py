#!/usr/bin/env python
"""V5(a) 结构接法: 用 recurrence 场构造 world-model 训练对(替代 CRAVE milestone)。
每 ep 按 r-低谷分段; 目标 = 下一段的 r-脊(canonical 收敛点, V3b 证明>边界>固定); 末段锚末帧(不丢)。
= 修 milestone+1 的两个病根: ① 目标从"边界(分歧点)"改"脊(canonical)"; ② 末段不再丢弃。
输出与 p1_libero_milestone_pairs 同格式 → p1_train_lmwm_libero 可直接用(仅换 PAIRS 路径)。
用法: srpo python p1_libero_rvalley_pairs.py   → lmwm/data/libero_rvalley/pairs.npz
"""
import os, glob
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
OUT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_rvalley"
THR = 0.03
def l2(x): return x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)

def r_and_segments(gd):
    """返回 每 ep: r[n], 段边界 seg(含0,n), 段脊 ridge(全局帧内局部 idx)。"""
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    lens = [len(gd[e]) for e in eps]; offs = np.cumsum([0]+lens)
    D = cdist(F, F); dmin = np.full((len(F), ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, ep == j].min(1)
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    r = (np.exp(-dmin**2/(2*sig*sig))*other).sum(1)/(ne-1)
    res = {}
    for i, e in enumerate(eps):
        s, en = offs[i], offs[i+1]; n = en-s; rr = r[s:en]
        v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n//12))
        seg = [0]+list(v)+[n]; ridge = [a+int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]
        res[e] = (seg, ridge, n)
    return res

def main():
    import sys
    files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    ep2task = pd.read_parquet(dpar[0], columns=["episode_index", "task_index"]).groupby("episode_index")["task_index"].first().to_dict()
    ep_gist = {}
    for f in files:
        e = int(os.path.basename(f)[2:-4]); ep_gist[e] = np.load(f)["grid"].astype(np.float32).mean(1)
    from collections import defaultdict
    task_eps = defaultdict(list)
    for e in ep_gist: task_eps[ep2task.get(e, -1)].append(e)
    print(f"[tasks] {len(task_eps)} 任务", flush=True)
    cur_ep, cur_fi, tgt_fi, cur_ms, pair_task = [], [], [], [], []
    nseg = []; ndiscard_check = 0
    for tk, teps in sorted(task_eps.items()):
        if len(teps) < 5: continue
        res = r_and_segments({e: ep_gist[e] for e in teps})
        for e in teps:
            seg, ridge, n = res[e]; nseg.append(len(ridge))
            for p in range(n):
                si = np.searchsorted(seg, p, "right")-1
                if si+1 < len(ridge):
                    tgt = ridge[si+1]                        # 下一段 canonical 脊
                else:
                    tgt = n-1                                # 末段: 锚末帧(不丢!)
                cur_ep.append(e); cur_fi.append(p); tgt_fi.append(tgt); cur_ms.append(si); pair_task.append(tk)
    cur_ep = np.array(cur_ep); cur_fi = np.array(cur_fi); tgt_fi = np.array(tgt_fi); cur_ms = np.array(cur_ms); pair_task = np.array(pair_task)
    tot_frames = sum(len(v) for v in ep_gist.values())
    print(f"[seg] 每ep段数 中位={int(np.median(nseg))} 范围[{min(nseg)},{max(nseg)}]", flush=True)
    print(f"[pairs] {len(cur_ep)} 对 / {tot_frames} 帧 = 覆盖 {len(cur_ep)/tot_frames*100:.0f}% (milestone版会丢末段 → <100%)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/pairs.npz", cur_ep=cur_ep, cur_fi=cur_fi, tgt_fi=tgt_fi, cur_ms=cur_ms, pair_task=pair_task)
    print(f"[save] {OUT}/pairs.npz\nDONE", flush=True)

if __name__ == "__main__":
    main()
