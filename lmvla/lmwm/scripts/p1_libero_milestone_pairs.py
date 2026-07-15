#!/usr/bin/env python
"""P1: 在 LIBERO DINOv3-base grid 特征上跑 CRAVE milestone,构造 (frame_t, milestone+1) 训练对。
输出索引对(不复制 grid, 训练时按需从 ep{e}.npz 加载)。
用法: srpo python p1_libero_milestone_pairs.py [--maxep N] [--K auto]
输入: lmwm/data/libero_dinov3base/ep*.npz (grid [N,256,768] fp16)
输出: lmwm/data/libero_milestone/pairs.npz + milestone_meta.npz
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd

ROOT = "/home/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
OUT  = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_milestone"
CRAVE_SRC = "/home/tim/workspace/deepdive_kai0/lmvla/crave/src"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxep", type=int, default=0, help="0=all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, CRAVE_SRC)
    from crave.clustering import build_clusters

    files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    if args.maxep: files = files[:args.maxep]
    print(f"[load] {len(files)} episodes", flush=True)

    # 每 episode 的 task_index(LIBERO 40 任务, CRAVE 须 per-task 挖 milestone)
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    dd = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar])
    ep2task = dd.groupby("episode_index")["task_index"].first().to_dict()

    ep_gist, ep_ids = {}, []
    for f in files:
        e = int(os.path.basename(f)[2:-4])
        g = np.load(f)["grid"].astype(np.float32)
        ep_gist[e] = g.mean(1)                       # [N,768] pooled gist
        ep_ids.append(e)
    print(f"[feat] {len(ep_ids)} eps, {sum(len(v) for v in ep_gist.values())} frames", flush=True)

    # 按 task 分组
    from collections import defaultdict
    task_eps = defaultdict(list)
    for e in ep_ids: task_eps[ep2task.get(e, -1)].append(e)
    print(f"[tasks] {len(task_eps)} 个任务; 每任务 ep 数中位 {int(np.median([len(v) for v in task_eps.values()]))}", flush=True)

    cur_ep, cur_fi, tgt_fi, cur_ms, pair_task = [], [], [], [], []
    Ms = []
    for tk, teps in sorted(task_eps.items()):
        if len(teps) < 5: continue                   # 太少无法挖 recurrence
        F = np.concatenate([ep_gist[e] for e in teps])
        E = np.concatenate([np.full(len(ep_gist[e]), e) for e in teps])
        Tv = np.concatenate([np.linspace(0, 1, len(ep_gist[e])) for e in teps])
        cl = build_clusters(F, E, Tv, len(teps), seed=0)
        C = cl["C"]; M = cl["M"]; Ms.append(M)
        def assign(ge):
            raw = np.linalg.norm(ge[:, None] - C[None], axis=2).argmin(1)
            # 中值平滑去单帧尖峰(BUG_AUDIT MODERATE-4: 纯 cummax 棘轮, 单噪声帧永久锁死后续)
            n = len(raw); w = 5; sm = raw.copy()
            for i in range(n):
                sm[i] = int(np.median(raw[max(0, i - w): i + w + 1]))
            return np.maximum.accumulate(sm)   # 平滑后再单调
        for e in teps:
            ms = assign(ep_gist[e]); n = len(ms); first = {}
            for i, m in enumerate(ms):
                if m not in first: first[m] = i
            for i in range(n):
                m = int(ms[i])
                if m + 1 in first:
                    cur_ep.append(e); cur_fi.append(i); tgt_fi.append(first[m+1]); cur_ms.append(m); pair_task.append(tk)
    cur_ep = np.array(cur_ep); cur_fi = np.array(cur_fi); tgt_fi = np.array(tgt_fi); cur_ms = np.array(cur_ms)
    pair_task = np.array(pair_task)
    print(f"[milestone] per-task M 中位={int(np.median(Ms))} 范围[{min(Ms)},{max(Ms)}]", flush=True)
    print(f"[pairs] {len(cur_ep)} 训练对 (跨 {len(set(pair_task.tolist()))} 任务)", flush=True)

    if args.smoke:
        print(f"[smoke] 样例对: ep{cur_ep[0]} frame{cur_fi[0]}(ms{cur_ms[0]})→frame{tgt_fi[0]}(ms+1)", flush=True)
        print(f"[smoke] cur_ms 分布(前12): {np.bincount(cur_ms).tolist()[:12]}", flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/pairs.npz", cur_ep=cur_ep, cur_fi=cur_fi, tgt_fi=tgt_fi, cur_ms=cur_ms, pair_task=pair_task)
    print(f"[save] {OUT}/pairs.npz ({len(cur_ep)} pairs, per-task M 中位={int(np.median(Ms))})", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
