#!/usr/bin/env python
"""robotwin 版 rvalley milestone 训练对 —— p1_libero_rvalley_pairs.py 的 robotwin 移植(同配方)。
recurrence 场分段: 每 ep 按 r-低谷分段; 目标=下一段 r-脊(canonical 收敛点); 末段锚末帧(不丢)。
与 LIBERO 侧逐字同配方(THR/平滑/find_peaks/脊选取), 仅改:
  ① 路径 → robotwin2.0 / robotwin_dinov3base
  ② 特征 key: LIBERO 用 ["grid"].mean(1); robotwin 特征已池化 → 直接 ["pooled"] ([N,768])
  ③ ep→task 映射: robotwin 27500ep 分 28 chunk, 不能 concat 全表 → 逐 ep parquet glob 读 task_index[0]
     (与 robotwin_revalidate.py 同法); 仅处理已抽特征的 5000 ep。
注意: robotwin task_index = 语言增强指令级(非任务类别), 故 ≥5ep 组只覆盖一部分 ep(见打印)。
特征帧 1:1 对齐 parquet frame_index(抽取时逐帧, 无 stride) → cur_fi/tgt_fi 即真实帧号。
输出 lmwm/data/robotwin_milestone/pairs.npz, keys 与 LIBERO pairs 同 → provider 可直接用。
用法: CUDA_VISIBLE_DEVICES= srpo python p1_robotwin_rvalley_pairs.py
"""
import os, glob
import numpy as np, pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
FEAT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"
OUT = f"{REPO}/lmvla/lmwm/data/robotwin_milestone"
THR = 0.03
MIN_EP = 5  # 与 LIBERO rvalley 同阈值(len(teps) < 5: continue)


def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def r_and_segments(gd):
    """返回 每 ep: r[n], 段边界 seg(含0,n), 段脊 ridge(全局帧内局部 idx)。(逐字同 LIBERO)"""
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    lens = [len(gd[e]) for e in eps]; offs = np.cumsum([0] + lens)
    D = cdist(F, F); dmin = np.full((len(F), ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, ep == j].min(1)
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    r = (np.exp(-dmin**2 / (2 * sig * sig)) * other).sum(1) / (ne - 1)
    res = {}
    for i, e in enumerate(eps):
        s, en = offs[i], offs[i + 1]; n = en - s; rr = r[s:en]
        v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n // 12))
        seg = [0] + list(v) + [n]; ridge = [a + int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]
        res[e] = (seg, ridge, n)
    return res


def main():
    files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    cached = [int(os.path.basename(f)[2:-4]) for f in files]
    # ep→task: 逐 ep parquet glob(robotwin 分 chunk, 不 concat 全表)
    ep2task = {}
    for e in cached:
        fs = glob.glob(f"{ROOT}/data/chunk-*/episode_{e:06d}.parquet")
        if fs:
            ep2task[e] = int(pq.read_table(fs[0], columns=["task_index"]).column(0)[0].as_py())
    ep_gist = {}
    for f in files:
        e = int(os.path.basename(f)[2:-4]); ep_gist[e] = np.load(f)["pooled"].astype(np.float32)
    task_eps = defaultdict(list)
    for e in ep_gist:
        task_eps[ep2task.get(e, -1)].append(e)
    n_used_tasks = sum(1 for v in task_eps.values() if len(v) >= MIN_EP)
    print(f"[tasks] {len(task_eps)} distinct task_index; {n_used_tasks} 有>={MIN_EP}ep(将建对)", flush=True)

    cur_ep, cur_fi, tgt_fi, cur_ms, pair_task = [], [], [], [], []
    nseg = []; used_eps = 0
    for tk, teps in sorted(task_eps.items()):
        if len(teps) < MIN_EP:
            continue
        res = r_and_segments({e: ep_gist[e] for e in teps})
        for e in teps:
            seg, ridge, n = res[e]; nseg.append(len(ridge)); used_eps += 1
            for p in range(n):
                si = np.searchsorted(seg, p, "right") - 1
                if si + 1 < len(ridge):
                    tgt = ridge[si + 1]   # 下一段 canonical 脊
                else:
                    tgt = n - 1           # 末段: 锚末帧(不丢)
                cur_ep.append(e); cur_fi.append(p); tgt_fi.append(tgt); cur_ms.append(si); pair_task.append(tk)
    cur_ep = np.array(cur_ep); cur_fi = np.array(cur_fi); tgt_fi = np.array(tgt_fi)
    cur_ms = np.array(cur_ms); pair_task = np.array(pair_task)
    tot_frames = sum(len(v) for v in ep_gist.values())
    print(f"[seg] 每ep段数 中位={int(np.median(nseg))} 范围[{min(nseg)},{max(nseg)}]", flush=True)
    print(f"[cover] 建对用 {used_eps}/{len(ep_gist)} eps ({used_eps/len(ep_gist)*100:.0f}%), "
          f"{len(set(pair_task.tolist()))} 任务", flush=True)
    print(f"[pairs] {len(cur_ep)} 对 / {tot_frames} 帧(全 5000ep)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/pairs.npz", cur_ep=cur_ep, cur_fi=cur_fi, tgt_fi=tgt_fi, cur_ms=cur_ms, pair_task=pair_task)
    print(f"[save] {OUT}/pairs.npz\nDONE", flush=True)


if __name__ == "__main__":
    main()
