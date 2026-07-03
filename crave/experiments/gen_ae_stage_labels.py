#!/usr/bin/env python
"""为 KAI0-AE 蒸馏生成两套 stage_progress_gt(替代人工 Step-0 标注)。

方法 A(anchor-linear):段内离簇心最相似帧=锚点,value=Pord;+ start(0,0)/end(末,1.0);
                       isotonic 兜单调;线性连接 → 逐帧 value。
方法 B(viterbi):milestone bin 上 Viterbi-DP 读出(λ|Δbin| 转移惩罚)→ value=Pord[path],中值平滑。
两者均在 DINOv3-H milestones(milestones_uniform_dinov3h.npz)上,3Hz 生成 → 线性插值到 native 30Hz。

--sanity: 只跑少量 ep + 出对比图(A vs B vs 旧 AE stage_progress_gt/absolute_value)。
--full  : 全 3055 ep,存 native-fps 标签 → temp/crave_ae_labels/{anchor,viterbi}/ep*.npy。
Run: REPO=... PYTHONPATH=crave/src /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/gen_ae_stage_labels.py --sanity
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FEAT = REPO / "temp/crave_full_dinov3h"
BASE = REPO / "kai0/data/Task_A/kai0_base"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
OUTL = REPO / "temp/crave_ae_labels"
VIZ = REPO / "crave/docs/visualization/ae_distill"
CSB = 1000


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_feats():
    z = np.load(FEAT / "index.npz")
    E, FR, n = z["E"].astype(np.int64), z["FR"].astype(np.int64), int(z["n"])
    feat = np.zeros((n, 1280), np.float16); valid = np.zeros(n, bool)
    for f in sorted(glob.glob(str(FEAT / "shard_*.npz"))):
        s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
    return E, FR, l2(feat.astype(np.float32)), valid


def isotonic(v):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(increasing=True).fit_transform(np.arange(len(v)), v)


def method_anchor(sim, assign, pord, t3):
    """anchor-linear:段内峰值相似帧=锚,+start0/end1,isotonic,线性插值。sim:(n,K), t3:(n,)归一时间"""
    n = len(assign)
    # 连续段
    segs = []; s0 = 0
    for i in range(1, n + 1):
        if i == n or assign[i] != assign[s0]:
            segs.append((s0, i - 1, assign[s0])); s0 = i
    at, av = [0.0], [0.0]                                   # start 锚
    for (a, b, m) in segs:
        j = a + int(np.argmax(sim[a:b + 1, m]))            # 段内离簇心最相似帧
        at.append(t3[j]); av.append(float(pord[m]))
    at.append(1.0); av.append(1.0)                          # end 锚
    at, av = np.array(at), np.array(av)
    # 同一时间去重(保后者), 时间排序
    o = np.argsort(at, kind="stable"); at, av = at[o], av[o]
    keep = np.concatenate([[True], np.diff(at) > 1e-6]); at, av = at[keep], av[keep]
    av = isotonic(av)                                       # 兜单调
    return np.clip(np.interp(t3, at, av), 0, 1)


def method_viterbi(sim, pord, lam=8.0, medw=5, alpha=0.6):
    """Viterbi-DP:bins=milestone(按 Pord 升序), emit=(1-sim)+α|tn-Pord| 时间先验(破起末别名),
    转移 λ|Δbin|; value=Pord[path], 中值平滑。α 时间先验 = sym-adaptive-vote 的 distance-correction 思路。"""
    order = np.argsort(pord); Pv = pord[order]; K = len(order)
    n = len(sim); tn = np.linspace(0, 1, n)
    emit = (1.0 - sim[:, order]) + alpha * np.abs(tn[:, None] - Pv[None, :])   # +时间先验:t 处偏好 Pord≈tn 的态
    pen = lam * np.abs(np.arange(K)[:, None] - np.arange(K)[None]) / max(1, K - 1)
    cost = emit[0].copy(); bp = np.zeros((n, K), int)
    for t in range(1, n):
        tr = cost[None, :] + pen                            # to k from j
        bp[t] = tr.argmin(1); cost = emit[t] + tr[np.arange(K), bp[t]]
    path = np.zeros(n, int); path[-1] = cost.argmin()
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    v = Pv[path]
    if medw > 1:                                            # 中值平滑
        v = np.array([np.median(v[max(0, i - medw): i + medw + 1]) for i in range(n)])
    return np.clip(v, 0, 1)


def native_len(e):
    df = pd.read_parquet(BASE / f"data/chunk-{e // CSB:03d}/episode_{e:06d}.parquet", columns=["frame_index"])
    return len(df)


def gen_ep(e, feat, E, FR, C, pord):
    m = np.where(E == e)[0]
    if len(m) < 3:
        return None
    o = np.argsort(FR[m]); gi = m[o]; fr = FR[gi]
    f = feat[gi]; sim = f @ C.T; assign = sim.argmax(1)
    t3 = np.linspace(0, 1, len(gi))
    vA = method_anchor(sim, assign, pord, t3)
    vB = method_viterbi(sim, pord)
    nl = native_len(e)
    xi = np.arange(nl)
    xa = np.clip(fr, 0, nl - 1)                              # 3Hz 帧的 native 索引
    natA = np.clip(np.interp(xi, xa, vA), 0, 1)
    natB = np.clip(np.interp(xi, xa, vB), 0, 1)

    def norm01(v):                                           # 端点归一到 0→1(kai0_base 全成功完整折;单调曲线 min=首/max=末)
        lo, hi = float(v.min()), float(v.max())
        return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1) if hi - lo > 1e-3 else v
    natA, natB = norm01(natA), norm01(natB)
    return {"vA3": vA, "vB3": vB, "natA": natA, "natB": natB, "t3": t3, "nl": nl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    E, FR, feat, valid = load_feats()
    E = np.where(valid, E, -1)                               # 只用 valid
    mz = np.load(FEAT / "milestones_uniform_dinov3h.npz")
    C = l2(mz["C"].astype(np.float32)); pord = mz["Pord"].astype(np.float32)
    eps = sorted(set(E[E >= 0].tolist()))
    print(f"{len(eps)} episodes, {len(C)} milestones", flush=True)

    if a.sanity or not a.full:
        VIZ.mkdir(parents=True, exist_ok=True)
        samp = [2302, 23, 2238, 763, 1527, 2088]
        csq = json.load(open(Q5 / "meta/info.json"))["chunks_size"]
        fig, axs = plt.subplots(2, 3, figsize=(18, 8)); axs = axs.ravel()
        for k, e in enumerate(samp):
            r = gen_ep(e, feat, E, FR, C, pord)
            if r is None:
                continue
            try:
                dq = pd.read_parquet(Q5 / f"data/chunk-{e // csq:03d}/episode_{e:06d}.parquet",
                                     columns=["stage_progress_gt", "absolute_value"])
                sg = dq["stage_progress_gt"].to_numpy(); av = dq["absolute_value"].to_numpy()
            except Exception:
                sg = av = None
            x = np.linspace(0, 1, r["nl"])
            ax = axs[k]
            ax.plot(x, r["natA"], color="#1f77b4", lw=2, label="A: anchor-linear (CRAVE)")
            ax.plot(x, r["natB"], color="#2ca02c", lw=2, label="B: viterbi (CRAVE)")
            if sg is not None:
                ax.plot(np.linspace(0, 1, len(sg)), sg, color="#888", lw=1.5, ls="--", label="旧 AE stage_progress_gt(人工)")
                ax.plot(np.linspace(0, 1, len(av)), av, color="#d62728", lw=1, alpha=.6, label="旧 AE absolute_value")
                cA = pearsonr(r["natA"], np.interp(x, np.linspace(0, 1, len(sg)), sg))[0]
                cB = pearsonr(r["natB"], np.interp(x, np.linspace(0, 1, len(sg)), sg))[0]
                mA = float((np.diff(r["natA"]) >= -1e-6).mean()); mB = float((np.diff(r["natB"]) >= -1e-6).mean())
                ax.set_title(f"ep{e}  corr(A,人工)={cA:.2f} corr(B,人工)={cB:.2f} | mono A={mA:.2f} B={mB:.2f}", fontsize=9)
            else:
                ax.set_title(f"ep{e}", fontsize=10)
            ax.set_xlabel("归一时间"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.set_ylim(-.03, 1.03)
            if k == 0:
                ax.legend(fontsize=8, loc="lower right")
        fig.suptitle("KAI0-AE 蒸馏标签:两种 CRAVE stage_progress_gt vs 旧人工标注 · DINOv3-H · kai0_base",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(VIZ / "stage_label_compare.png", dpi=120); print("SAVED", VIZ / "stage_label_compare.png", flush=True)

    if a.full:
        for nm in ("anchor", "viterbi"):
            (OUTL / nm).mkdir(parents=True, exist_ok=True)
        done = 0
        for e in eps:
            r = gen_ep(e, feat, E, FR, C, pord)
            if r is None:
                continue
            np.save(OUTL / "anchor" / f"ep{e}.npy", r["natA"].astype(np.float32))   # A: anchor-linear
            np.save(OUTL / "viterbi" / f"ep{e}.npy", r["natB"].astype(np.float32))  # B: viterbi + 时间先验
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(eps)}", flush=True)
        print(f"FULL_DONE {done} eps -> {OUTL}/{{anchor,viterbi}}", flush=True)


if __name__ == "__main__":
    main()
