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


def method_anchor(sim, pord, t3, min_frames=2):
    """anchor-linear (FIXED 2026-07-05):每个**访问到的 milestone** 取一个锚(全 ep 最相似帧=峰值成员),
    value=Pord;按 Pord(进度序)排;对锚**时间**做单调投影(**不是对值** → 避免 isotonic 把非单调锚值
    pool 成等值平台);+ start(0,0)/end(末,1.0);distinct 值线性连接 → 连续折线,无平台。sim:(n,K)。

    旧版 bug:按 argmax 逐帧连续段建锚(噪声→几十微段,锚值乱跳)+ isotonic 对**值**投影 →
    把值 pool 成长平台(ep3033: 69 锚里 59 对等值)→ 87% 平台标签 → AE value 塌缩到 0。"""
    K = sim.shape[1]
    counts = np.bincount(sim.argmax(1), minlength=K)
    cand = [(int(np.argmax(sim[:, k])), float(pord[k])) for k in range(K) if counts[k] >= min_frames]
    if not cand:
        return np.clip(t3, 0, 1)
    cand.sort(key=lambda x: x[1])                          # 按 Pord 进度序(value 天然 distinct 递增)
    at = np.array([t3[j] for j, _ in cand], float)
    av = np.array([v for _, v in cand], float)
    at = isotonic(at)                                      # 单调投影 TIME(保持 value distinct,不产生平台)
    at = np.concatenate([[0.0], at, [1.0]]); av = np.concatenate([[0.0], av, [1.0]])
    keep = np.concatenate([[True], np.diff(at) > 1e-6]); at, av = at[keep], av[keep]
    return np.clip(np.interp(t3, at, av), 0, 1)


def method_viterbi(sim, pord, t3, lam=8.0, alpha=0.6):
    """Viterbi-DP (FIXED 2026-07-05):bins=milestone(按 Pord 升序), emit=(1-sim)+α|tn-Pord| 时间先验
    (破起末别名), 转移 λ|Δbin| → 读出 milestone 路径。在 milestone **转移点**设锚(time=t3,value=Pord[path]),
    start(0,0)/end(末,1.0),线性连接 → **连续折线**(替代旧版离散 `Pv[path]` 阶梯 → 90%+ 平台 → value 塌缩)。"""
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
    vp = Pv[path]
    ch = np.concatenate([[0], np.where(np.abs(np.diff(vp)) > 1e-9)[0] + 1])   # milestone 转移点=锚
    at = np.concatenate([[0.0], t3[ch], [1.0]]); av = np.concatenate([[0.0], vp[ch], [1.0]])
    av = isotonic(av)                                       # 兜单调(path 已近单调)
    keep = np.concatenate([[True], np.diff(at) > 1e-6]); at, av = at[keep], av[keep]
    return np.clip(np.interp(t3, at, av), 0, 1)


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
    vA = method_anchor(sim, pord, t3)
    vB = method_viterbi(sim, pord, t3)
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
