#!/usr/bin/env python
"""最终版 CRAVE value 鲁棒性测试:多随机 episode 渲染 value 曲线 + 全量聚合统计。

用已生成的最终标签(temp/crave_ae_labels/{anchor,viterbi},DINOv3-H 上的 CRAVE value)。
- 图1:N 个随机 ep 的 value 曲线(anchor + viterbi + 人工 stage_progress_gt 参照)。
- 图2:全 3055 ep 鲁棒性汇总:corr(CRAVE, 人工) 分布 + 最差 6 个 ep(暴露失败模式)。
Run: REPO=... PYTHONPATH=crave/src /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_value_robustness.py
输出: crave/docs/visualization/ae_distill/crave_value_robustness_{gallery,summary}.png
"""
from __future__ import annotations

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
LAB = REPO / "temp/crave_ae_labels"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
VIZ = REPO / "crave/docs/visualization/ae_distill"
CSQ = 1000
rng = np.random.RandomState(42)


def manual_gt(e):
    try:
        return pd.read_parquet(Q5 / f"data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet",
                               columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    except Exception:
        return None


def main():
    VIZ.mkdir(parents=True, exist_ok=True)
    eps = sorted(int(Path(p).stem[2:]) for p in glob.glob(str(LAB / "viterbi/ep*.npy")))
    print(f"{len(eps)} eps with CRAVE value", flush=True)

    # ---- aggregate robustness over ALL eps ----
    rows = []
    for e in eps:
        vv = np.load(LAB / "viterbi" / f"ep{e}.npy"); va = np.load(LAB / "anchor" / f"ep{e}.npy")
        gt = manual_gt(e)
        cor_v = cor_a = np.nan
        if gt is not None and gt.std() > 1e-3:
            gq = np.interp(np.linspace(0, 1, len(vv)), np.linspace(0, 1, len(gt)), gt)
            if vv.std() > 1e-3:
                cor_v = pearsonr(vv, gq)[0]
            if va.std() > 1e-3:
                cor_a = pearsonr(va, gq)[0]
        mono_v = float((np.diff(vv) >= -1e-6).mean()); mono_a = float((np.diff(va) >= -1e-6).mean())
        # 早跳指标:达到 0.9 的相对时间(过早=可能塌缩/别名)
        t90_v = float(np.argmax(vv >= 0.9) / len(vv)) if (vv >= 0.9).any() else 1.0
        rows.append((e, cor_v, cor_a, mono_v, mono_a, t90_v))
    R = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in rows], float)
    cv, ca, mv, ma, t90 = R.T
    def frac(x, th): return float(np.mean(x[~np.isnan(x)] >= th))
    stats = {"n_eps": len(eps),
             "corr_viterbi_vs_manual": {"mean": float(np.nanmean(cv)), "median": float(np.nanmedian(cv)),
                                        "frac>=0.8": frac(cv, 0.8), "frac>=0.7": frac(cv, 0.7)},
             "corr_anchor_vs_manual": {"mean": float(np.nanmean(ca)), "median": float(np.nanmedian(ca)),
                                       "frac>=0.8": frac(ca, 0.8), "frac>=0.7": frac(ca, 0.7)},
             "mono_viterbi_mean": float(np.nanmean(mv)), "mono_anchor_mean": float(np.nanmean(ma)),
             "t90_viterbi_mean": float(np.nanmean(t90))}
    (VIZ / "crave_value_robustness_summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    # ---- gallery: 24 random eps ----
    samp = rng.choice(eps, 24, replace=False)
    fig, axs = plt.subplots(4, 6, figsize=(22, 12)); axs = axs.ravel()
    for k, e in enumerate(samp):
        vv = np.load(LAB / "viterbi" / f"ep{e}.npy"); va = np.load(LAB / "anchor" / f"ep{e}.npy")
        gt = manual_gt(e)
        x = np.linspace(0, 1, len(vv))
        axs[k].plot(x, vv, color="#2ca02c", lw=1.8, label="viterbi")
        axs[k].plot(x, va, color="#1f77b4", lw=1.5, label="anchor")
        if gt is not None:
            axs[k].plot(np.linspace(0, 1, len(gt)), gt, color="#888", lw=1, ls="--", label="人工")
        cvk = next(r[1] for r in rows if r[0] == e)
        axs[k].set_title(f"ep{e}  corr(vit,人工)={cvk:.2f}", fontsize=9)
        axs[k].set_ylim(-.03, 1.03); axs[k].grid(alpha=.25); axs[k].tick_params(labelsize=7)
        if k == 0:
            axs[k].legend(fontsize=8)
    fig.suptitle(f"最终版 CRAVE value 鲁棒性 · 24 随机 ep · DINOv3-H · kai0_base "
                 f"(全量 corr(vit,人工) 均值 {np.nanmean(cv):.2f}, ≥0.7 占 {frac(cv,0.7):.0%})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(VIZ / "crave_value_robustness_gallery.png", dpi=110)
    print("SAVED", VIZ / "crave_value_robustness_gallery.png", flush=True)
    plt.close(fig)

    # ---- worst 6 by corr(viterbi,manual) (暴露失败模式) ----
    valid = [r for r in rows if not np.isnan(r[1])]
    worst = sorted(valid, key=lambda r: r[1])[:6]
    fig, axs = plt.subplots(1, 6, figsize=(22, 3.6)); axs = axs.ravel()
    for k, r in enumerate(worst):
        e = r[0]; vv = np.load(LAB / "viterbi" / f"ep{e}.npy"); va = np.load(LAB / "anchor" / f"ep{e}.npy"); gt = manual_gt(e)
        x = np.linspace(0, 1, len(vv))
        axs[k].plot(x, vv, color="#2ca02c", lw=1.8, label="viterbi")
        axs[k].plot(x, va, color="#1f77b4", lw=1.5, label="anchor")
        if gt is not None:
            axs[k].plot(np.linspace(0, 1, len(gt)), gt, color="#888", lw=1, ls="--")
        axs[k].set_title(f"ep{e} corr={r[1]:.2f}", fontsize=9); axs[k].set_ylim(-.03, 1.03); axs[k].grid(alpha=.25)
    fig.suptitle("最差 6 个 ep(按 corr(viterbi,人工))—— 暴露失败模式", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(VIZ / "crave_value_robustness_summary.png", dpi=115)
    print("SAVED", VIZ / "crave_value_robustness_summary.png", flush=True)


if __name__ == "__main__":
    main()
