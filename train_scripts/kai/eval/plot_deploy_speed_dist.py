#!/usr/bin/env python3
"""部署速度分布图 — 扫描所有 speed_ablation 标记的 episode, 画手臂/夹爪速度 KDE 分布
(线性 + log-x), 输出到 docs/deployment/inference/assets/deploy_speed/。

分布图消除了 episode 长度差异 (density 归一化), log-x 分开低速主峰与高速尾, 便于看
波峰波谷。均值不可比 (场景未控), 但分布形状 + 高速尾削峰是可归因信号。见
docs/deployment/inference/deploy_speed_analysis.md。

用法:
  /data1/miniconda3/bin/python train_scripts/kai/eval/plot_deploy_speed_dist.py
  # 自动 glob 所有 experiment.study==rtc_ema_speed_ablation 的 episode, 按 group 分 V0/V1 画。
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
FPS = 30
OUT = "/data1/tim/workspace/deepdive_kai0/docs/deployment/inference/assets/deploy_speed"
COLORS = {"off_off": "#888888", "rtc_only": "#d62728", "ema_only": "#1f77b4", "both": "#2ca02c"}


def _kind(e):
    r, m = e["rtc"], e["ema"]
    return "both" if (r and m) else "rtc_only" if r else "ema_only" if m else "off_off"


def collect():
    """→ {'v0': [(label, kind, states_path)], 'v1': [...]}"""
    out = {"v0": [], "v1": []}
    for mp in glob.glob("/data1/DATA_IMP/KAI0/Task_A/*/*/*/meta/episodes.jsonl"):
        root = mp.rsplit("/meta", 1)[0]
        for l in open(mp):
            d = json.loads(l)
            e = d.get("experiment")
            if not (e and e.get("study") == "rtc_ema_speed_ablation"):
                continue
            ep = d["episode_id"]
            pq = f"{root}/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
            if os.path.exists(pq):
                out[e["ckpt_variant"]].append((e["group"], _kind(e), pq))
    for k in out:
        out[k].sort()
    return out


def speeds(pq, dims, agg, thr):
    s = np.stack(pd.read_parquet(pq)["observation.state"].values).astype(float)
    d = np.abs(np.diff(s[:, dims], axis=0))
    v = (d.mean(1) if agg == "mean" else d.max(1)) * FPS
    return v[v > thr]


def plot(groups, dims, agg, thr, fname, title, lo, hi):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    lg = np.linspace(np.log10(lo), np.log10(hi), 400)
    for col, ver in enumerate(("v0", "v1")):
        for label, kind, pq in groups[ver]:
            v = speeds(pq, dims, agg, thr)
            if len(v) < 20:
                continue
            axes[col].plot(lg, gaussian_kde(np.log10(v), bw_method=0.12)(lg),
                           color=COLORS[kind], lw=2, label=f"{label} (n={len(v)})")
        axes[col].set_title(f"{title}  {ver.upper()} (log-x)")
        xt = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0) if lo <= t <= hi]
        axes[col].set_xticks([np.log10(t) for t in xt])
        axes[col].set_xticklabels([str(t) for t in xt])
        axes[col].set_xlabel("speed (log)")
        axes[col].legend()
        axes[col].grid(alpha=0.25)
    axes[0].set_ylabel("density")
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(f"{OUT}/{fname}", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/{fname}")


def main():
    g = collect()
    print("collected:", {k: len(v) for k, v in g.items()})
    plot(g, ARM, "mean", 0.03, "arm_speed_dist.png", "Arm speed dist", 0.03, 2.0)
    plot(g, GRIP, "max", 0.02, "gripper_speed_dist.png", "Gripper speed dist [EMA-excluded=pure RTC]", 0.02, 0.6)


if __name__ == "__main__":
    main()
