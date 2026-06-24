"""决定性验证: 生产 crave_value.py 三路特征(raw⊕armmask⊕proprio) 对 ep763 的 value 是否到 1.0。
假设: 我整套分析用纯图像(DINOv2-large)→ 起末视觉别名(折好≈摊平)→ ep763 折好态吸到低 milestone。
生产多本体感(臂状态)能分起/末 → 应到 1.0。
跑法: kai0/.venv/bin/python train_scripts/kai/data/crave_value_prod_test.py
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_value import FeatureSpace, DiscreteValue, loadep, mono
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_readout import smooth_monotone
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_base"
RAW = REPO / "temp/tcc_kai0_raw/feat_cache"; ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"
TRIPLE = REPO / "temp/_triple_prodtest"; TRIPLE.mkdir(exist_ok=True)
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"
TEST = [763, 2302]


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def build_triple(e):
    out = TRIPLE / f"ep{e}.npz"
    if out.exists(): return True
    try:
        a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    except Exception: return False
    n = min(len(a), len(r)); np.savez(out, armmask=a[:n], raw=r[:n], state=lpst(e, n)); return True


def main():
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mine = sorted(all_eps)                     # 用全部 550 缓存 ep 挖矿
    print(f"挖矿 {len(mine)} ep(三路缓存); 打包 triple ...", flush=True)
    mine = [e for e in mine if build_triple(e)]
    for e in TEST: build_triple(e)
    fs = FeatureSpace(TRIPLE, mine)
    dv = DiscreteValue(fs, mine, k=96, select="fixed")
    print(f"milestones={len(dv.order)}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(TEST), figsize=(6 * len(TEST), 4))
    for k, e in enumerate(TEST):
        a, r, s, n = loadep(TRIPLE, e); v = dv.value(a, r, s); vc = smooth_monotone(v, fps=3.0)
        tn = np.arange(n) / max(1, n - 1)
        print(f"ep{e}: n={n} 3-path value max={vc.max():.2f} last={vc[-3:].mean():.2f} mono={mono(vc):.2f}", flush=True)
        ax = axes[k] if len(TEST) > 1 else axes
        ax.plot(tn, vc, color="#1a7f37", lw=2, label="3-path (raw⊕armmask⊕proprio)")
        ax.set_title(f"ep{e} ({n}fr) PROD 3-path value  max={vc.max():.2f} last={vc[-3:].mean():.2f}", fontsize=10)
        ax.set_xlabel("progress (norm time)"); ax.set_ylabel("value"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Production crave_value.py (3-path) value — does ep763 reach 1.0? (vs image-only stuck at 0.15)", fontsize=11)
    fig.tight_layout(); out = OUTV / "crave_value_prod_test.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}", flush=True)


if __name__ == "__main__":
    main()
