"""决定性测试: 旧 small 特征(raw⊕armmask⊕state, 3Hz feat_cache) vs 我的 large 特征, 同一读出, 看 coffee ep0 是否单调。
隔离: 是特征(small 3路)还是配方/全帧 导致 ep0 别名崩。"""
import sys, glob
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from pathlib import Path
import crave_align_analyze as A
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = Path("/vePFS/tim/workspace/deepdive_kai0"); FC = REPO / "temp/generalization_value_eval/coffee/feat_cache"


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def L2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def build(eps, mode):
    F, E, T = [], [], []
    Pall = np.concatenate([mkp(np.load(FC / f"ep{e}.npz")["state"]) for e in eps])
    PMU, PSD = Pall.mean(0), Pall.std(0) + 1e-8
    for e in eps:
        d = np.load(FC / f"ep{e}.npz")
        n = min(len(d["raw"]), len(d["armmask"]), len(d["state"]))   # 三路对齐到公共长度
        rn = L2(d["raw"][:n].astype(np.float32)); an = L2(d["armmask"][:n].astype(np.float32))
        pn = L2((mkp(d["state"])[:n] - PMU) / PSD)
        if mode == "small3": f = np.concatenate([rn, an, pn], 1)
        elif mode == "small_raw_proprio": f = np.concatenate([rn, pn], 1)
        F.append(f); E.append(np.full(n, e)); T.append(np.arange(n) / max(1, n - 1))
    return np.concatenate(F), np.concatenate(E), np.concatenate(T)


eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for row, mode in enumerate(["small3", "small_raw_proprio"]):
    F, E, T = build(eps, mode); cl = A.build_clusters(F, E, T, len(eps))
    for col, e in enumerate([0, 1]):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(T[fi])]; Fq = F[fi]
        v, ms = A.readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0)
        corr = float(np.corrcoef(v, T[fi])[0, 1]) if v.std() > 0 else 0
        ax = axes[row, col]; ax.plot(v, color="#1a7f37", lw=2); ax.plot(np.linspace(0, 1, len(v)), color="0.6", ls="--", lw=1)
        ax.set_ylim(-.02, 1.02); ax.set_title(f"[{mode}] coffee ep{e}  M={cl['M']}  corr(v,t)={corr:.3f}", fontsize=10); ax.grid(alpha=.3)
fig.suptitle("旧 small 特征 + 我的读出: coffee ep0/ep1 是否单调(对比我 large 双路 ep0 崩)", fontsize=12)
out = REPO / "temp/crave_align/coffee_feat_test.png"
fig.tight_layout(); fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); print("SAVED", out)
