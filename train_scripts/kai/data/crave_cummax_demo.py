"""演示: cummax 单调(防崩/不谎报) + 完成残差flag, 对比 no-anchor。3Hz cache, 不重渲全帧。"""
import sys
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
import numpy as np
import crave_align_analyze as A
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = Path("/vePFS/tim/workspace/deepdive_kai0/temp/crave_align")


def L2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


DS_EPS = {"coffee": [0, 1], "xvla": [7, 39], "vis": [2, 121]}
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
for row, ds in enumerate(["coffee", "xvla", "vis"]):
    c = np.load(OUT / f"{ds}_cache.npz")
    img = L2(c["img"].astype(np.float32)); st = c["state"].astype(np.float32)
    stn = L2((st - st.mean(0)) / (st.std(0) + 1e-8))
    F = np.concatenate([img, stn], 1); E = c["ep"]; T = c["tpos"]; eps_s = sorted(set(E.tolist()))
    cl = A.build_clusters(F, E, T, len(eps_s))
    EPe = np.concatenate([F[np.where(E == e)[0][np.argsort(T[np.where(E == e)[0]])][-2:]] for e in eps_s])
    ek = KMeans(8, n_init=2, random_state=0).fit(EPe).cluster_centers_
    de_tr = np.array([float(np.linalg.norm(F[np.where(E == e)[0][np.argmax(T[np.where(E == e)[0]])]][None] - ek, axis=1).min()) for e in eps_s])
    de_thr = float(np.quantile(de_tr, 0.9)) * 1.3
    for col, e in enumerate(DS_EPS[ds]):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(T[fi])]; Fq = F[fi]; tt = T[fi]
        v, ms = A.readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0)   # 原始 value(保留震荡, 不cummax)
        de_end = float(np.linalg.norm(Fq[-3:][:, None] - ek[None], axis=2).min())
        comp = de_end <= de_thr
        ax = axes[row, col]
        ax.plot(np.linspace(0, 1, len(v)), color="0.7", ls="--", lw=1, label="norm time")
        ax.plot(v, color="#1a7f37", lw=2, label=f"value (corr {np.corrcoef(v,tt)[0,1]:.2f})")
        ax2 = ax.twinx(); ax2.step(range(len(ms)), ms, where="post", color="#9c27b0", lw=1, alpha=.55); ax2.set_ylabel("milestone", color="#9c27b0", fontsize=7); ax2.tick_params(labelsize=6)
        ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
        flag = "COMPLETE" if comp else "INCOMPLETE(半完成)"
        ax.set_title(f"[{ds}] ep{e}  M={cl['M']}  末value={v[-1]:.2f}  | flag: {flag}(resid {de_end:.2f}/thr {de_thr:.2f})", fontsize=9, color=("#1a7f37" if comp else "#d62728"))
fig.suptitle("保留震荡(循环milestone重访) + 完成残差flag  (不 cummax, 不 end-anchor)", fontsize=13)
fig.tight_layout(); out = OUT / "osc_flag_demo.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); print("SAVED", out)
