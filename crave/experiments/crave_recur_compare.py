"""对比循环 milestone 三种处理: baseline(对称DP) / 前向偏置DP / 多模态相对value(循环簇放回+时间模式+前向).
目标: 既解决重访震荡又保留任务跟踪. 在 3Hz cache 上, 三数据集长 ep 出对比图 + corr(value,时间)。

Thin entrypoint over `crave`: `L2`/`med`/`otsu`/`smooth_monotone`/`viterbi_forward` come
from `crave.utils`, `gpu_kmeans` from `crave.clustering`, REPO/out paths from
`crave.config`. The legacy inlined `viterbi_states` is exactly `viterbi_forward(..,
hard_start=True)` (symmetric when up==down) and is replaced by that import; `time_modes`
/ `recur_rate` are this script's unique logic and stay local.

跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_recur_compare.py
"""
import numpy as np
from sklearn.mixture import GaussianMixture

from crave.clustering import gpu_kmeans
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import L2, med, otsu, smooth_monotone, viterbi_forward

OUT = REPO / "temp/crave_align"
DS_EPS = {"coffee": [0, 1], "xvla": [7, 39], "vis": [2, 121]}


def time_modes(tm):
    if len(tm) < 30: return [float(tm.mean())]
    X = tm.reshape(-1, 1); best = None
    for k in (1, 2, 3):
        try:
            g = GaussianMixture(k, random_state=0, n_init=2).fit(X); b = g.bic(X)
            if best is None or b < best[1]: best = (g, b)
        except Exception: pass
    g = best[0]; ms = [float(m) for m, w in zip(g.means_.ravel(), g.weights_) if w > 0.18]
    return sorted(ms) if ms else [float(tm.mean())]


def recur_rate(lab, c, E, T):
    eps_c = sorted(set(E[lab == c].tolist()))
    if len(eps_c) < 10: return 0.0
    return float(np.mean([1 + int((np.diff(np.sort(T[(E == e) & (lab == c)])) > 0.05).sum()) >= 2 for e in eps_c]))


def smv(v, fps=3.0):
    mw = max(5, int(round(5 * fps / 3))) | 1
    return smooth_monotone(med(v, mw), fps=fps)


def main():
    plt = setup_mpl()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for row, ds in enumerate(["coffee", "xvla", "vis"]):
        c = np.load(OUT / f"{ds}_cache.npz")
        img = L2(c["img"].astype(np.float32)); st = c["state"].astype(np.float32)
        stn = L2((st - st.mean(0)) / (st.std(0) + 1e-8)); F = np.concatenate([img, stn], 1)
        E = c["ep"]; T = c["tpos"]; ne = len(set(E.tolist()))
        N = len(F); K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320)); cen, lab = gpu_kmeans(F, K0)
        tpos = np.array([T[lab == k].mean() if (lab == k).any() else 0 for k in range(K0)])
        cov = np.array([len(set(E[lab == k].tolist())) / ne if (lab == k).any() else 0 for k in range(K0)])
        tstd = np.array([T[lab == k].std() if (lab == k).sum() > 2 else 9.0 for k in range(K0)])
        rr = np.array([recur_rate(lab, k, E, T) for k in range(K0)])
        tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
        # baseline/forward: 纯度选中
        sel = sorted([k for k in range(K0) if cov[k] >= tau_cov and tstd[k] <= tau_pur], key=lambda k: tpos[k])
        Cs = cen[sel]; vs = tpos[sel]
        # multimode: cov通过(放回循环簇) + 时间模式展开
        mm = [k for k in range(K0) if cov[k] >= tau_cov and (tstd[k] <= tau_pur or rr[k] > 0.3)]
        Cmm, vmm = [], []
        for k in mm:
            for mv in time_modes(T[lab == k]): Cmm.append(cen[k]); vmm.append(mv)
        Cmm = np.array(Cmm); vmm = np.array(vmm)
        nrec = sum(rr[k] > 0.3 for k in mm)
        for col, e in enumerate(DS_EPS[ds]):
            fi = np.where(E == e)[0]; fi = fi[np.argsort(T[fi])]; Fq = F[fi]; tt = T[fi]
            emit_s = np.linalg.norm(Fq[:, None] - Cs[None], axis=2)
            emit_m = np.linalg.norm(Fq[:, None] - Cmm[None], axis=2)
            v_base = smv(vs[viterbi_forward(emit_s, vs, up=8.0, down=8.0, hard_start=True)])      # 对称
            v_fwd = smv(vs[viterbi_forward(emit_s, vs, up=3.0, down=25.0, hard_start=True)])      # 前向偏置
            v_mm = smv(vmm[viterbi_forward(emit_m, vmm, up=3.0, down=25.0, hard_start=True)])     # 多模态+前向
            ax = axes[row, col]
            ax.plot(np.linspace(0, 1, len(tt)), color="0.75", ls="--", lw=1, label="norm time")
            ax.plot(v_base, color="0.5", lw=1.6, label=f"baseline 对称 (corr {np.corrcoef(v_base,tt)[0,1]:.2f})")
            ax.plot(v_fwd, color="#e45756", lw=1.8, label=f"前向偏置 (corr {np.corrcoef(v_fwd,tt)[0,1]:.2f})")
            ax.plot(v_mm, color="#1a7f37", lw=2.2, label=f"多模态相对value (corr {np.corrcoef(v_mm,tt)[0,1]:.2f})")
            ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
            ax.set_title(f"[{ds}] ep{e}  sel={len(sel)}  mm态={len(vmm)}(循环簇{nrec})", fontsize=10)
    fig.suptitle("循环 milestone 处理对比: baseline对称 vs 前向偏置 vs 多模态相对value — 既解震荡又保跟踪", fontsize=13)
    fig.tight_layout(); out = OUT / "recur_compare.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); print("SAVED", out)


if __name__ == "__main__":
    main()
