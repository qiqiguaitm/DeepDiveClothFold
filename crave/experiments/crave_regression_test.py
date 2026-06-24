"""步骤3: 回退测试。造合成轨迹(ep2302 前进到中段再倒放回早段)= 真实操作回退/失误。
看读出 value 是否正确"先升后降"跟踪回退。验证: value 与真实进度相关 + 倒放段 value 确实下降。
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_regression_test.py

Thin entrypoint over `crave`: otsu/med/viterbi/mkp/smooth_monotone from the package,
gpu_kmeans from crave.clustering, kai0_base dataset from crave.config; full-scale dino
shard cache (temp/crave_full, gidx/feat/valid layout) is read inline (a different cache
from crave.data.load_dino_shards — see TODO).
"""
import glob, time

import numpy as np, pandas as pd

from crave.clustering import gpu_kmeans
from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import med, mkp, otsu, smooth_monotone, viterbi

# TODO(crave-lib): the full-scale dino shard cache temp/crave_full uses an index_dino.npz
# (E/FR/T/n) + dino/shard_*.npz (gidx/feat/valid) layout, distinct from the "f"-key shards
# that crave.data.load_dino_shards reads. Re-inlined verbatim here.
DS_CFG = resolve_dataset("kai0_base")
DS = DS_CFG.root
cs = kai0.chunks_size(DS)
OUTV = REPO / "crave/docs/visualization/centroid_decoder"; OUTD = REPO / "temp/crave_full"; ENC = "dino"; EP = 2302


def main():
    from pathlib import Path
    DSp = Path(DS)
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; img = feat[vi].astype(np.float32); img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    Ev, FRv, Tv = E[vi], FR[vi], T[vi]; ep_list = sorted(set(Ev.tolist()))
    P = np.zeros((len(vi), 28), np.float32)
    for e in ep_list:
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        st = np.stack(pd.read_parquet(DSp / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
        P[loc] = mkp(st[np.minimum(FRv[loc], len(st) - 1)])
    PMU, PSD = P.mean(0), P.std(0) + 1e-8
    Pn = (P - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1); K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    print(f"GPU KMeans K0={K0} ...", flush=True); cen, lab = gpu_kmeans(F, K0); ne = len(ep_list)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel)
    fe = np.full((ne, M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == sel[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i != j:
                both = np.isfinite(fe[:, i]) & np.isfinite(fe[:, j])
                if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), 1); prec = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec]); Pord = np.asarray(iso, float)
    order = [sel[p] for p in prec]; C = cen[order]
    SP = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][:2]] for e in ep_list])
    EPp = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][-2:]] for e in ep_list])
    from sklearn.cluster import KMeans
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_; ek = KMeans(8, n_init=2, random_state=0).fit(EPp).cluster_centers_
    bins = np.linspace(0, 1, 41); cb = [[int(np.argmin(abs(bins - Pord[m])))] for m in range(M)]

    def readout(Fq, lam=8.0, eb=2.0, endanchor=True):
        nn = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nn, 41), 1e3)
        for m in range(M):
            for b in cb[m]: em[:, b] = np.minimum(em[:, b], d[:, m])
        dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        if endanchor:
            dex = np.linalg.norm(Fq[:, None] - ek[None], axis=2).min(1)
            em[:, 40] = np.minimum(em[:, 40], np.where(tx > 0.6, dex, dex + (0.6 - tx) * 6))
        return med(viterbi(em, bins, lam, eb)[0], 5)            # 不 smooth_monotone, 看原始(允许下降)

    # ---- 合成回退轨迹: ep2302 前进到 f180 再倒放回 f60 ----
    fi = np.where(Ev == EP)[0]; oo = np.argsort(FRv[fi]); fi = fi[oo]; nep = len(fi)
    fwd = list(range(0, min(180, nep))); back = list(range(min(180, nep) - 1, 60, -1)); traj = fwd + back
    true_prog = Tv[fi][traj]                              # 真实进度: 先升后降
    # 正确合成: 图像按轨迹取(顺序无关), proprio 按轨迹 state 重算 mkp(Δ 自动正确反向)
    img_traj = img[fi][traj]
    stE = np.stack(pd.read_parquet(DSp / "data" / f"chunk-{EP//cs:03d}" / f"episode_{EP:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
    st_traj = stE[np.minimum(FRv[fi][traj], len(stE) - 1)]
    Pt = mkp(st_traj); Pt = (Pt - PMU) / PSD; Pt /= (np.linalg.norm(Pt, axis=1, keepdims=True) + 1e-9)
    Fq = np.concatenate([img_traj, Pt], 1)               # 一致特征(倒放段 Δstate 已反向)
    # 多 lam 看回退灵敏度
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(13, 4.8)); x = np.arange(len(traj))
    ax.plot(x, true_prog, color="k", lw=1.5, ls="--", label="true progress (orig frame Tv) — 先升后降")
    configs = [("旧:lam=8 +endBonus +anchor", dict(lam=8, eb=2.0, endanchor=True), "#999"),
               ("修:lam=4 无endBonus 无anchor", dict(lam=4, eb=0.0, endanchor=False), "#1a7f37"),
               ("修:lam=2 无endBonus 无anchor", dict(lam=2, eb=0.0, endanchor=False), "#e45756")]
    for name, kw, col in configs:
        v = readout(Fq, **kw)
        peak = v[:len(fwd)].max(); endv = v[-3:].mean(); drop = peak - endv
        corr = float(np.corrcoef(v, true_prog)[0, 1])
        ax.plot(x, v, color=col, lw=2, label=f"{name} (corr={corr:.2f}, 回退降幅={drop:+.2f})")
        print(f"{name}: peak={peak:.2f} end={endv:.2f} 回退降幅={drop:+.2f} corr(value,真进度)={corr:.2f}", flush=True)
    ax.axvline(len(fwd) - 0.5, color="0.6", lw=1); ax.text(len(fwd), 0.02, "← 前进 | 倒放 →", fontsize=9)
    ax.set_ylim(-.02, 1.02); ax.set_xlabel("synthetic frame"); ax.set_ylabel("value"); ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_title("回退测试: 合成轨迹(前进→倒放). value 应跟随真实进度先升后降. 降幅>0 且 corr 高 = 能表示回退")
    fig.tight_layout(); out = OUTV / "crave_regression_test.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
