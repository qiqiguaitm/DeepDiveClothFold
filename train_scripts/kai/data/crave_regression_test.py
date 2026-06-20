"""步骤3: 回退测试。造合成轨迹(ep2302 前进到中段再倒放回早段)= 真实操作回退/失误。
看读出 value 是否正确"先升后降"跟踪回退。验证: value 与真实进度相关 + 倒放段 value 确实下降。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_regression_test.py
"""
import sys, glob, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, torch, pandas as pd
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_decoder_scale_ablation import REPO, DS, cs
from crave_readout import smooth_monotone
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"; OUTD = REPO / "temp/crave_full"; ENC = "dino"; EP = 2302


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def medf(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def viterbi(emit, bins, lam, eb=2.0):
    NB = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= eb; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def gpu_kmeans(F, K, iters=25, chunk=30000, seed=0):
    g = torch.device("cuda"); Fg = torch.from_numpy(np.ascontiguousarray(F, np.float32)).to(g); Nn = Fg.shape[0]
    torch.manual_seed(seed); C = Fg[torch.randperm(Nn, device=g)[:K]].clone(); lab = torch.zeros(Nn, dtype=torch.long, device=g)
    for it in range(iters):
        Cn = (C * C).sum(1)
        for s in range(0, Nn, chunk):
            Fc = Fg[s:s + chunk]; lab[s:s + chunk] = ((Fc * Fc).sum(1, keepdim=True) - 2 * Fc @ C.T + Cn[None]).argmin(1)
        nc = torch.zeros_like(C); cnt = torch.zeros(K, device=g)
        nc.index_add_(0, lab, Fg); cnt.index_add_(0, lab, torch.ones(Nn, device=g)); msk = cnt > 0; C[msk] = nc[msk] / cnt[msk, None]
    return C.cpu().numpy(), lab.cpu().numpy()


def main():
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
        st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
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
        return medf(viterbi(em, bins, lam, eb), 5)            # 不 smooth_monotone, 看原始(允许下降)

    # ---- 合成回退轨迹: ep2302 前进到 f180 再倒放回 f60 ----
    fi = np.where(Ev == EP)[0]; oo = np.argsort(FRv[fi]); fi = fi[oo]; nep = len(fi)
    fwd = list(range(0, min(180, nep))); back = list(range(min(180, nep) - 1, 60, -1)); traj = fwd + back
    true_prog = Tv[fi][traj]                              # 真实进度: 先升后降
    # 正确合成: 图像按轨迹取(顺序无关), proprio 按轨迹 state 重算 mkp(Δ 自动正确反向)
    img_traj = img[fi][traj]
    stE = np.stack(pd.read_parquet(DS / "data" / f"chunk-{EP//cs:03d}" / f"episode_{EP:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
    st_traj = stE[np.minimum(FRv[fi][traj], len(stE) - 1)]
    Pt = mkp(st_traj); Pt = (Pt - PMU) / PSD; Pt /= (np.linalg.norm(Pt, axis=1, keepdims=True) + 1e-9)
    Fq = np.concatenate([img_traj, Pt], 1)               # 一致特征(倒放段 Δstate 已反向)
    # 多 lam 看回退灵敏度
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
