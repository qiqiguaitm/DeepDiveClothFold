"""挖矿 episode 数 对 CRAVE value 质量的影响(回答"多少 ep 才够/最优")。
固定 K=20 直接聚类、3Hz 缓存特征;变 mine-n,在固定 held-out 测试集上测 corr(value,时间)+ 单调性。
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_mine_episode_sweep.py
"""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_base"; ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"; RAW = REPO / "temp/tcc_kai0_raw/feat_cache"
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"; OUTJ = REPO / "temp/crave_a1a2"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]; NB = 21; BINS = np.linspace(0, 1, NB)


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def dpHB(emit, lam=8.0):
    pen = lam * np.abs(BINS[:, None] - BINS[None]); NF = len(emit); cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return BINS[path]


def main():
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    test = sorted(perm[-100:].tolist()); pool = perm[:-100]      # 固定 100 held-out
    print(f"全集 {len(all_eps)} | 可挖 {len(pool)} | 固定测试 {len(test)}", flush=True)

    # 测试集特征预载
    PMU = PSD = None
    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    rows = []
    for mn in [50, 100, 200, 300, len(pool)]:
        mined = sorted(pool[:mn].tolist())
        Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
        A, R, S, T = [], [], [], []
        for e in mined:
            aa, rr, st, n = loadep(e); A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1))
        A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T)
        G = emb(A, R, S)
        km = KMeans(20, n_init=3, random_state=0).fit(G); lab = km.labels_; cen = km.cluster_centers_
        tpos = np.array([T[lab == c].mean() for c in range(20)]); order = np.argsort(tpos); C = cen[order]; Pord = tpos[order]
        cb = [int(np.argmin(abs(BINS - p))) for p in Pord]
        cors, mons = [], []
        for e in test:
            aa, rr, st, n = loadep(e); Fq = emb(aa, rr, st)
            d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((n, NB), 1e3)
            for ci in range(20): em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d[:, ci])
            v = dpHB(em); tn = np.arange(n) / max(1, n - 1)
            if v.std() > 1e-6: cors.append(float(np.corrcoef(v, tn)[0, 1]))
            mons.append(float(np.mean(np.diff(v) >= -1e-6)))
        rows.append((mn, round(float(np.mean(cors)), 3), round(float(np.median(cors)), 3), round(float(np.mean(mons)), 3)))
        print(f"  mine-n={mn:4d}: corr_mean={rows[-1][1]} median={rows[-1][2]} mono={rows[-1][3]}", flush=True)

    json.dump({"test_n": len(test), "K": 20, "freq": "3Hz(cached)", "rows": [{"mine_n": r[0], "corr_mean": r[1], "corr_median": r[2], "mono": r[3]} for r in rows]},
              open(OUTJ / "mine_episode_sweep.json", "w"), indent=2, ensure_ascii=False)
    x = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r[1] for r in rows], "o-", label="corr mean")
    ax.plot(x, [r[2] for r in rows], "s--", label="corr median")
    ax.plot(x, [r[3] for r in rows], "^:", label="monotonicity")
    ax.set_xlabel("# mining episodes"); ax.set_ylabel("test value quality"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
    ax.set_title("CRAVE value quality vs # mining episodes (K=20, 3Hz, fixed 100-ep test)")
    fig.tight_layout(); fig.savefig(OUTV / "crave_mine_episode_sweep.png", dpi=120); plt.close(fig)
    print("SAVED crave_mine_episode_sweep.png", flush=True)


if __name__ == "__main__":
    main()
