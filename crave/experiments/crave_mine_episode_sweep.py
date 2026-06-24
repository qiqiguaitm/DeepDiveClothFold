"""挖矿 episode 数 对 CRAVE value 质量的影响(回答"多少 ep 才够/最优")。
固定 K=20 直接聚类、3Hz 缓存特征;变 mine-n,在固定 held-out 测试集上测 corr(value,时间)+ 单调性。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_mine_episode_sweep.py

Thin entrypoint over `crave`: kai0_base via resolve_dataset, arm/raw caches + state subsample
via crave.data.kai0, mkp/viterbi from crave.utils, viz_dir from crave.config.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from crave.config import REPO, resolve_dataset, viz_dir
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import mkp, viterbi

plt = setup_mpl()

_cfg = resolve_dataset("kai0_base")
ARM = Path(_cfg.arm_cache); RAW = Path(_cfg.raw_cache)
OUTV = viz_dir("centroid_decoder"); OUTJ = REPO / "temp/crave_a1a2"
NB = 21; BINS = np.linspace(0, 1, NB)


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], kai0.state_subsampled(_cfg, e, n), n


def dpHB(emit, lam=8.0):
    return BINS[viterbi(emit, BINS, lam=lam, end_bonus=2)[1]]


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
