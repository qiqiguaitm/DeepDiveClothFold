"""V2.4 milestone-value mining over the FULL A_smooth800_dagger_all (1117ep) → per-frame mv_value.
Refactor of smooth800_v24_ep0.py (single ep0) for the AWBC milestone-value A/B plan (P0b).

Mine the V2.4 milestone model ONCE on a random subset (KMeans(96) + coverage-corrected,
progress-bucketed milestones + endpoint anchors), then apply the Viterbi-DP value() to ALL
episodes → per-frame value v∈[0,1] (0→1 progress). Upsample 3Hz→30fps to parquet length.

Outputs (non-mutating; A1 builds ds_A from these):
  temp/mv_value_full/ep{N}.npy      per-ep per-frame mv_value (parquet-length)
  temp/mv_value_full/corr.json      per-ep corr(mv_value, normalized-time) + |corr|<0.5 bad list

Run: kai0/.venv/bin/python train_scripts/kai/data/smooth800_v24_full.py [--mine-n 700]
"""
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
ARM = REPO / "temp/tcc_smooth800_dagger_armmask/feat_cache"
RAW = REPO / "temp/tcc_smooth800_dagger_raw/feat_cache"
OUT = REPO / "temp/mv_value_full"
cs = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=700, help="random subset size for milestone mining")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:min(a.mine_n, len(all_eps))].tolist())
    print(f"全集 {len(all_eps)} ep; 挖掘子集 {len(mined)} ep", flush=True)

    # proprio normalization from mined subset
    Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = loadep(e); g = emb(aa, rr, st)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
        SP.append(g[:2]); EP.append(g[-2:])
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
    G = emb(A, R, S)
    km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
    N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
    Pstart = {}
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
    cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(96)])
    bk = np.linspace(0, 1, 11); sel = []
    for b in range(10):
        inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
        if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
    sel = sorted(set(sel), key=lambda c: tpos[c])

    def gr(idx):
        o = []; s = None; pv = None
        for i in idx:
            if pv is None or i != pv + 1:
                if s is not None: o.append((s, pv))
                s = i
            pv = i
        if s is not None: o.append((s, pv))
        return [x for x in o if x[1] - x[0] >= 1]

    Pk = {}
    for c in sel:
        fe = []
        for e in sorted(set(E.tolist())):
            m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
            if rs: fe.append(T[rs[0][0]])
        Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
    order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]
    print(f"V2.4 milestones: {len(order)} 前段(P<0.5): {sum(1 for c in order if Pk[c] < 0.5)}", flush=True)
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
    NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]

    def dpHB(emit, lam=8.0):
        pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit); cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
        for j in range(1, NF):
            tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
        cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
        for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
        return bins[path]

    def med(arr, w):
        h = w // 2; return np.array([np.median(arr[max(0, j - h):j + h + 1]) for j in range(len(arr))])

    def value(aa, rr, st):
        Fq = emb(aa, rr, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
        for ci in range(len(order)):
            for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        return med(dpHB(em), 9)

    # apply to ALL episodes → mv_value sidecar + corr
    corr = {}; bad = []
    for k, e in enumerate(all_eps):
        aa, rr, st, n = loadep(e); v = value(aa, rr, st)
        nf = len(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["frame_index"]))
        vf = np.repeat(v, 10)[:nf]
        if len(vf) < nf: vf = np.concatenate([vf, np.full(nf - len(vf), vf[-1] if len(vf) else 0.0)])
        np.save(OUT / f"ep{e}.npy", vf.astype(np.float32))
        tnorm = np.arange(nf) / max(1, nf - 1)
        cc = float(np.corrcoef(vf, tnorm)[0, 1]) if nf > 2 and vf.std() > 1e-6 else 0.0
        corr[e] = cc
        if abs(cc) < 0.5: bad.append(e)
        if (k + 1) % 200 == 0: print(f"  {k+1}/{len(all_eps)} applied", flush=True)
    json.dump({"corr": corr, "bad_lt0.5": bad, "n_bad": len(bad), "n_total": len(all_eps),
               "mine_n": len(mined), "milestones": len(order)},
              open(OUT / "corr.json", "w"), indent=1)
    print(f"DONE: mv_value for {len(all_eps)} ep → {OUT}; bad(|corr|<0.5)={len(bad)} ({100*len(bad)/len(all_eps):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
