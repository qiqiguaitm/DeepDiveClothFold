"""CRAVE P1 (A1/A2) 低成本本地验证 —— 全 CPU,复用 smooth800_v24_full.py 的逐字挖掘。

一次挖掘(KMeans96 + coverage修正 + 进度分桶 + 端点锚,seed0,与 AB-plan mv_value 同模型),
然后产出 4 个可落地能力的实证 + 留痕:

  A1 milestone 分段        每条 demo 切成 milestone-progress 基元段;段表 + 段数分布 + 跨 ep 次序一致性(Kendall τ,不靠 DP)
  A2 keyframe              milestone 跨越帧 = 关键帧导出 + 计数分布
  A2 OOD/残差             (a) 域内:残差能否分出 corr.json 的坏 ep;(b) 跨任务:vision-only(raw⊕arm)残差 ROC-AUC 分 smooth800 vs xvla/coffee
  A2 dedup/质量            milestone 覆盖率 = 免费质量分(vs corr 散点);milestone 序列签名找近重复 ep

输出(非破坏): temp/crave_a1a2/{segments.json, summary.json, *.png}
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_a1a2_validate.py [--mine-n 700] [--max-ood 200]
"""
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
ARM = REPO / "temp/tcc_smooth800_dagger_armmask/feat_cache"
RAW = REPO / "temp/tcc_smooth800_dagger_raw/feat_cache"
GEN = REPO / "temp/generalization_value_eval"          # xvla/coffee combined npz (raw/armmask/state)
OUT = REPO / "temp/crave_a1a2"
FIG = REPO / "docs/visualization/cross_episode_recurrence_value"
CORR = REPO / "temp/mv_value_full/corr.json"            # 已有: 全集 corr(mv_value, time)
cs = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)


# ----------------------------- 数据加载 -----------------------------
def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def med(arr, w):
    h = w // 2; return np.array([np.median(arr[max(0, j - h):j + h + 1]) for j in range(len(arr))])


def kendall_tau(x, y):
    """无依赖 O(n^2) Kendall τ-b(够用, 每 ep 帧数 ~65)。"""
    n = len(x); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j]); sy = np.sign(y[i] - y[j])
            if sx * sy > 0: c += 1
            elif sx * sy < 0: d += 1
    return (c - d) / max(1, c + d)


# ----------------------------- 挖掘(与 full.py 逐字一致) -----------------------------
def mine(mine_n):
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:min(mine_n, len(all_eps))].tolist())
    print(f"全集 {len(all_eps)} ep; 挖掘子集 {len(mined)} ep", flush=True)
    Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    def embvis(a_, r_):                      # vision-only(raw⊕arm), 跨任务可比(不含 proprio)
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        return np.concatenate([rn, an], 1)

    A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = loadep(e); g = emb(aa, rr, st)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
        SP.append(g[:2]); EP.append(g[-2:])
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
    G = emb(A, R, S); Gvis = embvis(A, R)
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
    Pord = np.array([Pk[c] for c in order])
    Cvis = np.stack([Gvis[lab == c].mean(0) for c in order])    # vision-only milestone 中心
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
    print(f"V2.4 milestones: {len(order)} 前段(P<0.5): {int((Pord < 0.5).sum())}", flush=True)
    return dict(all_eps=all_eps, mined=mined, emb=emb, embvis=embvis, order=order, C=C, Cvis=Cvis,
                Pord=Pord, startK=startK, endK=endK)


def value_and_labels(M, aa, rr, st):
    """复刻 DiscreteValue.value: 返回 v(3Hz), 最近 milestone idx, 残差 d_min(3-path)。"""
    emb, C, startK, endK, Pord = M["emb"], M["C"], M["startK"], M["endK"], M["Pord"]
    NB = 21; bins = np.linspace(0, 1, NB)
    Fq = emb(aa, rr, st); nq = len(Fq)
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    cb = [int(np.argmin(abs(bins - p))) for p in Pord]
    em = np.full((nq, NB), 1e3)
    for ci in range(len(C)):
        em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1)
    de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    # Viterbi-DP (lam=8)
    pen = 8.0 * np.abs(bins[:, None] - bins[None]); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((nq, NB), int)
    for j in range(1, nq):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(nq, int); path[-1] = cost.argmin()
    for j in range(nq - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    v = med(bins[path], 9)
    return v, d.argmin(1), d.min(1)


def auc(pos, neg):
    """ROC-AUC via Mann-Whitney(pos 应有更高分)。"""
    pos = np.asarray(pos); neg = np.asarray(neg); n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0: return float("nan")
    allv = np.concatenate([pos, neg]); rank = pd.Series(allv).rank().to_numpy()
    return float((rank[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


# ----------------------------- 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=700)
    ap.add_argument("--max-ood", type=int, default=200, help="cross-task OOD 用的 smooth800 held-out ep 数")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    M = mine(a.mine_n)
    order, Pord = M["order"], M["Pord"]; nM = len(order)
    corr = json.load(open(CORR))["corr"] if CORR.exists() else {}

    # ---- 遍历全集: 分段 / keyframe / 覆盖 / 残差 / 一致性 ----
    segments = {}; per_ep = {}
    seg_counts = []; taus = []; mean_res = []; coverage = []; ep_ids = []; sigs = {}
    held_out = [e for e in M["all_eps"] if e not in set(M["mined"])]
    for k, e in enumerate(M["all_eps"]):
        aa, rr, st, n = loadep(e)
        v, lab_near, res = value_and_labels(M, aa, rr, st)
        # milestone 跨越帧(3Hz idx) = 第一次 v>=Pord[j]
        cross = [int(np.argmax(v >= Pord[j])) if (v >= Pord[j]).any() else -1 for j in range(nM)]
        reached = [j for j in range(nM) if cross[j] >= 0]
        cov = len(reached) / nM
        # 段: 相邻跨越帧之间(3Hz)
        kf = sorted(set(cf for cf in cross if cf >= 0))
        segs = []
        bounds = [0] + kf + [n]
        bs = sorted(set(bounds))
        for i in range(len(bs) - 1):
            s0, s1 = bs[i], bs[i + 1]
            if s1 > s0:
                seg_prog = float(v[min(s0, n - 1)])
                segs.append([s0, s1, round(seg_prog, 3)])
        segments[str(e)] = {"n_frames_3hz": int(n), "milestones_reached": reached,
                            "cross_frames_3hz": cross, "segments": segs, "coverage": round(cov, 3)}
        # 跨 ep 次序一致性: 最近-milestone-rank(原始 argmin, 不靠 DP) vs 时间
        if n >= 8:
            taus.append(kendall_tau(np.arange(n), lab_near))
        seg_counts.append(len(reached)); mean_res.append(float(res.mean())); coverage.append(cov); ep_ids.append(e)
        sigs[e] = tuple(reached)
        per_ep[e] = dict(cov=cov, res=float(res.mean()), nseg=len(reached))
        if (k + 1) % 300 == 0: print(f"  {k+1}/{len(M['all_eps'])} processed", flush=True)

    seg_counts = np.array(seg_counts); taus = np.array(taus); mean_res = np.array(mean_res); coverage = np.array(coverage)

    # ---- A2-OOD (a) 域内: 残差分坏 ep ----
    good = np.array([mean_res[i] for i, e in enumerate(ep_ids) if abs(corr.get(str(e), 1.0)) >= 0.5])
    bad = np.array([mean_res[i] for i, e in enumerate(ep_ids) if abs(corr.get(str(e), 1.0)) < 0.5])
    auc_bad = auc(bad, good)   # 坏 ep 残差应更高

    # ---- A2-OOD (b) 跨任务: vision-only 残差 ROC ----
    embvis, Cvis = M["embvis"], M["Cvis"]
    def vis_res(rawf, armf):
        q = embvis(armf, rawf); return np.linalg.norm(q[:, None] - Cvis[None], axis=2).min(1)
    rng = np.random.RandomState(0)
    ho = list(rng.permutation(held_out)[:a.max_ood])
    in_res = np.concatenate([vis_res(np.load(RAW / f"ep{e}.npz")["f"], np.load(ARM / f"ep{e}.npz")["f"]) for e in ho])
    ood_res = {}
    for name in ["xvla", "coffee"]:
        fc = GEN / name / "feat_cache"
        eps = sorted(int(p.stem[2:]) for p in fc.glob("ep*.npz"))
        rr = np.concatenate([vis_res(np.load(fc / f"ep{e}.npz")["raw"], np.load(fc / f"ep{e}.npz")["armmask"]) for e in eps])
        ood_res[name] = rr
    auc_xvla = auc(ood_res["xvla"], in_res); auc_coffee = auc(ood_res["coffee"], in_res)

    # ---- A2-dedup: 覆盖 vs corr; 近重复(milestone 序列签名相同) ----
    cov_corr = float(np.corrcoef(coverage, [abs(corr.get(str(e), 0.0)) for e in ep_ids])[0, 1])
    from collections import Counter
    sig_counts = Counter(sigs.values()); dup_groups = {str(k): v for k, v in sig_counts.items() if v >= 3}
    n_dup_eps = sum(v for v in sig_counts.values() if v >= 3)

    # ----------------------------- 留痕: summary.json -----------------------------
    summary = {
        "n_milestones": nM, "n_eps": len(ep_ids), "mine_n": len(M["mined"]),
        "A1_segmentation": {
            "seg_count_mean": float(seg_counts.mean()), "seg_count_median": float(np.median(seg_counts)),
            "seg_count_p10_p90": [float(np.percentile(seg_counts, 10)), float(np.percentile(seg_counts, 90))],
            "order_consistency_kendall_tau_mean": float(taus.mean()),
            "order_consistency_tau_frac_gt0.7": float((taus > 0.7).mean()),
        },
        "A2_keyframe": {"keyframes_per_ep_mean": float(seg_counts.mean()),
                        "note": "keyframe = milestone 跨越帧; 数=该 ep 到达的 milestone 数"},
        "A2_OOD": {
            "in_dist_bad_vs_good_residual_AUC": auc_bad,
            "good_res_mean": float(good.mean()), "bad_res_mean": float(bad.mean()) if len(bad) else None,
            "n_good": int(len(good)), "n_bad": int(len(bad)),
            "cross_task_vis_AUC_xvla": auc_xvla, "cross_task_vis_AUC_coffee": auc_coffee,
            "in_dist_vis_res_mean": float(in_res.mean()),
            "xvla_vis_res_mean": float(ood_res["xvla"].mean()), "coffee_vis_res_mean": float(ood_res["coffee"].mean()),
        },
        "A2_dedup": {"coverage_mean": float(coverage.mean()), "coverage_vs_corr": cov_corr,
                     "n_dup_eps_sig>=3": int(n_dup_eps), "n_dup_groups": len(dup_groups)},
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump(segments, open(OUT / "segments.json", "w"))
    print("\n==== SUMMARY ====", flush=True); print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    # ----------------------------- 留痕: 图 -----------------------------
    # persist OOD residual arrays for cheap re-plotting / audit
    np.savez(OUT / "ood_residuals.npz", in_res=in_res, xvla=ood_res["xvla"], coffee=ood_res["coffee"],
             good_res=good, bad_res=bad)

    # FIG1: example ep segmentation (ep808)
    ex = 808 if 808 in M["all_eps"] else M["all_eps"][len(M["all_eps"]) // 2]
    aa, rr, st, n = loadep(ex); v, lab_near, res = value_and_labels(M, aa, rr, st)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(v, lw=2, color="C0", label="CRAVE value (DP)")
    for j in range(nM):
        if (v >= Pord[j]).any():
            cf = int(np.argmax(v >= Pord[j])); ax[0].axvline(cf, color="C3", ls="--", alpha=.5)
    ax[0].set_ylabel("progress value"); ax[0].set_title(f"A1 milestone segmentation - ep{ex} (dashed = milestone crossing / keyframe)"); ax[0].legend(loc="lower right")
    ax[1].step(range(n), lab_near, where="post", color="C2"); ax[1].set_ylabel("nearest milestone idx"); ax[1].set_xlabel("frame (3Hz)")
    ax[1].set_title("raw nearest-milestone (no DP) rises with time -> ordered skill phases")
    plt.tight_layout(); plt.savefig(FIG / "crave_a1_segmentation_example.png", dpi=110); plt.close()

    # FIG2: segment-count + tau distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(seg_counts, bins=range(0, nM + 2), color="C0", alpha=.8); ax[0].axvline(seg_counts.mean(), color="k", ls="--")
    ax[0].set_title(f"milestones reached per ep (keyframes)\nmean={seg_counts.mean():.1f}/{nM}"); ax[0].set_xlabel("# milestones reached")
    ax[1].hist(taus, bins=20, color="C2", alpha=.8); ax[1].axvline(taus.mean(), color="k", ls="--")
    ax[1].set_title(f"cross-ep order consistency Kendall tau\nmean={taus.mean():.2f}, frac>0.7 = {100*(taus>0.7).mean():.0f}%"); ax[1].set_xlabel("tau(nearest-milestone, time)")
    plt.tight_layout(); plt.savefig(FIG / "crave_a1_consistency.png", dpi=110); plt.close()

    # FIG3: OOD - in-dist residual + cross-task residual
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(good, bins=30, alpha=.6, density=True, label=f"good ep (n={len(good)})", color="C0")
    if len(bad): ax[0].hist(bad, bins=30, alpha=.6, density=True, label=f"bad ep |corr|<0.5 (n={len(bad)})", color="C3")
    ax[0].set_title(f"A2-OOD in-dist: residual vs bad ep (AUC={auc_bad:.3f}, near chance)"); ax[0].set_xlabel("mean residual (3-path)"); ax[0].legend()
    ax[1].hist(in_res, bins=40, alpha=.6, density=True, label="smooth800 in-dist", color="C0")
    ax[1].hist(ood_res["xvla"], bins=40, alpha=.5, density=True, label="xvla OOD", color="C1")
    ax[1].hist(ood_res["coffee"], bins=40, alpha=.5, density=True, label="coffee OOD", color="C3")
    ax[1].set_title(f"A2-OOD cross-task (vision-only): AUC xvla={auc_xvla:.3f} coffee={auc_coffee:.3f}"); ax[1].set_xlabel("min residual to milestone (raw+arm)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "crave_a2_ood_residual.png", dpi=110); plt.close()

    # FIG4: dedup coverage vs corr
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cc = np.array([abs(corr.get(str(e), 0.0)) for e in ep_ids])
    ax.scatter(coverage, cc, s=8, alpha=.3, color="C0")
    ax.set_xlabel("milestone coverage (free quality score)"); ax.set_ylabel("|corr(mv_value, time)|")
    ax.set_title(f"A2-dedup/quality: coverage vs corr  r={cov_corr:.2f}")
    plt.tight_layout(); plt.savefig(FIG / "crave_a2_dedup_coverage.png", dpi=110); plt.close()

    print(f"\n图已存 {FIG}/crave_a1_*.png crave_a2_*.png; json 存 {OUT}", flush=True)


if __name__ == "__main__":
    main()
