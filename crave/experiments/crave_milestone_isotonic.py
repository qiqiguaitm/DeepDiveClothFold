"""precedence 排序 + isotonic 度量 value(保信息)。对比 原 Pk(首达时间中位) vs isotonic 后:
  - 改了哪几个 milestone 的 value、改多少
  - advantage(相邻 milestone 间距)是否保住:原间距 vs isotonic 间距
isotonic = 把度量 value Pk 沿 precedence 序做保序回归(PAVA)→ 仅把逆序的几个并到单调水平, 其余精确保留。
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_milestone_isotonic.py
"""
from __future__ import annotations

import glob
import json
import time

import numpy as np

from crave.config import REPO
from crave.utils import otsu

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; ENC = "dino"


def main():
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; F = feat[vi].astype(np.float32); F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    Ev, Tv = E[vi], T[vi]
    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); lab = km.predict(F)
    ne = len(set(E.tolist()))
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    gap0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= gap0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); cl = np.array(sel)
    print(f"K0={K0} → {M} milestones; precedence + isotonic ...", flush=True)

    # 每 ep 首达时间 → Pk(度量 value, 携带间距信息) + precedence
    ep_list = sorted(set(Ev.tolist())); fe = np.full((len(ep_list), M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == cl[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])              # 原度量 value(首达中位)
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            both = ~np.isnan(fe[:, i]) & ~np.isnan(fe[:, j])
            if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.array([np.nansum(Pbef[i, :]) for i in range(M)])
    prec_order = list(np.argsort(-soft))                                  # precedence 序(早→晚)

    # isotonic: 把 Pk 沿 precedence 序做保序回归(PAVA), 单调且 L2 最近
    from sklearn.isotonic import IsotonicRegression
    y_raw = Pk[prec_order]                                                # 原 Pk 沿 precedence 序(含下降=逆序)
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), y_raw)
    iso_val = np.empty(M); iso_val[np.array(prec_order)] = iso            # 回到 milestone 索引

    # 对比统计
    changed = [m for m in range(M) if abs(iso_val[m] - Pk[m]) > 1e-4]
    g_raw = np.diff(y_raw); g_iso = np.diff(iso)                          # 相邻间距 = advantage 量纲
    neg = int((g_raw < -1e-9).sum())                                     # 原逆序(负间距)数
    preserved = int((np.abs(g_raw - g_iso) < 1e-4).sum())                # 完全保住的间距数
    summ = {"M": M, "value_changed_milestones": len(changed),
            "L1_value_change_total": round(float(np.abs(iso_val - Pk).sum()), 4),
            "max_single_value_change": round(float(np.abs(iso_val - Pk).max()), 4),
            "neg_gaps_raw(inversions)": neg, "gaps_preserved_exactly": f"{preserved}/{M-1}",
            "frac_gaps_preserved": round(preserved / (M - 1), 3)}
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump(summ, open(OUTD / "milestone_isotonic_summary.json", "w"), indent=2)

    # 图: 上=value 曲线(原 Pk vs isotonic, 沿 precedence 序), 下=相邻间距(advantage)对比
    from crave.render import setup_mpl
    plt = setup_mpl()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7))
    x = np.arange(M)
    a1.plot(x, y_raw, "o-", color="#999", ms=5, lw=1, label="raw Pk (median first-entry) — along precedence order")
    a1.plot(x, iso, "s-", color="#1a7f37", ms=4, lw=1.5, label="isotonic Pk (precedence-monotone, metric kept)")
    chg_pos = [k for k in range(M) if abs(iso[k] - y_raw[k]) > 1e-4]
    a1.scatter([x[k] for k in chg_pos], [y_raw[k] for k in chg_pos], s=90, facecolors="none", edgecolors="red", lw=2, label=f"changed ({len(chg_pos)})", zorder=5)
    for k in chg_pos: a1.annotate("", xy=(x[k], iso[k]), xytext=(x[k], y_raw[k]), arrowprops=dict(arrowstyle="->", color="red", lw=1))
    a1.set_xlabel("milestone (precedence order, early→late)"); a1.set_ylabel("value (progress)"); a1.grid(alpha=.3); a1.legend(fontsize=9)
    a1.set_title(f"VALUE: raw Pk has {neg} down-steps (inversions) → isotonic monotonizes; only {len(changed)} milestones changed, rest keep exact metric value")
    w = 0.4
    a2.bar(x[:-1] - w / 2, g_raw, w, color="#bbb", label="raw gap (advantage between adjacent milestones)")
    a2.bar(x[:-1] + w / 2, g_iso, w, color="#1a7f37", label="isotonic gap")
    a2.axhline(0, color="k", lw=0.5); a2.set_xlabel("adjacent milestone pair (precedence order)"); a2.set_ylabel("Δvalue (advantage spacing)")
    a2.grid(alpha=.3); a2.legend(fontsize=9)
    a2.set_title(f"ADVANTAGE spacing: {preserved}/{M-1} gaps preserved EXACTLY ({summ['frac_gaps_preserved']*100:.0f}%); only the {neg} negative(inverted) gaps pooled to ≥0 — no value-info loss elsewhere")
    fig.tight_layout(); out = OUTV / "crave_milestone_isotonic.png"; fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
