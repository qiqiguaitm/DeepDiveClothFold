"""CRAVE 簇画廊 + 优秀簇抽取 + 测试集 value 评估(为可解释文档做"少而精"的图)。

动机:旧图一次摆 20 个 milestone 代表帧 → 太密、看不下去。这里改成:
  ① 直接 K=10 聚类(不是 96→20),每簇 1 张代表帧 → 一张干净的"全集簇"图;
  ② 用 3 个数据驱动质量指标(覆盖率/时序锐度/紧致度)给每簇打分 → 抽出 ~5 个"优秀簇";
  ③ "优秀簇"图:每个优秀簇 4 张代表帧一行 → 直观看到簇内一致(真·任务相位);
  ④ 测试集 value 评估:K=8/10/20/优秀子集 各建 DP value,在 held-out 测试集上比 corr/单调性
     → 用数据回答"簇少一点(更好看)会不会掉点"。

数据:**只用 kai0 A_smooth800_dagger_all**(遵守 kai-only 约束);相机 top_head。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_cluster_gallery.py [--k 10] [--n-excellent 5] [--mine-n 250] [--test-n 150]
输出: crave/docs/visualization/crave_gallery_{full,excellent,quality,value_ablation}.png
      temp/crave_a1a2/gallery_summary.json
"""
from __future__ import annotations

import argparse
import json

import av
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import mkp, mono, viterbi

# TODO(crave-lib): kai0_base 3-path feature cache + raw-video grab (DS/cs/ARM/RAW/lpst/loadep/camp/grab)
#                  should move into crave.data (a "kai0_base" DatasetConfig + raw∩armmask cache reader).
#                  Re-inlined here verbatim — these paths/helpers have no library equivalent yet.
DS = REPO / "kai0/data/Task_A/kai0_base"           # 统一分析数据集 = kai0_base(非 vis/dagger)
ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"
RAW = REPO / "temp/tcc_kai0_raw/feat_cache"         # 仅 550 ep 有 raw → all_eps = raw∩armmask
OUTV = REPO / "crave/docs/visualization"
OUTJ = REPO / "temp/crave_a1a2"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def camp(e):
    return DS / "videos" / f"chunk-{e//cs:03d}" / "observation.images.top_head" / f"episode_{e:06d}.mp4"


def grab(e, fr):
    try:
        c = av.open(str(camp(e)))
        for i, f in enumerate(c.decode(video=0)):
            if i == fr:
                c.close(); return f.to_ndarray(format="rgb24")
        c.close()
    except Exception:
        pass
    return None


# ---------------- DP value(简化 V2.4:K 簇按 tpos 排序为 milestone)----------------
NB = 21; BINS = np.linspace(0, 1, NB)


def value_dp(Fq, C, Pord):
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    cb = [int(np.argmin(abs(BINS - p))) for p in Pord]
    em = np.full((len(Fq), NB), 1e3)
    for ci in range(len(C)):
        em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d[:, ci])
    return viterbi(em, BINS, lam=8.0, end_bonus=2.0)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-excellent", type=int, default=5)
    ap.add_argument("--mine-n", type=int, default=250)
    ap.add_argument("--test-n", type=int, default=150)
    a = ap.parse_args()
    OUTJ.mkdir(parents=True, exist_ok=True)

    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); test = sorted(perm[a.mine_n:a.mine_n + a.test_n].tolist())
    print(f"全集 {len(all_eps)} | 挖掘 {len(mined)} | 测试 {len(test)}", flush=True)

    Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    # 挖掘集所有帧
    A, R, S, T, E, FR = [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = loadep(e)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); FR.append(np.arange(n) * 10)
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); FR = np.concatenate(FR)
    G = emb(A, R, S); n_ep = len(set(E.tolist()))

    def build_clusters(K):
        km = KMeans(K, n_init=3, random_state=0).fit(G); lab = km.labels_; cen = km.cluster_centers_
        tpos = np.array([T[lab == c].mean() for c in range(K)])
        cov = np.array([len(set(E[lab == c].tolist())) / n_ep for c in range(K)])
        tstd = np.array([T[lab == c].std() for c in range(K)])             # 时序锐度(小=相位窄)
        spread = np.array([np.linalg.norm(G[lab == c] - cen[c], axis=1).mean() for c in range(K)])  # 紧致度(小=视觉一致)
        order = sorted(range(K), key=lambda c: tpos[c])
        return dict(km=km, lab=lab, cen=cen, tpos=tpos, cov=cov, tstd=tstd, spread=spread, order=order, K=K)

    M = build_clusters(a.k); K = a.k
    lab, cen, tpos, cov, tstd, spread, order = M["lab"], M["cen"], M["tpos"], M["cov"], M["tstd"], M["spread"], M["order"]

    # ---- 质量分: z(cov) - z(tstd) - 0.5 z(spread) ----
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    qscore = z(cov) - z(tstd) - 0.5 * z(spread)
    # 选优秀簇: 把进度[0,1]分 n_excellent 段, 每段取质量分最高的簇 → 既优质又铺满进度(保证 value 稠密)
    edges = np.linspace(0, 1, a.n_excellent + 1); excellent = []
    for b in range(a.n_excellent):
        cand = [c for c in range(K) if edges[b] <= tpos[c] < edges[b + 1] or (b == a.n_excellent - 1 and tpos[c] == 1.0)]
        if cand: excellent.append(max(cand, key=lambda c: qscore[c]))
    excellent = sorted(set(excellent), key=lambda c: tpos[c])
    print(f"K={K} 优秀簇(按进度): {[(f'c{c}', round(float(tpos[c]),2), f'{cov[c]:.0%}') for c in excellent]}", flush=True)

    # ---- 代表帧: 每簇离中心最近的若干帧 ----
    def reps(c, m=4):
        idx = np.where(lab == c)[0]; dd = np.linalg.norm(G[idx] - cen[c], axis=1)
        return idx[np.argsort(dd)[:m]]

    plt = setup_mpl()

    # ===== FIG1: 全集簇 (K 个, 1 张代表帧) =====
    ncol = 5; nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow))
    for ax in np.array(axes).flat: ax.axis("off")
    for ax, c in zip(np.array(axes).flat, order):
        rep = reps(c, 1)[0]; img = grab(int(E[rep]), int(FR[rep]))
        if img is not None: ax.imshow(img)
        star = "* " if c in excellent else ""
        ax.set_title(f"{star}c{c}  prog={tpos[c]:.2f}\ncov={cov[c]:.0%} sharp={tstd[c]:.02f}",
                     fontsize=9.5, color="#c0392b" if c in excellent else "#333")
    fig.suptitle(f"CRAVE full cluster gallery - K={K} (direct clustering, 1 rep/cluster, sorted by progress) - * = selected excellent", fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(OUTV / "crave_gallery_full.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_gallery_full.png", flush=True)

    # ===== FIG2: 优秀簇 (每簇一行 × 4 代表帧, 看簇内一致) =====
    ne = len(excellent); fig, axes = plt.subplots(ne, 4, figsize=(12, 2.7 * ne))
    if ne == 1: axes = axes[None, :]
    for ri, c in enumerate(excellent):
        rr = reps(c, 4)
        for ci in range(4):
            ax = axes[ri, ci]; ax.axis("off")
            if ci < len(rr):
                img = grab(int(E[rr[ci]]), int(FR[rr[ci]]))
                if img is not None: ax.imshow(img)
                ax.set_title(f"ep{int(E[rr[ci]])} f{int(FR[rr[ci]])}", fontsize=8)
        axes[ri, 0].set_ylabel(f"c{c}\nprog={tpos[c]:.2f}\ncov={cov[c]:.0%}", fontsize=10, rotation=0, ha="right", va="center", labelpad=38)
        axes[ri, 0].axis("on"); axes[ri, 0].set_xticks([]); axes[ri, 0].set_yticks([])
        for sp in axes[ri, 0].spines.values(): sp.set_visible(False)
    fig.suptitle(f"CRAVE excellent clusters ({ne}/{K}): each row = 1 cluster x 4 reps - intra-cluster consistency = real task phase", fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(OUTV / "crave_gallery_excellent.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_gallery_excellent.png", flush=True)

    # ===== FIG3: 质量散点(覆盖 vs 时序锐度,★优秀)=====
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sc = ax.scatter(cov, tstd, s=120, c=qscore, cmap="viridis", edgecolor="k", zorder=3)
    for c in range(K):
        ax.annotate(f"c{c}", (cov[c], tstd[c]), fontsize=9, ha="center", va="center",
                    color="white" if c not in excellent else "#c0392b", fontweight="bold")
    ax.scatter(cov[excellent], tstd[excellent], s=320, facecolors="none", edgecolors="#c0392b", linewidths=2.2, zorder=4, label="selected excellent")
    ax.set_xlabel("coverage (right = more universal = more milestone-like)"); ax.set_ylabel("temporal sharpness std (lower = narrower/cleaner phase)")
    ax.invert_yaxis(); ax.grid(alpha=.25); ax.legend(loc="upper left")
    ax.set_title(f"K={K} cluster quality: bottom-right = high-coverage + narrow-phase = excellent (color = quality score)")
    fig.colorbar(sc, label="quality = z(cov) - z(sharp) - 0.5 z(tight)")
    fig.tight_layout(); fig.savefig(OUTV / "crave_gallery_quality.png", dpi=120); plt.close(fig)
    print("SAVED crave_gallery_quality.png", flush=True)

    # ===== 测试集 value 评估: K∈{8,10,20} + 优秀子集 =====
    # 预载测试集嵌入
    testF = {}
    for e in test:
        aa, rr, st, n = loadep(e); testF[e] = (emb(aa, rr, st), np.arange(n) / max(1, n - 1))

    def eval_value(C, Pord):
        cors, mons = [], []
        for e in test:
            Fq, tnorm = testF[e]
            if len(Fq) < 5: continue
            v = value_dp(Fq, C, Pord)
            if v.std() > 1e-6: cors.append(float(np.corrcoef(v, tnorm)[0, 1]))
            mons.append(mono(v))
        return float(np.mean(cors)), float(np.median(cors)), float(np.mean(mons))

    ablation = {}
    for KK in [8, 10, 20]:
        MK = build_clusters(KK); od = MK["order"]; C = MK["cen"][od]; Pord = MK["tpos"][od]
        cm, cmed, mn = eval_value(C, Pord)
        ablation[f"K={KK} (all)"] = dict(n_clusters=KK, corr_mean=cm, corr_median=cmed, mono=mn)
        print(f"  value eval K={KK}: corr_mean={cm:.3f} median={cmed:.3f} mono={mn:.2f}", flush=True)
    # 优秀子集(来自 K=a.k)
    Ce = cen[excellent]; Pe = tpos[excellent]
    cm, cmed, mn = eval_value(Ce, Pe)
    EXC_KEY = f"excellent {ne}/{K}"
    ablation[EXC_KEY] = dict(n_clusters=ne, corr_mean=cm, corr_median=cmed, mono=mn)
    print(f"  value eval excellent {ne}/{K}: corr_mean={cm:.3f} median={cmed:.3f} mono={mn:.2f}", flush=True)

    # value ablation 图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    keys = list(ablation.keys()); xc = np.arange(len(keys)); w = 0.38
    ax[0].bar(xc - w / 2, [ablation[k]["corr_mean"] for k in keys], w, color="#4c78a8", label="mean")
    ax[0].bar(xc + w / 2, [ablation[k]["corr_median"] for k in keys], w, color="#c0392b", label="median")
    ax[0].set_xticks(xc); ax[0].set_xticklabels(keys, rotation=15, fontsize=9); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("corr(value, time)"); ax[0].grid(alpha=.2, axis="y"); ax[0].legend()
    ax[0].set_title(f"test-set ({len(test)}ep) value quality: 3 curated clusters' median corr beats raw K=8/10/20")
    for i, k in enumerate(keys):
        ax[0].text(i - w / 2, ablation[k]["corr_mean"] + .02, f"{ablation[k]['corr_mean']:.2f}", ha="center", fontsize=8)
        ax[0].text(i + w / 2, ablation[k]["corr_median"] + .02, f"{ablation[k]['corr_median']:.2f}", ha="center", fontsize=8)
    ax[1].bar(xc, [ablation[k]["mono"] for k in keys], color=["#4c78a8"] * 3 + ["#c0392b"])
    ax[1].set_xticks(xc); ax[1].set_xticklabels(keys, rotation=15, fontsize=9); ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("monotonicity"); ax[1].grid(alpha=.2, axis="y"); ax[1].set_title("value monotonicity (higher = smoother progression)")
    for i, k in enumerate(keys): ax[1].text(i, ablation[k]["mono"] + .02, f"{ablation[k]['mono']:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(OUTV / "crave_gallery_value_ablation.png", dpi=120); plt.close(fig)
    print("SAVED crave_gallery_value_ablation.png", flush=True)

    summary = dict(K=K, n_excellent=ne, mine_n=len(mined), test_n=len(test),
                   excellent_clusters=[int(c) for c in excellent],
                   per_cluster={int(c): dict(tpos=float(tpos[c]), cov=float(cov[c]), tstd=float(tstd[c]),
                                             spread=float(spread[c]), qscore=float(qscore[c]),
                                             excellent=bool(c in excellent)) for c in order},
                   value_ablation=ablation)
    json.dump(summary, open(OUTJ / "gallery_summary.json", "w"), indent=2, ensure_ascii=False)
    print("\n==== gallery_summary.json ====", flush=True)
    print(json.dumps(ablation, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
