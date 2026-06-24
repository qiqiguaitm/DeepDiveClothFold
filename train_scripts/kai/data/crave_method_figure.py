"""CRAVE 方法图(科研风, 真实数据多面板):images → latent → cluster → attributes → select → pin progress。

单次 kai0_base 挖矿(kai-only), 五个阶段各配一张真实数据子图, 拼成一张 method overview figure:
  (a) Input        多条 demo 的真实相机帧(顶视, 不同进度)
  (b) Latent       frozen DINOv2 三路嵌入的 t-SNE 2D, 按归一化进度着色 → 流形随任务推进展开
  (c) Clustering   同一 t-SNE 按 KMeans 簇着色 + 选中 milestone 质心(星, 大小=覆盖率)
  (d) Attributes   每簇 (进度 tpos, 覆盖率 covE) 散点 + Otsu 阈值 → 选出 milestone
  (e) Vocabulary   选中 milestone 的代表帧按进度 Pord 钉在 0→1 轴上 = 任务词表

英文标注(科研图惯例 + 本机无中文字体)。数据: kai0_base, 相机 top_head。
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_method_figure.py [--mine-n 220] [--k 96] [--tsne-n 4000]
输出: docs/visualization/cross_episode_recurrence_value/crave_method_figure.png
"""
import argparse, json, os
import numpy as np, pandas as pd, av, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pathlib import Path

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_base"
ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"
RAW = REPO / "temp/tcc_kai0_raw/feat_cache"
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


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


def otsu(x):
    """1D Otsu 阈值(在 covE 上分两类: milestone vs 噪声簇)。"""
    xs = np.sort(x); best_t, best_var = xs[0], -1
    for t in np.unique(xs):
        a = x[x < t]; b = x[x >= t]
        if len(a) == 0 or len(b) == 0: continue
        wa, wb = len(a) / len(x), len(b) / len(x)
        v = wa * wb * (a.mean() - b.mean()) ** 2
        if v > best_var: best_var, best_t = v, t
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=220)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--tsne-n", type=int, default=4000)
    a = ap.parse_args()

    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    print(f"全集 {len(all_eps)} | 挖掘 {len(mined)}", flush=True)

    Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    A, R, S, T, E, FR = [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = loadep(e)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); FR.append(np.arange(n) * 10)
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); FR = np.concatenate(FR)
    G = emb(A, R, S); K = a.k
    print(f"frames {len(G)}  → KMeans {K}", flush=True)
    km = KMeans(K, n_init=3, random_state=0).fit(G); lab = km.labels_; cen = km.cluster_centers_
    tpos = np.array([T[lab == c].mean() for c in range(K)])
    cov = np.array([len(set(E[lab == c].tolist())) / len(mined) for c in range(K)])
    # first-occurrence median = Pord
    def gr(idx):
        o = []; s0 = None; pv = None
        for i in idx:
            if pv is None or i != pv + 1:
                if s0 is not None: o.append((s0, pv))
                s0 = i
            pv = i
        if s0 is not None: o.append((s0, pv))
        return [x for x in o if x[1] - x[0] >= 1]
    Pk = np.zeros(K)
    for c in range(K):
        fe = []
        for e in mined:
            m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
            if rs: fe.append(T[rs[0][0]])
        Pk[c] = np.median(fe) if fe else tpos[c]

    tau = otsu(cov); selected = np.where(cov >= tau)[0]
    selected = selected[np.argsort(Pk[selected])]
    print(f"Otsu τ(cov)={tau:.3f} → 选出 {len(selected)} milestones", flush=True)

    # ---- t-SNE on subsample ----
    rng = np.random.RandomState(1); sub = rng.choice(len(G), min(a.tsne_n, len(G)), replace=False)
    print("t-SNE ...", flush=True)
    XY = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(G[sub])
    subT = T[sub]; subLab = lab[sub]
    cen_xy = {c: XY[subLab == c].mean(0) for c in range(K) if (subLab == c).any()}

    # ---- representative frame per selected milestone (closest to center) ----
    def rep_frame(c):
        idx = np.where(lab == c)[0]; dd = np.linalg.norm(G[idx] - cen[c], axis=1); j = idx[dd.argmin()]
        return grab(int(E[j]), int(FR[j]))

    OUTV.mkdir(parents=True, exist_ok=True)
    msk = np.zeros(K, bool); msk[selected] = True

    # pick a bright input episode (avoid dark/transition frames)
    fracs = [0.06, 0.30, 0.55, 0.78, 0.95]
    best_imgs, best_score = None, -1
    for ea in mined[::max(1, len(mined) // 12)][:12]:
        _, _, _, na = loadep(ea)
        imgs = [grab(ea, int(min(na - 1, fr * na) * 10)) for fr in fracs]
        if any(im is None for im in imgs): continue
        score = min(im.mean() for im in imgs)
        if score > best_score: best_score, best_imgs = score, imgs

    # =============== FIG 1: input frames → DINOv2 latent → clustering ===============
    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1, 1], wspace=0.16, left=0.02, right=0.98, top=0.84, bottom=0.08)
    sub_a = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 0], hspace=0.12, wspace=0.06)
    for k_, (frac, img) in enumerate(zip(fracs, (best_imgs or [None] * 5))):
        ax = fig.add_subplot(sub_a[k_ // 3, k_ % 3]); ax.axis("off")
        if img is not None: ax.imshow(img)
        ax.set_title(f"progress {frac:.2f}", fontsize=8.5, pad=1)
    fig.add_subplot(sub_a[1, 2]).axis("off")
    fig.text(0.13, 0.88, "(a) Input: real demo frames (top-head)", fontsize=12.5, fontweight="bold", ha="center")

    axb = fig.add_subplot(gs[0, 1])
    scb = axb.scatter(XY[:, 0], XY[:, 1], c=subT, cmap="viridis", s=8, alpha=0.75, linewidths=0)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("(b) Frozen DINOv2 latent (t-SNE), colored by progress", fontsize=12.5, fontweight="bold")
    cb = fig.colorbar(scb, ax=axb, fraction=0.046, pad=0.02); cb.set_label("task progress (0→1)", fontsize=9)

    axc = fig.add_subplot(gs[0, 2])
    axc.scatter(XY[:, 0], XY[:, 1], c=subLab % 20, cmap="tab20", s=8, alpha=0.55, linewidths=0)
    for c in selected:
        if c in cen_xy:
            axc.scatter(*cen_xy[c], s=40 + 360 * cov[c], facecolors="none", edgecolors="#c0392b", linewidths=1.8, zorder=4)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title(f"(c) KMeans → {K} recurrent states; red ○ = milestone (size∝coverage)", fontsize=12.5, fontweight="bold")
    fig.suptitle("Step 1–3 · frozen DINOv2 features → latent manifold → recurrent-state clusters", fontsize=14.5, fontweight="bold", y=0.98)
    for x0 in [0.345, 0.66]:
        fig.text(x0, 0.46, "→", fontsize=30, ha="center", va="center", color="#999")
    fig.savefig(OUTV / "crave_method_fig1_latent.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_method_fig1_latent.png", flush=True)

    # =============== FIG 2: attribute-based selection (Otsu) ===============
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.22))
    ax[0].scatter(tpos[~msk], cov[~msk], s=42, c="#b0b7c0", label="rejected cluster", zorder=2)
    ax[0].scatter(tpos[msk], cov[msk], s=70, c="#c0392b", edgecolor="k", linewidths=.5, label="selected milestone", zorder=3)
    ax[0].axhline(tau, color="#1f77b4", ls="--", lw=1.8, label=f"Otsu threshold τ={tau:.2f}")
    ax[0].set_xlabel("progress position  tpos", fontsize=11); ax[0].set_ylabel("coverage  covE", fontsize=11)
    ax[0].set_title("(d) Each cluster = (progress, coverage)", fontsize=12.5, fontweight="bold")
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=9.5, loc="lower center")
    order_cov = np.argsort(-cov); colors = ["#c0392b" if msk[c] else "#b0b7c0" for c in order_cov]
    ax[1].bar(range(K), cov[order_cov], color=colors)
    ax[1].axhline(tau, color="#1f77b4", ls="--", lw=1.8)
    ax[1].set_xlabel("clusters ranked by coverage", fontsize=11); ax[1].set_ylabel("coverage  covE", fontsize=11)
    ax[1].set_title(f"(e) Otsu auto-threshold → keep {len(selected)} milestones", fontsize=12.5, fontweight="bold")
    ax[1].grid(alpha=.2, axis="y")
    fig.suptitle("Step 4 · select milestones by coverage (Otsu auto-threshold, zero hand-tuning)", fontsize=14.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUTV / "crave_method_fig2_select.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_method_fig2_select.png", flush=True)

    # =============== FIG 3: milestone vocabulary pinned to progress (big frames, staggered) ===============
    fig = plt.figure(figsize=(18, 4.6)); axe = fig.add_axes([0.015, 0.12, 0.97, 0.74])
    axe.set_xlim(-0.01, 1.01); axe.set_ylim(0, 1); axe.set_yticks([])
    axe.set_xlabel("milestone progress  Pord  (0 → 1)", fontsize=12)
    axe.spines[["top", "left", "right"]].set_visible(False)
    NSHOW = min(15, len(selected))
    pick = selected[np.linspace(0, len(selected) - 1, NSHOW).round().astype(int)]
    slots = np.linspace(0.035, 0.965, NSHOW); w = 0.07
    for i, (slot, c) in enumerate(zip(slots, pick)):
        img = rep_frame(c)
        if img is None: continue
        x = float(Pk[c]); ytop = 0.52 if i % 2 == 0 else 0.16   # 交错两行 → 缩略图更大不重叠
        ax_in = axe.inset_axes([slot - w / 2, ytop, w, 0.40], transform=axe.transData)
        ax_in.imshow(img); ax_in.axis("off"); ax_in.set_title(f"P={x:.2f}", fontsize=8, pad=1)
        axe.plot([slot, x], [ytop, 0.06], color="#c0392b", lw=0.9, alpha=0.85)
        axe.plot(x, 0.04, "o", color="#c0392b", ms=5)
    axe.axhline(0.04, color="#888", lw=1)
    fig.suptitle("Step 5 · selected milestones pinned to progress Pord = zero-trained task vocabulary (flat cloth → folded)",
                 fontsize=14.5, fontweight="bold", y=0.99)
    fig.savefig(OUTV / "crave_method_fig3_vocab.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_method_fig3_vocab.png", flush=True)


if __name__ == "__main__":
    main()
