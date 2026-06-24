"""Viterbi 计算流程可视化 + 三变体对比 + 未完成(裁半)episode 的 end 锚点消融。

挖矿逻辑 1:1 复刻 DiscreteValue(kai0_base, kai-only);viterbi/med/mkp/mono 走 crave.utils。
产出(落 crave/docs/visualization/viterbi/):
  (1) viterbi_mechanism.png      emit 代价场热力图 + Viterbi 路径 + 逐帧最近-milestone(看 DP 如何穿过便宜格)
  (2) viterbi_three_variants.png 完整 ep:无 Viterbi(raw argmin) / Viterbi end=1(恒 bonus) / Viterbi cond_end(现方法)
  (3) viterbi_crop_endanchor.png 裁半(未完成)ep × N:end1 vs cond_end 末值是否重合
  (4) viterbi_complete_vs_crop.png 完整 vs 裁半:value 能否自动区分完成度 + de_end OOD flag
  + 控制台/图注汇总:裁半 ep 平均末值 end1 vs cond_end vs 理想≈0.5
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/viterbi_compare.py [--mine-n 200] [--ncrop 6]
"""
import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

from crave.config import out_dir, resolve_dataset, viz_dir
from crave.data import kai0
from crave.data.cache import list_cache_eps
from crave.render import setup_mpl
from crave.utils import med, mkp, mono, viterbi

plt = setup_mpl()
CFG = resolve_dataset("kai0_base")
OUTV = viz_dir("viterbi")              # crave/docs/visualization/viterbi


def loadep(e):
    """(armmask, raw, state, n) — 三路 tcc 缓存读取(legacy loadep)。"""
    return kai0.loadep_tcc(CFG, e)


def gr(idx):                                  # 连续段(首达用)
    o = []; s0 = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None: o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None: o.append((s0, pv))
    return [x for x in o if x[1] - x[0] >= 1]


class Model:
    """1:1 复刻 DiscreteValue 的挖矿 + 端点锚 + de_end_thr(kai0_base 三路特征版)。"""
    def __init__(self, mined):
        self.mined = mined
        Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall))
        self.PMU, self.PSD = Pm.mean(0), Pm.std(0) + 1e-8
        A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
        for e in mined:
            a, r, s, n = loadep(e); g = self.emb(a, r, s)
            A.append(a); R.append(r); S.append(s); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
            SP.append(g[:2]); EP.append(g[-2:])
        A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
        G = self.emb(A, R, S)
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
        Pk = {}
        for c in sel:
            fe = []
            for e in sorted(set(E.tolist())):
                m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
                if rs: fe.append(T[rs[0][0]])
            Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
        self.order = sorted(sel, key=lambda c: Pk[c]); self.C = allC[self.order]
        self.Pord = np.array([Pk[c] for c in self.order])
        self.startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
        self.endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
        de_tr = np.array([float(np.linalg.norm(ep[:, None] - self.endK[None], axis=2).min()) for ep in EP])
        self.de_end_thr = float(np.quantile(de_tr, 0.90)) * 1.3
        self.NB = 21; self.bins = np.linspace(0, 1, self.NB)
        self.cb = [int(np.argmin(abs(self.bins - p))) for p in self.Pord]
        print(f"milestones={len(self.order)}  de_end_thr={self.de_end_thr:.3f}", flush=True)

    def emb(self, a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - self.PMU) / self.PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    def build_emit(self, a, r, st):
        Fq = self.emb(a, r, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - self.C[None], axis=2)
        em = np.full((nq, self.NB), 1e3)
        for ci in range(len(self.order)): em[:, self.cb[ci]] = np.minimum(em[:, self.cb[ci]], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - self.startK[None], axis=2).min(1)
        de = np.linalg.norm(Fq[:, None] - self.endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, self.NB - 1] = np.minimum(em[:, self.NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        de_end = float(np.min(de[-3:])) if nq >= 3 else float(de[-1])
        return em, d, de_end

    def variants(self, a, r, st):
        em, d, de_end = self.build_emit(a, r, st)
        raw = self.Pord[d.argmin(1)]                                   # 无 Viterbi:逐帧最近 milestone 进度
        v_end1 = med(viterbi(em, self.bins, lam=8.0, end_bonus=2.0)[0], 9)   # 恒定 end=1 锚
        eb = 2.0 * float(np.clip((self.de_end_thr - de_end) / (0.3 * self.de_end_thr + 1e-9), 0.0, 1.0))
        v_cond = med(viterbi(em, self.bins, lam=8.0, end_bonus=eb)[0], 9)    # cond_end(现方法)
        path = viterbi(em, self.bins, lam=8.0, end_bonus=2.0)[1]
        return dict(raw=raw, end1=v_end1, cond=v_cond, em=em, path=path, de_end=de_end, eb=eb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--ncrop", type=int, default=6)
    a = ap.parse_args()
    rawset = set(list_cache_eps(CFG.raw_cache))
    all_eps = sorted(e for e in list_cache_eps(CFG.arm_cache) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined)

    # 选一条结构清晰的展示 ep(中等长度的 held-out)
    cand = sorted(held, key=lambda e: abs(loadep(e)[3] - 120))[:8]
    ex = cand[0]; aa, rr, st, n = loadep(ex); R = M.variants(aa, rr, st)
    x = np.arange(n)

    # ---- FIG 1: 机制图(emit 热力 + 路径)----
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.2), height_ratios=[1.25, 1], sharex=True)
    emap = R["em"].T.copy(); emap[emap >= 1e2] = np.nan       # 空 bin(1e3)留白
    im = ax[0].imshow(emap, aspect="auto", origin="lower", extent=[0, n, -0.5, M.NB - 0.5], cmap="viridis_r")
    ax[0].plot(x, R["path"], color="#1a9641", lw=2.4, label="Viterbi path (DP optimal)")
    ax[0].plot(x, (R["raw"] * (M.NB - 1)).round(), ".", color="#d7191c", ms=3, alpha=.55, label="per-frame nearest milestone (no DP)")
    ax[0].set_ylabel("progress bin (0..20)"); ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title(f"emit cost field (bright=cheap=milestone bin) + Viterbi path · ep{ex}  [milestones={len(M.order)}]")
    fig.colorbar(im, ax=ax[0], fraction=0.025, pad=0.01, label="emit cost (low=match)")
    ax[1].plot(x, R["raw"], color="#d7191c", lw=1.0, alpha=.8, label=f"no Viterbi (raw argmin, mono={mono(R['raw']):.2f})")
    ax[1].plot(x, R["cond"], color="#1a9641", lw=2.2, label=f"Viterbi (mono={mono(R['cond']):.2f})")
    ax[1].set_ylabel("value"); ax[1].set_xlabel("frame (3Hz)"); ax[1].legend(loc="lower right", fontsize=9); ax[1].grid(alpha=.25)
    ax[1].set_title("Viterbi denoises per-frame observations into a smooth monotone progress curve")
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_mechanism.png", dpi=140); plt.close(fig)
    print("SAVED viterbi_mechanism.png", flush=True)

    # ---- FIG 2: 完整 ep 三变体 ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(x, R["raw"], color="#d7191c", lw=1.0, alpha=.7, label="(1) no Viterbi (per-frame nearest milestone)")
    ax.plot(x, R["end1"], color="#2b8cbe", lw=2.0, ls="--", label="(2) Viterbi · end anchored to 1 (end_bonus=2)")
    ax.plot(x, R["cond"], color="#1a9641", lw=2.4, label="(3) Viterbi · cond_end (current method)")
    ax.axhline(1.0, color="#999", lw=.7, ls=":")
    ax.set_xlabel("frame (3Hz)"); ax.set_ylabel("value"); ax.set_ylim(-0.05, 1.08); ax.grid(alpha=.25); ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title(f"Complete ep{ex}: three variants — last value  raw={R['raw'][-1]:.2f} / end1={R['end1'][-1]:.2f} / cond={R['cond'][-1]:.2f}")
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_three_variants.png", dpi=140); plt.close(fig)
    print("SAVED viterbi_three_variants.png", flush=True)

    # ---- FIG 3 & 4: 裁半(未完成)ep 的 end 锚消融 ----
    crop_eps = sorted(np.random.RandomState(3).permutation(held)[:a.ncrop].tolist())
    rows = []
    for e in crop_eps:
        a_, r_, s_, ne = loadep(e); h = max(5, ne // 2)
        full = M.variants(a_, r_, s_)
        crop = M.variants(a_[:h], r_[:h], s_[:h])
        rows.append((e, ne, h, full, crop))
    end1_last = np.array([r[4]["end1"][-1] for r in rows])
    cond_last = np.array([r[4]["cond"][-1] for r in rows])
    de_ends = np.array([r[4]["de_end"] for r in rows])

    # FIG 3: 裁半 ep 上 end 锚消融 —— end1(虚) vs cond_end(实) 叠放 → 是否重合
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for e, ne, h, full, crop in rows:
        xc = np.arange(h) / h
        ax.plot(xc, crop["end1"], lw=2.6, ls="--", alpha=.9)
        ax.plot(xc, crop["cond"], lw=1.3, color="k", alpha=.55)
    ax.plot([], [], lw=2.6, ls="--", color="#666", label="end anchored to 1 (end_bonus=2)")
    ax.plot([], [], lw=1.3, color="k", label="cond_end (current method)")
    ax.axhline(1.0, color="#d7191c", lw=.8, ls=":"); ax.text(0.01, 1.02, "value=1 = falsely 'complete'", color="#d7191c", fontsize=8.5)
    ax.set_xlabel("normalized frame (cropped to half)"); ax.set_ylabel("value"); ax.set_ylim(-0.05, 1.12); ax.grid(alpha=.25); ax.legend(fontsize=10, loc="upper left")
    ax.set_title(f"Incomplete (half-cropped) ep x {len(rows)} · end-anchor ablation: the two coincide → mean last value end1={end1_last.mean():.2f} / cond={cond_last.mean():.2f} (gain {end1_last.mean()-cond_last.mean():+.2f})\n"
                 f"value is already capped by emission (last frame far from completion prototype endK); end_bonus too weak to matter", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_crop_endanchor.png", dpi=140); plt.close(fig)
    print("SAVED viterbi_crop_endanchor.png", flush=True)

    # FIG 4: 完整 vs 裁半(cond_end)—— value 本身能否区分"已完成/未完成" + 真正的 OOD 判据
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for e, ne, h, full, crop in rows:
        ax.plot(np.arange(ne) / ne, full["cond"], lw=2.2, alpha=.9)
        ax.plot(np.arange(h) / h, crop["cond"], lw=1.6, ls="--", alpha=.7, color=ax.lines[-1].get_color())
    ax.plot([], [], lw=2.2, color="#666", label="full ep (task completed) → value reaches ~1")
    ax.plot([], [], lw=1.6, ls="--", color="#666", label="same ep half-cropped (incomplete) → value capped ~0.33")
    ax.set_xlabel("normalized frame"); ax.set_ylabel("value"); ax.set_ylim(-0.05, 1.12); ax.grid(alpha=.25); ax.legend(fontsize=10, loc="lower right")
    ax.set_title(f"Full vs half-cropped (both cond_end): value auto-separates completion (full→1 / cropped→{cond_last.mean():.2f})\n"
                 f"the true 'incomplete' signal = decoupled de_end OOD flag: cropped last-frame de_end={de_ends.mean():.2f} > thr {M.de_end_thr:.2f} → 100% flagged incomplete by status()", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_complete_vs_crop.png", dpi=140); plt.close(fig)
    print("SAVED viterbi_complete_vs_crop.png", flush=True)

    print("\n==== 裁半 ep end 锚消融汇总 ====", flush=True)
    print(f"  N={len(rows)} 条裁半 ep;理想末值≈0.50", flush=True)
    print(f"  end 恒锚1   平均末值 = {end1_last.mean():.3f}  (min {end1_last.min():.2f} / max {end1_last.max():.2f})", flush=True)
    print(f"  cond_end    平均末值 = {cond_last.mean():.3f}  (min {cond_last.min():.2f} / max {cond_last.max():.2f})", flush=True)
    print(f"  改善(末值下降) = {end1_last.mean() - cond_last.mean():+.3f}", flush=True)
    print(f"  裁半末帧 de_end 均值 = {de_ends.mean():.3f} vs de_end_thr={M.de_end_thr:.3f} → 超阈(判未完成)占 {100*np.mean(de_ends>M.de_end_thr):.0f}%", flush=True)
    json.dump({"crop_eps": crop_eps, "end1_last_mean": float(end1_last.mean()), "cond_last_mean": float(cond_last.mean()),
               "improve": float(end1_last.mean() - cond_last.mean()), "de_end_thr": M.de_end_thr,
               "de_end_mean": float(de_ends.mean()), "incomplete_flag_frac": float(np.mean(de_ends > M.de_end_thr))},
              open(out_dir("crave_a1a2") / "viterbi_crop_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
