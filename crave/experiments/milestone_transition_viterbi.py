"""数据驱动的 milestone 转移概率 × Viterbi —— 把"下一个簇是谁"的统计学概率注入 DP。

动机:现 Viterbi 的转移惩罚是纯几何 `lam·|Pord_i − Pord_j|`(对称、只看进度距离),
不知道"经验上 A 之后几乎总是 B"。本实验:
  1) 重走挖矿 ep,统计 milestone i→j 的经验转移概率 P(下一簇=j | 当前=i)(带 Laplace 平滑)。
  2) 把 `−log P(i→j)` 折进 milestone-空间 Viterbi 的转移代价(非对称:前进便宜、回退/乱跳贵、合法跳级也便宜)。
  3) 与几何版 readout_viterbi_ms、raw 对比 value 走势 + mono + 顺序违例 + 平滑度。

转移矩阵天然非对称 → 数据驱动的"近单调"(不是硬锁),且学到合法跳级 / 真实回退概率。
数据 kai0_base,复用 viterbi_compare.Model。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/milestone_transition_viterbi.py [--mine-n 200] [--beta 2.0] [--lam-geo 3.0]
输出: crave/docs/visualization/viterbi/milestone_transition_matrix.png, milestone_transition_value.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from crave.config import out_dir
from crave.data.cache import list_cache_eps
from crave.render import setup_mpl
from crave.utils import med, mono

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viterbi_compare import CFG, Model, OUTV, loadep  # noqa: E402

plt = setup_mpl()


def visited_sequence(am, w=5, min_run=2):
    """逐帧最近-milestone → 中值滤波 → 折叠连续同簇(run≥min_run)为访问序列。"""
    a = med(am.astype(float), w).round().astype(int)
    seq = []; i = 0; n = len(a)
    while i < n:
        j = i
        while j < n and a[j] == a[i]: j += 1
        if j - i >= min_run: seq.append(int(a[i]))
        i = j
    # 去掉相邻重复(中值后偶发抖动)
    out = [seq[0]] if seq else []
    for s in seq[1:]:
        if s != out[-1]: out.append(s)
    return out


def estimate_transition(M, mined, alpha=0.02):
    """经验转移矩阵 P[i,j]=P(next=j|cur=i),行归一 + Laplace。返回 (P, counts)。"""
    Mn = len(M.order); counts = np.zeros((Mn, Mn))
    for e in mined:
        a_, r_, s_, n = loadep(e)
        am = np.linalg.norm(M.emb(a_, r_, s_)[:, None] - M.C[None], axis=2).argmin(1)
        seq = visited_sequence(am)
        for i, j in zip(seq[:-1], seq[1:]):
            counts[i, j] += 1
    P = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * Mn)
    return P, counts


def build_pen(counts, Pord, lam_geo, beta, back_barrier, alpha=0.05):
    """非对称转移代价 pen[i_prev, j_cur]:
       前进(Pord_j≥Pord_i):lam_geo·ΔPord + beta·(−log Pf[i,j]) —— Pf 仅在前进目标上归一,
                            让"经验常见的下一簇/合法跳级"便宜、罕见前进贵(数据驱动排序+跳级)。
       后退(Pord_j<Pord_i):lam_geo·|ΔPord| + back_barrier —— 守近单调,但 back_barrier 有限,
                            真回退(§6.1)仍能压过 → 不硬锁。
    """
    Mn = len(Pord); eye = np.eye(Mn, dtype=bool)
    fwd_move = (Pord[None] > Pord[:, None] + 1e-9)                   # 严格前进的"换簇"(排除停留 i→i)
    Pf = np.where(fwd_move, counts, 0.0)
    Pf = (Pf + alpha * fwd_move) / (Pf.sum(1, keepdims=True) + alpha * fwd_move.sum(1, keepdims=True) + 1e-9)
    fcost = -np.log(Pf + 1e-12)
    # 行内居中:每个 i 的"最可能下一簇"换簇成本=0(=纯几何,不抬高似然路径),
    # 仅把"不太可能的前进"相对变贵 → 纯锐化排序/促成合法跳级,绝不诱发回退。
    fcost = fcost - np.where(fwd_move, fcost, np.inf).min(1, keepdims=True)
    geo = lam_geo * np.abs(Pord[:, None] - Pord[None])              # 停留 i→i = 0(自然 dwell 免费)
    back = (Pord[None] < Pord[:, None] - 1e-9)
    pen = geo + np.where(fwd_move, beta * fcost, 0.0) + np.where(back, back_barrier, 0.0)
    return pen


def viterbi_pen(emit, pen, start_anchor=True):
    """milestone-空间 Viterbi,给定任意转移代价矩阵 pen[i_prev, j_cur]。"""
    nn, Mn = emit.shape
    cost = np.full(Mn, 1e9); cost[0] = emit[0, 0]
    if not start_anchor: cost = emit[0].copy()
    bp = np.zeros((nn, Mn), int)
    for t in range(1, nn):
        tr = cost[:, None] + pen                          # tr[i_prev, j_cur]
        k = tr.argmin(0)
        cost = emit[t] + tr[k, np.arange(Mn)]
        bp[t] = k
    ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
    for t in range(nn - 2, -1, -1): ms[t] = bp[t + 1, ms[t + 1]]
    return ms


def emit_ms(M, a_, r_, s_):
    Fq = M.emb(a_, r_, s_); nn = len(Fq)
    emit = np.linalg.norm(Fq[:, None] - M.C[None], axis=2)
    dsx = np.linalg.norm(Fq[:, None] - M.startK[None], axis=2).min(1); tx = np.arange(nn) / nn
    emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
    return emit


def order_violations(ms, Pord):
    """路径里"进度倒退"步数占比(后一帧 milestone 的 Pord < 前一帧)。"""
    p = Pord[ms]; d = np.diff(p)
    return float(np.mean(d < -1e-9)) if len(d) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--lam-geo", type=float, default=3.0)   # 转移版里的几何稳定项权重
    ap.add_argument("--back-barrier", type=float, default=4.0)  # 后退附加壁垒(有限→真回退仍可见)
    ap.add_argument("--lam-ms", type=float, default=8.0)    # 几何基线 readout_viterbi_ms 的 lam
    a = ap.parse_args()
    rawset = set(list_cache_eps(CFG.raw_cache))
    all_eps = sorted(e for e in list_cache_eps(CFG.arm_cache) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined)
    Mn = len(M.order); Pord = M.Pord

    P, counts = estimate_transition(M, mined)
    # 经验"主后继":每个簇最可能的下一簇
    nxt = {i: int(np.argmax(counts[i])) for i in range(Mn) if counts[i].sum() > 0}
    n_skip = sum(1 for i in nxt if Pord[nxt[i]] - Pord[i] > 0.12)   # 主后继跳级(>1档)
    print(f"milestones={Mn};转移统计自 {len(mined)} 条挖矿 ep,共 {int(counts.sum())} 次簇间转移;"
          f"主后继含 {n_skip} 个跳级(>0.12 Pord)", flush=True)

    # ---------- FIG 1: 转移矩阵热力 + 主后继链 ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.05, 1]})
    im = ax[0].imshow(P, cmap="magma", vmin=0, vmax=min(1, P.max()))
    ax[0].set_xlabel("next milestone j (ordered by Pord)"); ax[0].set_ylabel("current milestone i")
    ax[0].set_title(f"Empirical transition P(next=j | cur=i)\n{Mn} milestones, {int(counts.sum())} transitions from {len(mined)} eps")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="P(i→j)")
    # 主后继链(数据驱动的 milestone 图):x=Pord_i, y=Pord_{主后继},点大小∝出现次数
    xs = [Pord[i] for i in range(Mn)]; ys = [Pord[nxt[i]] if i in nxt else Pord[i] for i in range(Mn)]
    szs = [20 + 4 * counts[i].sum() for i in range(Mn)]
    ax[1].plot([0, 1], [0, 1], "k:", lw=.8, alpha=.5, label="identity (stay)")
    for i in range(Mn):
        if i in nxt:
            ax[1].annotate("", xy=(Pord[i], Pord[nxt[i]]), xytext=(Pord[i], Pord[i]),
                           arrowprops=dict(arrowstyle="->", color="#1a9641", alpha=.5, lw=1.2))
    ax[1].scatter(xs, ys, s=szs, c="#c0392b", alpha=.6, edgecolor="none", zorder=3)
    ax[1].set_xlabel("Pord of current milestone"); ax[1].set_ylabel("Pord of most-likely next milestone")
    ax[1].set_title("Data-driven milestone successor graph\n(arrow: cur→argmax next; mostly forward, learns skips)")
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUTV / "milestone_transition_matrix.png", dpi=140); plt.close(fig)
    print("SAVED milestone_transition_matrix.png", flush=True)

    # ---------- 对比:几何 ms-Viterbi(基线)vs 转移感知(β 扫描)----------
    geo_pen = a.lam_ms * np.abs(Pord[:, None] - Pord[None])          # 对称几何基线 = readout_viterbi_ms
    CONFIGS = [("geometric", geo_pen)]
    for beta in [0.5, 1.0, 2.0]:
        CONFIGS.append((f"trans β={beta}", build_pen(counts, Pord, a.lam_geo, beta, a.back_barrier)))

    held_use = [e for e in held[:140] if loadep(e)[3] >= 20]
    agg = {name: {"mono": [], "viol": [], "rough": [], "skipsmooth": []} for name, _ in CONFIGS}
    examples = sorted(held_use, key=lambda e: abs(loadep(e)[3] - 130))[:4]
    rows = {name: [] for name, _ in CONFIGS}
    for e in held_use:
        a_, r_, s_, n = loadep(e)
        em = emit_ms(M, a_, r_, s_)
        for name, pen in CONFIGS:
            ms = viterbi_pen(em, pen); v = med(Pord[ms], 9)
            dms = np.abs(np.diff(Pord[ms]))
            agg[name]["mono"].append(mono(v)); agg[name]["viol"].append(order_violations(ms, Pord))
            agg[name]["rough"].append(float(np.abs(np.diff(v)).mean()))
            agg[name]["skipsmooth"].append(float(dms[dms > 0.12].sum()))   # 前进跳级总幅(越大=越多生硬跳)
            if e in examples: rows[name].append((e, n, v))

    aggm = {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in agg.items()}
    # 选最佳转移配置:mono≥几何−0.01 且 viol≤几何 的里 rough 最低
    g = aggm["geometric"]
    cand = [n for n in aggm if n != "geometric" and aggm[n]["mono"] >= g["mono"] - 0.01 and aggm[n]["viol"] <= g["viol"] + 0.01]
    best = min(cand, key=lambda n: aggm[n]["rough"]) if cand else "trans β=1.0"

    # ---------- FIG 2: value 走势对比(4 条 held ep,几何 vs 最佳转移)----------
    raw_ex = {e: Pord[emit_ms(M, *loadep(e)[:3]).argmin(1)] for e, _, _ in rows["geometric"]}
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.2)); ax = ax.ravel()
    for k in range(len(rows["geometric"])):
        e, n, vg = rows["geometric"][k]; _, _, vt = rows[best][k]; x = np.arange(n)
        ax[k].plot(x, raw_ex[e], color="#d7191c", lw=0.8, alpha=.4, label="raw nearest-milestone")
        ax[k].plot(x, vg, color="#2b8cbe", lw=2.0, ls="--", label=f"geometric (mono={mono(vg):.2f})")
        ax[k].plot(x, vt, color="#1a9641", lw=2.4, label=f"{best} (mono={mono(vt):.2f})")
        ax[k].set_title(f"ep{e}", fontsize=10); ax[k].set_ylim(-0.05, 1.08); ax[k].grid(alpha=.25)
        ax[k].set_xlabel("frame (3Hz)"); ax[k].set_ylabel("value"); ax[k].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Milestone-space value: geometric vs data-driven forward-−logP transition ({best}, λ_geo={a.lam_geo}, back={a.back_barrier})", fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(OUTV / "milestone_transition_value.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("SAVED milestone_transition_value.png", flush=True)

    print("\n==== 几何基线 vs 转移感知(held-out 聚合,N={}) ====".format(len(held_use)), flush=True)
    print(f"  {'config':<14} {'mono':>6} {'viol':>7} {'rough':>8} {'skipJump':>9}", flush=True)
    for name in aggm:
        m = aggm[name]; star = "  ← best" if name == best else ""
        print(f"  {name:<14} {m['mono']:>6.3f} {m['viol']:>7.3f} {m['rough']:>8.4f} {m['skipsmooth']:>9.3f}{star}", flush=True)
    print("  (viol=进度倒退步占比↓ rough=mean|Δv|↓ skipJump=生硬前进跳级总幅↓)", flush=True)
    json.dump({"lam_geo": a.lam_geo, "back_barrier": a.back_barrier, "n_milestones": Mn,
               "n_transitions": int(counts.sum()), "n_skip_successors": n_skip,
               "best": best, "agg": aggm},
              open(out_dir("crave_a1a2") / "milestone_transition.json", "w"), indent=2)


if __name__ == "__main__":
    main()
