"""把 value 读出重构成一个干净的 milestone-HMM(承用户提议 / §8–9):
  - 去掉 startK/endK/end_bonus 三件套;start=最小 Pord milestone(强制起点),end=最大 Pord milestone(不强制)。
  - 发射 = 当前帧对各 milestone 的软归类概率 softmax(−d/τ)。
  - 转移 = 经验 milestone 转移概率 P(next|cur),并按帧级补自环 A = pstay·I + (1−pstay)·P_visit(dwell)。
  - 解码 = HMM Viterbi(max-product)→ value = Pord[path] → 中值平滑。
对比当前部署值(bin cond_end)与几何 ms-Viterbi;并复测 §6.1 回退可观测 / §6.2 复现鲁棒是否保住。

Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/milestone_hmm.py [--mine-n 200] [--tau 0.3] [--pstay auto]
输出: crave/docs/visualization/viterbi/milestone_hmm.png + temp/crave_a1a2/milestone_hmm.json
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
from milestone_transition_viterbi import estimate_transition, visited_sequence  # noqa: E402

plt = setup_mpl()


def mean_dwell(M, mined):
    """挖矿 ep 上 argmin-milestone 的平均停留帧长 → 估自环概率。"""
    runs = []
    for e in mined[:120]:
        a_, r_, s_, n = loadep(e)
        am = med(np.linalg.norm(M.emb(a_, r_, s_)[:, None] - M.C[None], axis=2).argmin(1).astype(float), 5).round().astype(int)
        i = 0
        while i < len(am):
            j = i
            while j < len(am) and am[j] == am[i]: j += 1
            runs.append(j - i); i = j
    return float(np.mean(runs))


def hmm_viterbi(logB, logA, logpi):
    NF, Mn = logB.shape
    delta = logpi + logB[0]; bp = np.zeros((NF, Mn), int)
    for t in range(1, NF):
        tr = delta[:, None] + logA            # tr[i_prev, j_cur]
        bp[t] = tr.argmax(0); delta = logB[t] + tr.max(0)
    path = np.zeros(NF, int); path[-1] = int(delta.argmax())
    for t in range(NF - 2, -1, -1): path[t] = bp[t + 1, path[t + 1]]
    return path


class HMM:
    def __init__(self, M, counts, tau, pstay, alpha=0.02):
        self.M = M; self.Pord = M.Pord; self.tau = tau; Mn = len(M.order)
        Pv = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * Mn)      # 经验转移(含 backward)
        A = pstay * np.eye(Mn) + (1 - pstay) * Pv                                # 帧级补自环 dwell
        A /= A.sum(1, keepdims=True)
        self.logA = np.log(A + 1e-12)
        self.logpi = np.full(Mn, -1e9); self.logpi[0] = 0.0                      # 强制起点=最小 Pord milestone

    def value(self, a, r, st):
        d = np.linalg.norm(self.M.emb(a, r, st)[:, None] - self.M.C[None], axis=2)   # (NF, M)
        logB = -d / self.tau                                                    # 软归类 log 发射
        path = hmm_viterbi(logB, self.logA, self.logpi)
        return med(self.Pord[path], 9), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--pstay", default="0.95")
    a = ap.parse_args()
    rawset = set(list_cache_eps(CFG.raw_cache))
    all_eps = sorted(e for e in list_cache_eps(CFG.arm_cache) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined); Pord = M.Pord
    P, counts = estimate_transition(M, mined)
    dwell = mean_dwell(M, mined)
    pstay = 1 - 1 / dwell if a.pstay == "auto" else float(a.pstay)
    hmm = HMM(M, counts, a.tau, pstay)
    geo_pen = 8.0 * np.abs(Pord[:, None] - Pord[None])
    print(f"milestones={len(M.order)}  mean_dwell={dwell:.1f}f  pstay={pstay:.3f}  tau={a.tau}", flush=True)

    def geo_ms(a_, r_, s_):
        em = np.linalg.norm(M.emb(a_, r_, s_)[:, None] - M.C[None], axis=2)
        nn = len(em); cost = np.full(len(M.order), 1e9); cost[0] = em[0, 0]; bp = np.zeros((nn, len(M.order)), int)
        for t in range(1, nn):
            tr = cost[:, None] + geo_pen; k = tr.argmin(0); cost = em[t] + tr[k, np.arange(len(M.order))]; bp[t] = k
        ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
        for t in range(nn - 2, -1, -1): ms[t] = bp[t + 1, ms[t + 1]]
        return med(Pord[ms], 9)

    # ---- 聚合指标:HMM vs geo vs production ----
    held_use = [e for e in held[:140] if loadep(e)[3] >= 30]
    agg = {m: {"mono": [], "rough": [], "v0": [], "vend": [], "div_prod": []} for m in ("prod", "geo", "hmm")}
    for e in held_use:
        a_, r_, s_, n = loadep(e)
        prod = M.variants(a_, r_, s_)["cond"]; g = geo_ms(a_, r_, s_); h, _ = hmm.value(a_, r_, s_)
        for nm, v in (("prod", prod), ("geo", g), ("hmm", h)):
            agg[nm]["mono"].append(mono(v)); agg[nm]["rough"].append(float(np.abs(np.diff(v)).mean()))
            agg[nm]["v0"].append(float(v[0])); agg[nm]["vend"].append(float(v[-1]))
            agg[nm]["div_prod"].append(float(np.abs(v - prod).max()))
    aggm = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in agg.items()}

    # ---- §6.1 回退可观测:undo→redo 注入,HMM 是否照样掉(挑 HMM 值确实升到高位的 ep)----
    cand61 = [(e, hmm.value(*loadep(e)[:3])[0]) for e in sorted(held_use, key=lambda e: abs(loadep(e)[3] - 150))[:25]]
    ex, vc = next(((e, v) for e, v in cand61 if v.max() >= 0.75), cand61[0])
    aa, rr, st, n = loadep(ex)
    hi = np.where(vc >= 0.7)[0]; k = int(hi[0]) if len(hi) else int(.7 * n)   # 此处 value 已高
    lo = np.where(vc <= 0.15)[0]; ai = int(lo[len(lo) // 2]) if len(lo) else int(.08 * n); ai = min(ai, k - 5)
    prefix = np.arange(0, k + 1); undo = np.arange(k, ai - 1, -1)             # 前缀收在高位 vc[k],再倒放回早期
    sel = np.r_[prefix, undo, np.arange(ai, n)]
    vpert, _ = hmm.value(aa[sel], rr[sel], st[sel]); reg = (len(prefix), len(prefix) + len(undo))
    pre = float(vc[k]); dip = float(vpert[reg[0]:reg[1]].min())

    # ---- §6.2 复现:取 ep2271(已知复现显著)----
    e2 = 2271 if 2271 in held else held_use[0]
    a2, r2, s2, n2 = loadep(e2); vh2, _ = hmm.value(a2, r2, s2)
    raw2 = Pord[np.linalg.norm(M.emb(a2, r2, s2)[:, None] - M.C[None], axis=2).argmin(1)]

    print("\n==== 聚合(held N={}) prod / geo / hmm ====".format(len(held_use)), flush=True)
    for k_ in ("mono", "rough", "v0", "vend", "div_prod"):
        print(f"  {k_:<9} {aggm['prod'][k_]:.3f} / {aggm['geo'][k_]:.3f} / {aggm['hmm'][k_]:.3f}", flush=True)
    print(f"§6.1 HMM 回退: ep{ex} pre={pre:.2f}→dip={dip:.2f} (能掉={'是' if pre-dip>0.15 else '否'})", flush=True)
    print(f"§6.2 HMM 复现: ep{e2} mono={mono(vh2):.3f} (raw mono={mono(raw2):.3f})", flush=True)

    # ---- 图 ----
    exs = sorted(held_use, key=lambda e: abs(loadep(e)[3] - 130))[:3]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4)); ax = ax.ravel()
    for kk, e in enumerate(exs):
        a_, r_, s_, nn = loadep(e); prod = M.variants(a_, r_, s_)["cond"]; g = geo_ms(a_, r_, s_); h, _ = hmm.value(a_, r_, s_)
        x = np.arange(nn)
        ax[kk].plot(x, prod, color="#2b8cbe", lw=2.0, ls="--", label=f"deployed bin cond (mono {mono(prod):.2f})")
        ax[kk].plot(x, g, color="#999", lw=1.5, alpha=.7, label=f"geom ms-Viterbi (mono {mono(g):.2f})")
        ax[kk].plot(x, h, color="#c0392b", lw=2.4, label=f"NEW milestone-HMM (mono {mono(h):.2f})")
        ax[kk].set_title(f"ep{e} (n={nn})", fontsize=10); ax[kk].set_ylim(-.05, 1.08); ax[kk].grid(alpha=.25)
        ax[kk].set_xlabel("frame (3Hz)"); ax[kk].set_ylabel("value"); ax[kk].legend(fontsize=8, loc="lower right")
    # §6.1
    xp = np.arange(len(sel))
    ax[3].axvspan(reg[0], reg[1], color="#f1c40f", alpha=.25, label="injected regression")
    ax[3].plot(xp, np.maximum.accumulate(vpert), color="#555", ls="--", lw=1.6, label="hard-monotone (erased)")
    ax[3].plot(xp, vpert, color="#c0392b", lw=2.4, label=f"HMM value {pre:.2f}→{dip:.2f}")
    ax[3].set_title(f"(§6.1) HMM regression observability ep{ex}", fontsize=10); ax[3].set_ylim(-.05, 1.08)
    ax[3].grid(alpha=.25); ax[3].set_xlabel("frame"); ax[3].set_ylabel("value"); ax[3].legend(fontsize=8, loc="lower right")
    # §6.2
    ax[4].plot(np.arange(n2), raw2, color="#d7191c", lw=.8, alpha=.4, label="raw nearest-ms")
    ax[4].plot(np.arange(n2), vh2, color="#c0392b", lw=2.4, label=f"HMM (mono {mono(vh2):.2f})")
    ax[4].set_title(f"(§6.2) HMM recurrence robustness ep{e2}", fontsize=10); ax[4].set_ylim(-.05, 1.08)
    ax[4].grid(alpha=.25); ax[4].set_xlabel("frame"); ax[4].set_ylabel("value"); ax[4].legend(fontsize=8, loc="lower right")
    # metrics bar
    labels = ["mono", "rough×5", "start v0", "end v"]
    prod_b = [aggm["prod"]["mono"], aggm["prod"]["rough"] * 5, aggm["prod"]["v0"], aggm["prod"]["vend"]]
    hmm_b = [aggm["hmm"]["mono"], aggm["hmm"]["rough"] * 5, aggm["hmm"]["v0"], aggm["hmm"]["vend"]]
    xb = np.arange(4); w = .38
    ax[5].bar(xb - w/2, prod_b, w, color="#2b8cbe", label="deployed")
    ax[5].bar(xb + w/2, hmm_b, w, color="#c0392b", label="milestone-HMM")
    ax[5].set_xticks(xb); ax[5].set_xticklabels(labels, fontsize=9); ax[5].set_title("Aggregate (held N={})".format(len(held_use)), fontsize=10)
    ax[5].legend(fontsize=9); ax[5].grid(alpha=.25, axis="y")
    fig.suptitle(f"Milestone-HMM value readout (start=minPord, end=maxPord, emission=soft-assign, transition=empirical+dwell) "
                 f"τ={a.tau} pstay={pstay:.2f}", fontsize=12.5, y=1.0)
    fig.tight_layout(); fig.savefig(OUTV / "milestone_hmm.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nSAVED milestone_hmm.png", flush=True)
    json.dump({"tau": a.tau, "pstay": pstay, "mean_dwell": dwell, "agg": aggm,
               "regression": {"ep": int(ex), "pre": pre, "dip": dip},
               "recurrence": {"ep": int(e2), "hmm_mono": float(mono(vh2)), "raw_mono": float(mono(raw2))}},
              open(out_dir("crave_a1a2") / "milestone_hmm.json", "w"), indent=2)


if __name__ == "__main__":
    main()
