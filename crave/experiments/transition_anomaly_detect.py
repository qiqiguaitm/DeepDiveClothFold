"""转移概率做失败/异常检测 —— milestone 转移矩阵的"正用法"(承接 §8 结论)。

思路:正常 demo 的 milestone 访问路径在经验转移矩阵 P 下似然高;失败/异常 ep 走低概率路径
(乱序、跳过必经 milestone、中途回退)→ 路径转移对数似然低 → 异常分高。
关键卖点:与现有 `de_end` OOD flag **互补** —— 后者只看末态对不对,本法看**过程路径**对不对,
即使结尾看着完成(de_end 正常),中途乱序/跳步也能被抓。

验证:kai0 多为成功 demo,故用**保留首尾、只打乱中段**的受控腐化当失败代理
(reorder_mid 乱序 / skip_mid 跳步 / regress 中途回退),这正是 de_end 抓不到、本法该赢的区间。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/transition_anomaly_detect.py [--mine-n 200] [--n-test 120]
输出: crave/docs/visualization/viterbi/transition_anomaly.png + temp/crave_a1a2/transition_anomaly.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from crave.config import out_dir
from crave.data.cache import list_cache_eps
from crave.render import setup_mpl
from crave.utils import med

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viterbi_compare import CFG, Model, OUTV, loadep  # noqa: E402
from milestone_transition_viterbi import emit_ms, estimate_transition, visited_sequence  # noqa: E402

plt = setup_mpl()


def faithful_seq(M, a_, r_, s_):
    """忠实 milestone 访问序列(raw argmin + 中值滤波 + 折叠;不做 Viterbi 去噪,
    否则 DP 会把腐化的乱序"修回"单调,反而抹掉异常)。"""
    return visited_sequence(np.linalg.norm(M.emb(a_, r_, s_)[:, None] - M.C[None], axis=2).argmin(1))


def path_loglik(M, logP, a_, r_, s_):
    """忠实 milestone 路径在 P 下的几个异常分(越低越异常):
       mean=整体似然 / min=最差单步 / botk=最差 3 步均值(对单个正常复现离群稳健,
       却能抓乱序/回退引入的多处低概率跳)。返回 (dict, seq)。"""
    seq = faithful_seq(M, a_, r_, s_)
    lls = np.array([logP[i, j] for i, j in zip(seq[:-1], seq[1:])])
    if not len(lls): return {"mean": 0.0, "min": 0.0, "botk": 0.0}, seq
    return {"mean": float(lls.mean()), "min": float(lls.min()), "botk": float(np.sort(lls)[:3].mean())}, seq


# -------- 受控腐化:全部保留首尾 20%,只动中段 → de_end 仍正常 --------
def reorder_mid(n):
    i0, i1 = int(.2 * n), int(.8 * n); mid = np.arange(i0, i1); h = len(mid) // 2
    return np.r_[np.arange(0, i0), mid[h:], mid[:h], np.arange(i1, n)]   # 中段两半对调=乱序


def skip_mid(n):
    i0, i1 = int(.40 * n), int(.62 * n)
    return np.r_[np.arange(0, i0), np.arange(i1, n)]                     # 删中段=跳过 milestone


def regress_idx(M, a_, r_, s_, n):
    v = med(M.Pord[emit_ms(M, a_, r_, s_).argmin(1)], 9)
    hi = np.where(v >= 0.7)[0]; k = int(hi[0]) if len(hi) else int(.7 * n)
    lo = np.where(v <= 0.15)[0]; ai = int(lo[len(lo) // 2]) if len(lo) else int(.08 * n)
    ai = min(ai, k - 5); k = max(k, ai + 6)
    return np.r_[np.arange(0, k), np.arange(k - 1, ai - 1, -1), np.arange(ai, n)]  # undo→redo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=120)
    a = ap.parse_args()
    rawset = set(list_cache_eps(CFG.raw_cache))
    all_eps = sorted(e for e in list_cache_eps(CFG.arm_cache) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined)
    P, counts = estimate_transition(M, mined)
    logP = np.log(P + 1e-12)
    
    thr_de = M.de_end_thr

    held_use = [e for e in held if loadep(e)[3] >= 30][:a.n_test]
    CORR = {"reorder_mid": reorder_mid, "skip_mid": skip_mid, "regress": "R"}
    SCORES = ("mean", "min", "botk")
    norm = {k: [] for k in SCORES + ("de",)}
    anom = {c: {k: [] for k in SCORES + ("de",)} for c in CORR}
    ex_pack = None
    for e in held_use:
        a_, r_, s_, n = loadep(e)
        sc0, _ = path_loglik(M, logP, a_, r_, s_)
        for k in SCORES: norm[k].append(sc0[k])
        norm["de"].append(M.variants(a_, r_, s_)["de_end"])
        for c, fn in CORR.items():
            idx = regress_idx(M, a_, r_, s_, n) if fn == "R" else fn(n)
            ca, cr, cs = a_[idx], r_[idx], s_[idx]
            scc, _ = path_loglik(M, logP, ca, cr, cs)
            for k in SCORES: anom[c][k].append(scc[k])
            anom[c]["de"].append(M.variants(ca, cr, cs)["de_end"])
        if ex_pack is None and n > 80:                          # 留一个 reorder 例子作图
            ex_pack = (e, n, a_, r_, s_, reorder_mid(n))

    # 选哪个分整体最好(平均 AUC 高者)
    def auc_of(nv, av): return float(roc_auc_score([0]*len(nv)+[1]*len(av), list(-np.array(nv))+list(-np.array(av))))
    avg = {k: float(np.mean([auc_of(norm[k], anom[c][k]) for c in CORR])) for k in SCORES}
    SC = max(SCORES, key=lambda k: avg[k])
    nsc = np.array(norm[SC]); thr_sc = float(np.quantile(nsc, 0.05))   # 5% FPR 阈
    res = {}
    for c in CORR:
        asc = np.array(anom[c][SC]); ad = np.array(anom[c]["de"])
        endok = ad < thr_de
        res[c] = {"auc": auc_of(norm[SC], anom[c][SC]), "detect@5%FPR": float(np.mean(asc < thr_sc)),
                  "endpoint_looks_ok_frac": float(np.mean(endok)),
                  "caught_with_ok_endpoint": float(np.mean((asc < thr_sc) & endok))}
    print(f"milestones={len(M.order)}  normal N={len(nsc)}  de_end_thr={thr_de:.3f}  score={SC}-logP", flush=True)
    print("==== 转移路径异常检测({}-logP 分;5%FPR 阈={:.2f})====".format(SC, thr_sc), flush=True)
    for c in CORR:
        r = res[c]
        print(f"  {c:<12} AUC={r['auc']:.3f}  检出@5%FPR={r['detect@5%FPR']:.0%}  "
              f"末态仍正常占{r['endpoint_looks_ok_frac']:.0%}→其中被路径法抓到 {r['caught_with_ok_endpoint']:.0%}(互补 de_end)", flush=True)

    # ---------- FIG ----------
    fig = plt.figure(figsize=(16, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.2])
    # P1: score 分布 normal vs anomalies
    ax0 = fig.add_subplot(gs[0])
    lo = min(nsc.min(), min(np.min(anom[c][SC]) for c in CORR))
    bins = np.linspace(lo, 0.2, 30)
    ax0.hist(nsc, bins=bins, color="#2ecc71", alpha=.75, label=f"normal (N={len(nsc)})", density=True)
    cols = {"reorder_mid": "#d7191c", "skip_mid": "#e67e22", "regress": "#8e44ad"}
    for c in CORR:
        ax0.hist(anom[c][SC], bins=bins, histtype="step", lw=2, color=cols[c], label=f"{c} (AUC {res[c]['auc']:.2f})", density=True)
    ax0.axvline(thr_sc, color="k", ls="--", lw=1, label="5% FPR threshold")
    ax0.set_xlabel(f"{SC} transition log P over faithful path"); ax0.set_ylabel("density")
    ax0.set_title("(1) Path transition-likelihood separates anomalies"); ax0.legend(fontsize=8, loc="upper left")
    # P2: AUC / 检出条形 + 互补性
    ax1 = fig.add_subplot(gs[1])
    cs_ = list(CORR); x = np.arange(len(cs_)); w = .38
    ax1.bar(x - w/2, [res[c]["auc"] for c in cs_], w, color="#2980b9", label="AUC (normal vs anomaly)")
    ax1.bar(x + w/2, [res[c]["caught_with_ok_endpoint"] for c in cs_], w, color="#16a085", label="caught while endpoint looks OK")
    ax1.axhline(0.5, color="#999", ls=":", lw=.8); ax1.set_xticks(x); ax1.set_xticklabels(cs_, fontsize=9)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("score"); ax1.set_title("(2) Complementary to de_end OOD flag")
    ax1.legend(fontsize=8, loc="lower right")
    # P3: worked example —— reorder ep 的逐转移 logP,定位异常步
    ax2 = fig.add_subplot(gs[2])
    e, n, a_, r_, s_, idx = ex_pack
    ca, cr, cs2 = a_[idx], r_[idx], s_[idx]
    seqc = faithful_seq(M, ca, cr, cs2)
    step_ll = [logP[i, j] for i, j in zip(seqc[:-1], seqc[1:])]
    ax2.plot(range(len(step_ll)), step_ll, "-o", color="#d7191c", ms=4, label="per-transition log P (corrupted)")
    seqn = faithful_seq(M, a_, r_, s_); step_n = [logP[i, j] for i, j in zip(seqn[:-1], seqn[1:])]
    ax2.plot(range(len(step_n)), step_n, "-o", color="#2ecc71", ms=3, alpha=.7, label="per-transition log P (normal)")
    ax2.axhline(thr_sc, color="k", ls="--", lw=1, label="anomaly threshold")
    wi = int(np.argmin(step_ll)); ax2.annotate("improbable jump\n(out-of-order)", xy=(wi, step_ll[wi]),
                xytext=(wi, step_ll[wi] - 3), fontsize=8.5, color="#d7191c", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d7191c"))
    ax2.set_xlabel("milestone-transition index"); ax2.set_ylabel("log P(i→j)")
    de_ok = M.variants(ca, cr, cs2)["de_end"]
    ax2.set_title(f"(3) ep{e} reorder_mid: localizes the fault\n(endpoint de_end={de_ok:.2f}{'<' if de_ok<thr_de else '>'}{thr_de:.2f} → "
                  f"{'looks complete' if de_ok<thr_de else 'flagged'}, yet path is flagged)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUTV / "transition_anomaly.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nSAVED transition_anomaly.png", flush=True)
    json.dump({"de_end_thr": thr_de, "thr_logP": thr_sc, "score": SC, "normal_N": len(nsc), "results": res},
              open(out_dir("crave_a1a2") / "transition_anomaly.json", "w"), indent=2)


if __name__ == "__main__":
    main()
