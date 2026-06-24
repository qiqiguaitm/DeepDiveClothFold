"""Viterbi 鲁棒性多次测试:两项跟踪能力做成统计量,给"是否够鲁棒"的判据。

(A) 回退跟踪鲁棒性 —— 分级深度注入(不只 on/off,测"按比例跟踪"):
    对每条 held ep,在高进度锚点 k 后注入 undo→redo,把折叠倒放回不同目标进度
    {0.50,0.35,0.20,0.08},量 observed_drop = pre − dip 是否随 regression_depth = pre − undo_floor
    线性跟踪。指标:跟踪保真度 corr、检出率(drop≥0.15)、恢复率、以及"干净 ep 不乱掉"的特异度。

(B) 循环 milestone 跟踪鲁棒性 —— 跨所有自然复现 ep:
    量 Viterbi vs raw 的"最大伪回退" max_backward = max(cummax(v)−v),以及 mono;
    复现把 raw 拽得乱抖(max_backward 大),Viterbi 应把它压住(小)且 mono 高。

数据 kai0_base,复用 viterbi_compare.Model。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/viterbi_robustness.py [--mine-n 200] [--n-test 80]
输出: crave/docs/visualization/viterbi/viterbi_robustness.png + temp/crave_a1a2/viterbi_robustness.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from crave.config import out_dir
from crave.data.cache import list_cache_eps
from crave.render import setup_mpl
from crave.utils import mono

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viterbi_compare import CFG, Model, OUTV, loadep  # noqa: E402

plt = setup_mpl()
TARGETS = [0.50, 0.35, 0.20, 0.08]            # undo 倒放回的目标进度档


def ranges(idx):
    o = []; s0 = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None: o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None: o.append((s0, pv))
    return [(s, e) for s, e in o if e - s >= 2]


def max_backward(v):
    """最大伪回退:cummax(v)−v 的峰值(平滑单调曲线≈0,复现/回退抬高它)。"""
    return float(np.max(np.maximum.accumulate(v) - v))


def inject_undo_redo(aa, rr, st, v_clean, k, a_idx):
    """到高进度 k 后沿真实帧倒放回 a_idx(undo),再正放续到末(redo)。返回 (v_pert, region)。"""
    down = np.arange(k - 1, a_idx - 1, -1)
    sel = np.r_[np.arange(0, k), down, np.arange(a_idx, len(aa))]
    Rp = M_GLOBAL.variants(aa[sel], rr[sel], st[sel])
    return Rp["cond"], (k, k + len(down))


def main():
    global M_GLOBAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)          # 换挖矿/held 划分,跨种子复测稳定性
    a = ap.parse_args()
    rawset = set(list_cache_eps(CFG.raw_cache))
    all_eps = sorted(e for e in list_cache_eps(CFG.arm_cache) if e in rawset)
    perm = np.random.RandomState(a.seed).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined); M_GLOBAL = M

    # ================= (A) 回退跟踪鲁棒性:分级深度注入 =================
    reg_depth, obs_drop, det_ok, rec_ok, tg_of = [], [], [], [], []
    clean_false_dip = []                          # 特异度:干净 ep 末段最大伪回退
    n_used = 0
    for e in held:
        if n_used >= a.n_test: break
        aa, rr, st, n = loadep(e)
        v_clean = M.variants(aa, rr, st)["cond"]
        hi = np.where(v_clean >= 0.75)[0]
        if len(hi) == 0 or n < 30: continue        # 没做到高进度的 ep 跳过
        k = int(hi[0]); pre = float(v_clean[max(0, k - 1)])
        if k < 12: continue
        n_used += 1
        clean_false_dip.append(max_backward(v_clean))   # 干净对照(无注入)
        for tg in TARGETS:
            cand = np.where(v_clean[:k - 4] <= tg)[0]
            a_idx = int(cand[-1]) if len(cand) else 2   # 最靠近 k 且 value≤tg 的帧
            a_idx = min(a_idx, k - 5)
            undo_floor = float(v_clean[a_idx])
            v_pert, reg = inject_undo_redo(aa, rr, st, v_clean, k, a_idx)
            dip = float(v_pert[reg[0]:reg[1]].min())
            post = float(v_pert[reg[1]:].max()) if reg[1] < len(v_pert) else dip
            reg_depth.append(pre - undo_floor)
            obs_drop.append(pre - dip)
            det_ok.append((pre - dip) >= 0.15)
            rec_ok.append(post >= pre - 0.12)
            tg_of.append(tg)
    reg_depth = np.array(reg_depth); obs_drop = np.array(obs_drop)
    fid = float(np.corrcoef(reg_depth, obs_drop)[0, 1]) if len(reg_depth) > 2 else float("nan")
    slope = float(np.polyfit(reg_depth, obs_drop, 1)[0]) if len(reg_depth) > 2 else float("nan")
    det_rate = float(np.mean(det_ok)); rec_rate = float(np.mean(rec_ok))
    clean_false_dip = np.array(clean_false_dip)
    tg_of = np.array(tg_of); det_ok = np.array(det_ok)
    det_by_tg = {f"{t:.2f}": float(det_ok[tg_of == t].mean()) for t in TARGETS}    # 各深度检出率

    # ================= (B) 循环 milestone 跟踪鲁棒性 =================
    rec_mono_v, rec_mono_raw, rec_back_v, rec_back_raw, n_recur_ep = [], [], [], [], 0
    for e in held[:max(120, a.n_test)]:
        a_, r_, s_, ne = loadep(e)
        R = M.variants(a_, r_, s_); v, raw = R["cond"], R["raw"]
        Fq = M.emb(a_, r_, s_); d = np.linalg.norm(Fq[:, None] - M.C[None], axis=2); am = d.argmin(1)
        tnorm = np.arange(ne) / max(1, ne - 1)
        has_recur = any(len(ranges(np.where(am == ci)[0].tolist())) >= 2 and
                        (lambda rg: tnorm[rg[-1][0]] - tnorm[rg[0][0]] > 0.25)(ranges(np.where(am == ci)[0].tolist()))
                        for ci in range(len(M.order)) if len(ranges(np.where(am == ci)[0].tolist())) >= 2)
        if not has_recur: continue
        n_recur_ep += 1
        rec_mono_v.append(mono(v)); rec_mono_raw.append(mono(raw))
        rec_back_v.append(max_backward(v)); rec_back_raw.append(max_backward(raw))
    rec_mono_v = np.array(rec_mono_v); rec_mono_raw = np.array(rec_mono_raw)
    rec_back_v = np.array(rec_back_v); rec_back_raw = np.array(rec_back_raw)
    suppress = float(np.median(rec_back_raw) / max(1e-6, np.median(rec_back_v)))

    # ================= 判据 =================
    verdict = {
        "A_tracking_fidelity_corr": fid, "A_slope": slope,
        "A_detection_rate(drop>=0.15)": det_rate, "A_detection_by_target": det_by_tg, "A_recovery_rate": rec_rate,
        "A_clean_false_dip_p90": float(np.quantile(clean_false_dip, 0.90)),
        "A_n_trials": int(len(reg_depth)), "A_n_eps": int(n_used),
        "B_n_recurring_eps": int(n_recur_ep),
        "B_viterbi_mono_mean": float(rec_mono_v.mean()) if len(rec_mono_v) else None,
        "B_raw_mono_mean": float(rec_mono_raw.mean()) if len(rec_mono_raw) else None,
        "B_viterbi_maxback_median": float(np.median(rec_back_v)) if len(rec_back_v) else None,
        "B_raw_maxback_median": float(np.median(rec_back_raw)) if len(rec_back_raw) else None,
        "B_suppression_factor": suppress,
    }
    print("\n==== (A) 回退跟踪鲁棒性 ====", flush=True)
    print(f"  trials={len(reg_depth)} (eps={n_used}×{len(TARGETS)} targets)", flush=True)
    print(f"  跟踪保真度 corr(regression_depth, observed_drop) = {fid:.3f}  (slope {slope:.2f})", flush=True)
    print(f"  检出率(drop≥0.15) 总={det_rate:.0%};按倒放深度 " +
          " ".join(f"tg{t:.2f}→{det_by_tg[f'{t:.2f}']:.0%}" for t in TARGETS), flush=True)
    print(f"  恢复率(redo后≥pre−0.12) = {rec_rate:.0%}", flush=True)
    print(f"  特异度:干净 ep 伪回退 p90 = {np.quantile(clean_false_dip,0.90):.3f}(越小越不误报)", flush=True)
    print("==== (B) 循环 milestone 跟踪鲁棒性 ====", flush=True)
    print(f"  recurring eps={n_recur_ep}", flush=True)
    print(f"  Viterbi mono={rec_mono_v.mean():.3f} vs raw mono={rec_mono_raw.mean():.3f}", flush=True)
    print(f"  最大伪回退 median: Viterbi={np.median(rec_back_v):.3f} vs raw={np.median(rec_back_raw):.3f} → 抑制 {suppress:.1f}×", flush=True)

    # ================= 图 =================
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # A1: 跟踪保真度散点
    ax[0].scatter(reg_depth, obs_drop, s=18, alpha=.45, color="#c0392b", edgecolor="none")
    lim = [0, max(0.9, float(obs_drop.max()) if len(obs_drop) else 0.9)]
    ax[0].plot(lim, lim, "k--", lw=1, alpha=.6, label="ideal y=x (faithful tracking)")
    ax[0].axhline(0.15, color="#888", lw=.8, ls=":", label="detection floor 0.15")
    ax[0].set_xlabel("injected regression depth (pre − undo_floor)")
    ax[0].set_ylabel("observed value drop (pre − dip)")
    ax[0].set_title(f"(A) Regression TRACKING fidelity\ncorr={fid:.2f}, slope={slope:.2f}, detect={det_rate:.0%}, recover={rec_rate:.0%}", fontsize=10.5)
    ax[0].legend(fontsize=8.5, loc="upper left"); ax[0].grid(alpha=.25)
    # A2: 特异度 —— 干净 ep 伪回退分布(应集中在低位)
    ax[1].hist(clean_false_dip, bins=18, color="#2980b9", alpha=.8)
    ax[1].axvline(0.15, color="#c0392b", lw=1.2, ls="--", label="detection floor 0.15")
    ax[1].set_xlabel("max spurious backward dip on CLEAN eps (no injection)")
    ax[1].set_ylabel("# eps")
    ax[1].set_title(f"(A) Specificity: clean eps don't false-dip\np90={np.quantile(clean_false_dip,0.90):.2f} (< floor → low false alarm)", fontsize=10.5)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)
    # B: 循环鲁棒 —— max_backward raw vs viterbi
    bp = ax[2].boxplot([rec_back_raw, rec_back_v], tick_labels=["raw\n(no Viterbi)", "Viterbi\ncond_end"], patch_artist=True, widths=.55)
    for patch, c in zip(bp["boxes"], ["#d7191c", "#1a9641"]): patch.set_facecolor(c); patch.set_alpha(.55)
    ax[2].set_ylabel("max spurious backward dip on recurring eps")
    ax[2].set_title(f"(B) Recurring-milestone robustness · {n_recur_ep} eps\nViterbi suppresses spurious oscillation {suppress:.1f}× (mono {rec_mono_v.mean():.2f} vs {rec_mono_raw.mean():.2f})", fontsize=10.5)
    ax[2].grid(alpha=.25, axis="y")
    fig.suptitle("Viterbi robustness over many held-out episodes — regression tracking (A) + recurring-milestone robustness (B)", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_robustness.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nSAVED viterbi_robustness.png", flush=True)
    json.dump(verdict, open(out_dir("crave_a1a2") / "viterbi_robustness.json", "w"), indent=2)


if __name__ == "__main__":
    main()
