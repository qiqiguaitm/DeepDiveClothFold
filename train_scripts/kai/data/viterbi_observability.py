"""Viterbi 的两项观测能力验证(承接 viterbi_compare):
  (A) 回退可观测:value 没被锁死为单调。注入一段"操作失误/无效操作"(把中段帧换成更早状态的真实帧)→
      value 应如实掉下去再恢复;对比"硬单调(running-max)"会把这个回退抹平 → 证明对称惩罚保住了观测能力。
  (B) 循环 milestone 兼顾:同一 milestone 簇在一条 ep 里被多次命中(状态复现)。短暂复现被转移惩罚+中值滤波
      平滑(value 不乱抖),只有持续回到早期态(真回退)才掉 → DP 自动区分良性复现 vs 真回退。

数据 kai0_base(kai-only),复用 viterbi_compare.Model。英文标注(venv 无中文字体)。
Run: kai0/.venv/bin/python train_scripts/kai/data/viterbi_observability.py [--mine-n 200]
输出: docs/visualization/cross_episode_recurrence_value/viterbi_regression.png, viterbi_recurrence.png
"""
import argparse, json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crave_value as cv
from viterbi_compare import Model, loadep, OUTV, RAW, ARM


def ranges(idx):
    """连续帧段 [(s,e)]。"""
    o = []; s0 = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None: o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None: o.append((s0, pv))
    return [(s, e) for s, e in o if e - s >= 2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mine-n", type=int, default=200); a = ap.parse_args()
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined)

    # ============ (A) 回退可观测:注入"无效操作/失误" ============
    ex = sorted(held, key=lambda e: abs(loadep(e)[3] - 150))[0]
    aa, rr, st, n = loadep(ex)
    v_clean = M.variants(aa, rr, st)["cond"]
    # 注入"深且持续"的回退:在高进度处(value≥0.7)插入一段真实"很早期(value≤0.15)"的帧 = 折叠被弄散
    # 忠实模拟"操作失误→回退→重做":到高进度 k 后,沿真实帧平滑倒放回早期 a(undo),再正放回去续到末(redo)
    hi = np.where(v_clean >= 0.7)[0]; k = int(hi[0]) if len(hi) else int(0.75 * n)
    lo = np.where(v_clean <= 0.15)[0]; a_idx = int(lo[len(lo) // 2]) if len(lo) else int(0.08 * n)
    a_idx = min(a_idx, k - 5)
    down = np.arange(k - 1, a_idx - 1, -1)                    # k→a 倒放 = undo(平滑后退)
    sel = np.r_[np.arange(0, k), down, np.arange(a_idx, n)]   # up → undo → redo+continue
    ap_, rp_, sp_ = aa[sel], rr[sel], st[sel]
    Rp = M.variants(ap_, rp_, sp_); v_pert = Rp["cond"]; raw_pert = Rp["raw"]
    v_hard = np.maximum.accumulate(v_pert)                    # 硬单调(running-max)对照
    xp = np.arange(len(sel)); reg = (k, k + len(down))
    pre = float(v_pert[max(0, k - 1)]); dip = float(v_pert[reg[0]:reg[1]].min())

    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.axvspan(reg[0], reg[1], color="#f1c40f", alpha=.25, label="injected regression: fold undone (real early-state frames)")
    ax.plot(xp, raw_pert, color="#d7191c", lw=0.9, alpha=.45, label="no Viterbi (raw)")
    ax.plot(xp, v_pert, color="#c0392b", lw=2.8, label=f"Viterbi cond_end — value CRASHES {pre:.2f}→{dip:.2f} then recovers")
    ax.plot(xp, v_hard, color="#555", lw=2.0, ls="--", label="hard-monotone (running-max) — regression ERASED")
    ax.annotate("mistake observed", xy=((reg[0] + reg[1]) / 2, dip), xytext=((reg[0] + reg[1]) / 2, dip - 0.0),
                fontsize=9, color="#c0392b", ha="center", va="top")
    ax.set_xlabel("frame (3Hz)"); ax.set_ylabel("value"); ax.set_ylim(-0.05, 1.08); ax.grid(alpha=.25); ax.legend(fontsize=9, loc="center right")
    ax.set_title(f"(A) Regression observability · ep{ex}: symmetric transition penalty lets value DIP on a sustained mistake "
                 f"({pre:.2f}→{dip:.2f}), then recover;\na hard-monotone lock (running-max) would silently erase the mistake", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_regression.png", dpi=140); plt.close(fig)
    print(f"SAVED viterbi_regression.png  (clean mono={cv.mono(v_clean):.2f}, perturbed mono={cv.mono(v_pert):.2f}, dip={v_pert[reg[0]:reg[1]].min():.2f})", flush=True)

    # ============ (B) 循环 milestone 兼顾 ============
    # 统计:跨 held-out ep,有多少 milestone 在一条 ep 里被多次命中(时间跨度大=复现)
    nrec, ntot, recur_eps = 0, 0, []
    for e in held[:120]:
        a_, r_, s_, ne = loadep(e)
        Fq = M.emb(a_, r_, s_); d = np.linalg.norm(Fq[:, None] - M.C[None], axis=2); am = d.argmin(1)
        tnorm = np.arange(ne) / max(1, ne - 1)
        ep_recur = 0
        for ci in range(len(M.order)):
            rg = ranges(np.where(am == ci)[0].tolist())
            if len(rg) >= 2:
                tspan = tnorm[rg[-1][0]] - tnorm[rg[0][0]]
                if tspan > 0.25: ep_recur += 1
            ntot += 1
        if ep_recur: nrec += ep_recur; recur_eps.append((e, ep_recur, ne))
    frac = nrec / max(1, ntot)
    recur_eps.sort(key=lambda t: -t[1])
    print(f"循环 milestone: {nrec} 个(milestone×ep)出现复现(≥2段且时间跨度>0.25); 占比 {frac:.1%}", flush=True)

    # 取复现最明显的一条 ep 画图
    e2, _, ne2 = recur_eps[0]; a2, r2, s2, _ = loadep(e2)
    R2 = M.variants(a2, r2, s2); v2, raw2 = R2["cond"], R2["raw"]
    Fq2 = M.emb(a2, r2, s2); d2 = np.linalg.norm(Fq2[:, None] - M.C[None], axis=2); am2 = d2.argmin(1)
    tnorm2 = np.arange(ne2) / max(1, ne2 - 1)
    # 找一个复现的 milestone(命中多段、跨度大)
    rec_ci, rec_rgs = None, None
    for ci in range(len(M.order)):
        rg = ranges(np.where(am2 == ci)[0].tolist())
        if len(rg) >= 2 and (tnorm2[rg[-1][0]] - tnorm2[rg[0][0]]) > 0.25:
            rec_ci, rec_rgs = ci, rg; break

    fig, ax = plt.subplots(figsize=(12, 5.0))
    x2 = np.arange(ne2)
    ax.plot(x2, raw2, color="#d7191c", lw=0.9, alpha=.45, label="no Viterbi (raw nearest-milestone progress)")
    ax.plot(x2, v2, color="#1a9641", lw=2.6, label=f"Viterbi cond_end (mono={cv.mono(v2):.2f})")
    if rec_ci is not None:
        for s, en in rec_rgs:
            ax.axvspan(s, en, color="#2b8cbe", alpha=.18)
        ax.axvspan(rec_rgs[0][0], rec_rgs[0][1], color="#2b8cbe", alpha=.18,
                   label=f"recurring milestone c (Pord={M.Pord[rec_ci]:.2f}) hit at {len(rec_rgs)} separated ranges")
        ax.axhline(M.Pord[rec_ci], color="#2b8cbe", lw=.8, ls=":")
    ax.set_xlabel("frame (3Hz)"); ax.set_ylabel("value"); ax.set_ylim(-0.05, 1.08); ax.grid(alpha=.25); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"(B) Recurring-milestone robustness · ep{e2}: a low-progress milestone recurs late (blue bands);\n"
                 f"raw nearest-milestone would yank value back, Viterbi (transition penalty + median) stays smooth — brief recurrence absorbed, no spurious oscillation", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUTV / "viterbi_recurrence.png", dpi=140); plt.close(fig)
    print(f"SAVED viterbi_recurrence.png  (ep{e2}, recurring milestone Pord={M.Pord[rec_ci] if rec_ci is not None else -1:.2f})", flush=True)

    json.dump({"regression": {"ep": int(ex), "mono_clean": float(cv.mono(v_clean)), "mono_perturbed": float(cv.mono(v_pert)),
                              "dip_value": float(v_pert[reg[0]:reg[1]].min())},
               "recurrence": {"frac_milestone_ep_recurring": frac, "n_recurring": nrec, "example_ep": int(e2),
                              "example_milestone_Pord": float(M.Pord[rec_ci]) if rec_ci is not None else None}},
              open(Path("/vePFS/tim/workspace/deepdive_kai0/temp/crave_a1a2/viterbi_observability.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
