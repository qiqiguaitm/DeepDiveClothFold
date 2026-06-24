"""把"转移感知 Viterbi"新方法的 value 曲线渲染出来直观看(承 §8)。
每条 held ep 叠三线:raw 最近-milestone / 当前部署值(bin-space cond_end)/ 新方法(转移感知 ms-Viterbi ④)。
并标注新方法 vs 部署值的最大偏差,诚实显示"差在哪/差多少"。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/render_transition_value.py [--mine-n 200] [--n 8]
输出: crave/docs/visualization/viterbi/transition_value_curves.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from crave.data.cache import list_cache_eps
from crave.render import setup_mpl
from crave.utils import med

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viterbi_compare import CFG, Model, OUTV, loadep  # noqa: E402
from milestone_transition_viterbi import build_pen, emit_ms, estimate_transition, viterbi_pen  # noqa: E402

plt = setup_mpl()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    rawset = set(list_cache_eps(CFG.raw_cache))
    all_eps = sorted(e for e in list_cache_eps(CFG.arm_cache) if e in rawset)
    perm = np.random.RandomState(0).permutation(all_eps)
    mined = sorted(perm[:a.mine_n].tolist()); held = [e for e in perm[a.mine_n:].tolist()]
    M = Model(mined); Pord = M.Pord
    P, counts = estimate_transition(M, mined)
    pen = build_pen(counts, Pord, lam_geo=8.0, beta=1.0, back_barrier=0.0)   # 新方法 ④(dwell-free 居中前进锐化)

    # 取长度多样的 held ep
    held = [e for e in held if loadep(e)[3] >= 40]
    held.sort(key=lambda e: loadep(e)[3])
    pick = [held[int(x)] for x in np.linspace(0, len(held) - 1, a.n)]

    rows = a.n // 2 + a.n % 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 3.1 * rows)); axes = axes.ravel()
    maxds = []
    for k, e in enumerate(pick):
        a_, r_, s_, n = loadep(e)
        prod = M.variants(a_, r_, s_)["cond"]                       # 当前部署值
        em = emit_ms(M, a_, r_, s_)
        vt = med(Pord[viterbi_pen(em, pen)], 9)                     # 新方法
        raw = Pord[em.argmin(1)]
        md = float(np.abs(vt - prod).max()); maxds.append(md)
        x = np.arange(n)
        ax = axes[k]
        ax.plot(x, raw, color="#d7191c", lw=0.8, alpha=.35, label="raw nearest-milestone")
        ax.plot(x, prod, color="#2b8cbe", lw=2.2, ls="--", label="current deployed (bin cond_end)")
        ax.plot(x, vt, color="#1a9641", lw=2.4, label="new: transition-aware ms")
        ax.set_title(f"ep{e}  (n={n}, max|new−deployed|={md:.2f})", fontsize=10)
        ax.set_ylim(-0.05, 1.08); ax.grid(alpha=.25); ax.set_xlabel("frame (3Hz)"); ax.set_ylabel("value")
        ax.legend(fontsize=7.5, loc="lower right")
    for k in range(len(pick), len(axes)): axes[k].axis("off")
    fig.suptitle(f"New transition-aware Viterbi value vs current deployed value · {a.n} held-out eps "
                 f"(mean max-divergence={np.mean(maxds):.2f})", fontsize=12.5, y=1.005)
    fig.tight_layout(); fig.savefig(OUTV / "transition_value_curves.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED transition_value_curves.png  eps={pick}", flush=True)
    print(f"新方法 vs 部署值 最大偏差: mean={np.mean(maxds):.3f}  max={np.max(maxds):.3f}  min={np.min(maxds):.3f}", flush=True)


if __name__ == "__main__":
    main()
