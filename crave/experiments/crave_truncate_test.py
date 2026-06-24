"""截断=失败rollout测试: 成功 ep 截到 50/70/90/100%(末帧是中间折叠态=未完成)。
验证: ① value 正确停在~截断进度(不被误拉到1.0); ② alignment-residual(末帧到 endK 完成态距离)能标出"未完成"=失败 flag。
对比 cond_end ON/OFF。

Thin entrypoint over `crave`: `FeatureSpace`/`DiscreteValue` come from `crave.value`,
`loadep` from `crave.data`, `smooth_monotone` from `crave.utils`, REPO from `crave.config`.
The kai0_base dataset + tcc_kai0_{raw,armmask} feature caches + the triple-cache
(_triple_prodtest) remain inlined — see TODO (shared with crave_value_prod_test.py).

跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_truncate_test.py
"""
import json

import numpy as np

from crave.config import REPO
from crave.data import loadep
from crave.render import setup_mpl
from crave.utils import smooth_monotone
from crave.value import DiscreteValue, FeatureSpace

# TODO(crave-lib): the kai0_base dataset + tcc_kai0_{raw,armmask} feature caches + the
# triple-cache (_triple_prodtest) should move into crave.config.datasets / crave.data —
# reused by crave_value_prod_test.py and crave_multimode_test.py too.
DS = REPO / "kai0/data/Task_A/kai0_base"
RAW = REPO / "temp/tcc_kai0_raw/feat_cache"; ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"; TRIPLE = REPO / "temp/_triple_prodtest"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]
OUTV = REPO / "crave/docs/visualization/centroid_decoder"


def main():
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    mine = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset and (TRIPLE / f"ep{e}.npz").exists())
    fs = FeatureSpace(TRIPLE, mine)
    dv_on = DiscreteValue(fs, mine, k=96, select="fixed", order="precedence", cond_end=True, log=lambda *_: None)
    dv_off = DiscreteValue(fs, mine, k=96, select="fixed", order="precedence", cond_end=False, log=lambda *_: None)
    thr = dv_on.de_end_thr; print(f"de_end_thr(完成阈)={thr:.3f}", flush=True)
    plt = setup_mpl()
    eps = [763, 2302, 2291]; fracs = [0.5, 0.7, 0.9, 1.0]; cols = {0.5: "#e45756", 0.7: "#f0a020", 0.9: "#4c78a8", 1.0: "#1a7f37"}
    fig, axes = plt.subplots(1, len(eps), figsize=(5.2 * len(eps), 4.6))
    for ci, e in enumerate(eps):
        a, r, s, n = loadep(TRIPLE, e); ax = axes[ci]
        for fr in fracs:
            k = max(5, int(n * fr)); at, rt, st = a[:k], r[:k], s[:k]
            v_on = smooth_monotone(dv_on.value(at, rt, st), fps=3.0); v_off = smooth_monotone(dv_off.value(at, rt, st), fps=3.0)
            Fq = fs.emb(at, rt, st); de_end = float(np.linalg.norm(Fq[-3:][:, None] - dv_on.endK[None], axis=2).min())  # 末帧到完成态残差
            flag = "完成" if de_end <= thr else "未完成(失败)"
            x = np.linspace(0, fr, k)
            ax.plot(x, v_on, color=cols[fr], lw=2, label=f"截{int(fr*100)}%: ON末={v_on[-2:].mean():.2f} OFF末={v_off[-2:].mean():.2f} | resid={de_end:.2f}→{flag}")
            ax.plot(x, v_off, color=cols[fr], lw=1, ls=":", alpha=0.7)
            print(f"ep{e} 截{int(fr*100)}%: ON末={v_on[-2:].mean():.2f} OFF末={v_off[-2:].mean():.2f} resid={de_end:.2f} thr={thr:.2f} → {flag}", flush=True)
        ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1, alpha=0.5, label="理想(value=进度)")
        ax.set_xlim(0, 1.02); ax.set_ylim(-.02, 1.02); ax.set_xlabel("截断进度"); ax.set_ylabel("value"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"ep{e} 截断(实线=cond_end ON, 点线=OFF)", fontsize=10)
    fig.suptitle("截断=失败rollout: value 应停在~截断进度(非1.0) + resid>thr 标出未完成. ON 末值应≤OFF(不被拉高)", fontsize=11)
    fig.tight_layout(); out = OUTV / "crave_truncate_test.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}", flush=True)


if __name__ == "__main__":
    main()
