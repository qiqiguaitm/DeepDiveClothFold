"""验证条件化 end-anchor: ① 成功 ep 仍到 1.0(不伤);② 合成回退 ep 末段不再被强拉高(cond_end 开 vs 关)。
跑法: kai0/.venv/bin/python train_scripts/kai/data/crave_condend_validate.py
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_value import FeatureSpace, DiscreteValue, loadep
from crave_readout import smooth_monotone
REPO = Path("/vePFS/tim/workspace/deepdive_kai0"); DS = REPO / "kai0/data/Task_A/kai0_base"
RAW = REPO / "temp/tcc_kai0_raw/feat_cache"; ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"; TRIPLE = REPO / "temp/_triple_prodtest"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"


def lpst(e, n):
    st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def bt(e):
    o = TRIPLE / f"ep{e}.npz"
    if o.exists(): return True
    try: a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    except Exception: return False
    n = min(len(a), len(r)); np.savez(o, armmask=a[:n], raw=r[:n], state=lpst(e, n)); return True


def main():
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    mine = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mine = [e for e in mine if bt(e)]
    fs = FeatureSpace(TRIPLE, mine)
    dv_on = DiscreteValue(fs, mine, k=96, select="fixed", order="precedence", cond_end=True, log=lambda *_: None)
    dv_off = DiscreteValue(fs, mine, k=96, select="fixed", order="precedence", cond_end=False, log=lambda *_: None)
    print(f"de_end_thr={dv_on.de_end_thr:.3f}", flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, e in enumerate([763, 2302]):
        a, r, s, n = loadep(TRIPLE, e)
        # 成功(正常)
        v_on = smooth_monotone(dv_on.value(a, r, s), fps=3.0); v_off = smooth_monotone(dv_off.value(a, r, s), fps=3.0)
        ax = axes[0, col]; ax.plot(v_off, color="#999", lw=2, label="cond_end OFF (旧)"); ax.plot(v_on, color="#1a7f37", lw=2, label="cond_end ON")
        ax.set_title(f"ep{e} 成功: ON max={v_on.max():.2f} last={v_on[-3:].mean():.2f} | OFF last={v_off[-3:].mean():.2f}", fontsize=9)
        ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
        print(f"ep{e} 成功: ON max={v_on.max():.2f} last={v_on[-3:].mean():.2f} | OFF max={v_off.max():.2f} last={v_off[-3:].mean():.2f}", flush=True)
        # 合成回退: 前进到完成(100%, value→1.0) 再倒放回 20%(完成后又散开)
        m6 = n; m2 = int(n * 0.2); traj = list(range(0, n)) + list(range(n - 1, m2, -1))
        ar, rr, sr = a[traj], r[traj], s[traj]
        vr_on = smooth_monotone(dv_on.value(ar, rr, sr), fps=3.0); vr_off = smooth_monotone(dv_off.value(ar, rr, sr), fps=3.0)
        ax = axes[1, col]; tp = np.array([t / max(1, n - 1) for t in traj])
        ax.plot(tp, color="k", ls="--", lw=1.3, label="true progress")
        ax.plot(vr_off, color="#999", lw=2, label="cond_end OFF (旧)"); ax.plot(vr_on, color="#9c27b0", lw=2, label="cond_end ON")
        ax.axvline(m6 - 0.5, color="0.6"); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(f"ep{e} 回退: ON end={vr_on[-3:].mean():.2f} | OFF end={vr_off[-3:].mean():.2f}  (越低=越不被拉高)", fontsize=9)
        print(f"ep{e} 回退: ON peak={vr_on[:m6].max():.2f} end={vr_on[-3:].mean():.2f} | OFF peak={vr_off[:m6].max():.2f} end={vr_off[-3:].mean():.2f}", flush=True)
    fig.suptitle("条件化 end-anchor: 上=成功(应仍到1.0) 下=合成回退(ON 末段应不被拉高)", fontsize=12)
    fig.tight_layout(); out = OUTV / "crave_condend_validate.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}", flush=True)


if __name__ == "__main__":
    main()
