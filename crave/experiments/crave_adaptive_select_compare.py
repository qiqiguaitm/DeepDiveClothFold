#!/usr/bin/env python
"""通过 crave.value 真跑对照: 固定(10bin·top2) vs 自适应(每bin·cov≥τ·封顶, τ数据驱动)。
先把 dagger 的分离 raw/armmask 缓存 + parquet state 打包成 DiscreteValue 期望的三元 npz(armmask/raw/state),
再用 crave.value.FeatureSpace + DiscreteValue(select=...) 构两套 milestone, 对照:
  #milestone / τ / 每ms最少帧 / 最大进度gap / ep808 value corr / 单调率 / 一批 held-out demo 的单调率&与时间corr。
输出: temp/crave_interp_ep808/adaptive_select_compare.png + 表。

Thin entrypoint over `crave`: REPO/paths from crave.config, FeatureSpace/DiscreteValue/loadep
from crave.value/crave.data, smooth_monotone from crave.utils, Agg+SimHei via crave.render.
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_adaptive_select_compare.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

from crave.config import REPO, resolve_dataset
from crave.data import loadep
from crave.render import setup_mpl
from crave.utils import smooth_monotone
from crave.value import DiscreteValue, FeatureSpace

plt = setup_mpl()

_cfg = resolve_dataset("smooth800_dagger")
DS = Path(_cfg.root)
ARM = Path(_cfg.arm_cache); RAW = Path(_cfg.raw_cache)
TRIPLE = REPO / "temp/_triple_dagger"; TRIPLE.mkdir(exist_ok=True)
OUT = REPO / "temp/crave_interp_ep808"; EP = 808
csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]


def build_triple(e):
    """打包 ep e 为 DiscreteValue 期望的 armmask/raw/state 三元 npz(3Hz 对齐)。"""
    out = TRIPLE / f"ep{e}.npz"
    if out.exists(): return
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]; n = min(len(a), len(r))
    pq = DS / f"data/chunk-{e//csDS:03d}/episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    np.savez(out, armmask=a[:n], raw=r[:n], state=st)


rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
perm = np.random.RandomState(0).permutation(all_eps).tolist()
mine = sorted(perm[:500]);
if EP not in mine: mine = sorted(mine + [EP])
test = sorted(perm[500:540])  # held-out demo(不参与挖矿)
print(f"打包三元缓存 {len(set(mine+test))} eps ...", flush=True)
for e in sorted(set(mine + test)): build_triple(e)

fs = FeatureSpace(TRIPLE, mine)
print("构建 milestone(两套) ...", flush=True)
dv_fix = DiscreteValue(fs, mine, k=96, select="fixed")
dv_ada = DiscreteValue(fs, mine, k=96, select="adaptive", nbins=10, cap_pb=3, tau_q=0.5)


def cont(dv, e):
    a, r, s, n = loadep(TRIPLE, e); v3 = dv.value(a, r, s)
    return smooth_monotone(np.repeat(v3, 10), fps=30.0)


def gap(P):
    return float(np.diff(np.concatenate([[0], np.sort(P), [1]])).max())


# 每 milestone 最近归属帧数(挖矿集)
G = fs.emb(*[np.concatenate(x) for x in zip(*[loadep(TRIPLE, e)[:3] for e in mine])])
def minframes(dv):
    nm = np.empty(len(G), int)
    for i in range(0, len(G), 20000): nm[i:i + 20000] = np.linalg.norm(G[i:i + 20000, None] - dv.C[None], axis=2).argmin(1)
    return int(np.bincount(nm, minlength=len(dv.order)).min())


vf, va = cont(dv_fix, EP), cont(dv_ada, EP)
# held-out 质量: 单调率 + 与归一化时间 corr(demo≈匀速进度)
def heldout(dv):
    mo, cr = [], []
    for e in test:
        v = cont(dv, e); t = np.arange(len(v)) / (len(v) - 1)
        mo.append(float(np.mean(np.diff(v) >= -1e-6))); cr.append(pearsonr(v, t)[0])
    return float(np.mean(mo)), float(np.mean(cr))


mo_f, cr_f = heldout(dv_fix); mo_a, cr_a = heldout(dv_ada)
print("\n指标                     固定(top2)     自适应(cov≥τ,封顶3)", flush=True)
print(f"#milestone               {len(dv_fix.order):<14}{len(dv_ada.order)}", flush=True)
print(f"τ(数据驱动)               {'—':<14}{dv_ada.tau:.2f}", flush=True)
print(f"每 ms 最少帧              {minframes(dv_fix):<14}{minframes(dv_ada)}", flush=True)
print(f"最大进度 gap             {gap(dv_fix.Pord):<14.2f}{gap(dv_ada.Pord):.2f}", flush=True)
print(f"ep808 单调率             {np.mean(np.diff(vf)>=-1e-6):<14.2%}{np.mean(np.diff(va)>=-1e-6):.2%}", flush=True)
print(f"ep808 corr(fix,ada)      {pearsonr(vf,va)[0]:.3f}", flush=True)
print(f"held-out({len(test)}) 单调率均    {mo_f:<14.2%}{mo_a:.2%}", flush=True)
print(f"held-out({len(test)}) corr时间均  {cr_f:<14.3f}{cr_a:.3f}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
t = np.arange(len(vf)) / 30
ax[0].plot(t, vf, color="#3b6fb0", lw=1.8, label=f"固定 top2 ({len(dv_fix.order)}档)")
ax[0].plot(t, va, color="#d7191c", lw=1.6, label=f"自适应 cov≥{dv_ada.tau:.2f} ({len(dv_ada.order)}档)")
ax[0].set_title(f"ep808 value: corr={pearsonr(vf,va)[0]:.2f}", fontsize=10); ax[0].set_xlabel("秒"); ax[0].set_ylabel("value"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
ax[1].plot(np.sort(dv_fix.Pord), range(len(dv_fix.order)), "o-", color="#3b6fb0", label="固定")
ax[1].plot(np.sort(dv_ada.Pord), range(len(dv_ada.order)), "s-", color="#d7191c", label="自适应")
ax[1].set_title("milestone 进度分布(自适应=按每档复现态数变长)", fontsize=10); ax[1].set_xlabel("进度 Pord"); ax[1].set_ylabel("第几个"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
xb = np.arange(2); w = .35
ax[2].bar(xb - w / 2, [mo_f * 100, cr_f * 100], w, label="固定", color="#3b6fb0")
ax[2].bar(xb + w / 2, [mo_a * 100, cr_a * 100], w, label="自适应", color="#d7191c")
ax[2].set_xticks(xb); ax[2].set_xticklabels([f"held-out\n单调率", f"held-out\ncorr时间"]); ax[2].set_ylim(0, 105)
ax[2].set_title(f"held-out {len(test)} demo 质量", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=.2, axis="y")
fig.suptitle("crave.value 真跑: 固定 top-2 vs 自适应(每bin·cov≥τ·封顶, τ数据驱动)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "adaptive_select_compare.png", dpi=120); plt.close(fig)
print("\nSAVED", OUT / "adaptive_select_compare.png", flush=True); print("ADAPTIVE_COMPARE_DONE", flush=True)
