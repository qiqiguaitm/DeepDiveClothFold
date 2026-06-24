"""B2 验证(零GPU训练): 终点可达性 + OOD残差 → 弱失败信号, 补 CRAVE 缺的 neg。
① 终点可达性 = CRAVE 末值(mv_value_full, 1117 dagger ep) → 低=未完成=失败候选。
② OOD残差 = 帧到最近 milestone 距离(挖 smooth800_dagger 模型) → 高=离 demo 流形=异常。
验证: 这两个信号是否抓到 AE 标 neg 的帧/ep(即 CRAVE-progress 漏掉的真失败)。
文献: 失败检测=OOD偏离训练流形(arXiv:2503.08558 / 2509.26308)。

Thin entrypoint over `crave`: paths/dataset from crave.config + crave.data.kai0;
mkp from crave.utils; mpl from crave.render. (mv_value_full / awbc task-index dataset
are not in the dataset registry, so their paths are built from REPO.)
"""
import json
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

from crave.config import REPO, resolve_dataset, viz_dir
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import mkp

plt = setup_mpl()

MV = REPO / "temp/mv_value_full"; AW = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc"
CFG = resolve_dataset("smooth800_dagger")
DS = Path(CFG.root)
ARM = Path(CFG.arm_cache); RAW = Path(CFG.raw_cache)
csAW = kai0.chunks_size(str(AW))


# ===== ① 终点可达性(瞬时, 无挖矿) =====
eps = sorted(int(p.stem[2:]) for p in MV.glob("ep*.npy"))
finalv, aeneg = [], []
for e in eps:
    v = np.load(MV / f"ep{e}.npy").astype(float)
    pq = AW / "data" / f"chunk-{e//csAW:03d}" / f"episode_{e:06d}.parquet"
    if not pq.exists(): continue
    ti = pd.read_parquet(pq, columns=["task_index"])["task_index"].to_numpy()
    finalv.append(float(np.mean(v[-30:]))); aeneg.append(float((ti == 0).mean()))
finalv = np.array(finalv); aeneg = np.array(aeneg)
incomplete = finalv < 0.7
r_term = pearsonr(1 - finalv, aeneg)[0]   # CRAVE 未完成度 vs AE neg 率
print(f"① 终点可达性: {len(finalv)}ep, 末值<0.7(未完成){incomplete.mean():.0%}; "
      f"corr(1-末值, AE-neg率)={r_term:.3f}; 未完成ep的AE-neg率{aeneg[incomplete].mean():.0%} vs 完成ep{aeneg[~incomplete].mean():.0%}", flush=True)


# ===== ② OOD 残差(挖 smooth800_dagger 模型) =====
def loadep(e):
    return kai0.loadep_tcc(CFG, e)


rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
mined = sorted(np.random.RandomState(0).permutation(all_eps)[:400].tolist())
Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True); return np.concatenate([rn, an, Pn], 1)


G = np.concatenate([emb(*loadep(e)[:3]) for e in mined])
km = KMeans(96, n_init=2, random_state=0).fit(G); C = km.cluster_centers_
# 训练集残差分布(标定 OOD 阈值)
train_res = np.linalg.norm(G[:, None] - C[None], axis=2).min(1)
thr = np.quantile(train_res, 0.95)   # 95 分位 = OOD 阈
print(f"② OOD残差: 挖 {len(mined)}ep, 训练残差中位{np.median(train_res):.3f} 95分位(阈){thr:.3f}", flush=True)

# 帧级: 残差 vs AE-neg(残差高是否对应 AE 标 neg)
res_neg, res_pos = [], []
samp = sorted(np.random.RandomState(1).permutation(all_eps)[:120].tolist())
hi_res_in_aeneg = []; aeneg_frac_all = []
for e in samp:
    aa, rr, st, n = loadep(e); res = np.linalg.norm(emb(aa, rr, st)[:, None] - C[None], axis=2).min(1)
    pq = AW / "data" / f"chunk-{e//csAW:03d}" / f"episode_{e:06d}.parquet"
    if not pq.exists(): continue
    ti = pd.read_parquet(pq, columns=["task_index"])["task_index"].to_numpy()
    ti3 = ti[np.minimum(np.arange(n) * 10, len(ti) - 1)]   # 3Hz 对齐
    res_neg.append(res[ti3 == 0]); res_pos.append(res[ti3 == 1])
rn_ = np.concatenate(res_neg); rp_ = np.concatenate(res_pos)
print(f"   AE标neg帧的残差均值{rn_.mean():.3f} vs AE标pos帧{rp_.mean():.3f} "
      f"({'neg残差更高→残差能区分✓' if rn_.mean()>rp_.mean() else '无区分'}); "
      f"高残差(>阈)帧里AE-neg占比{(np.concatenate([np.where(np.concatenate([r for r in res_neg]) > thr,1,0)]).mean() if res_neg else 0):.0%}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(1 - finalv, aeneg, s=8, alpha=.4, color="#2ca02c"); ax[0].set_xlabel("CRAVE 未完成度 (1-末值)"); ax[0].set_ylabel("AE neg 率")
ax[0].set_title(f"① 终点可达性 vs AE-neg (corr={r_term:.2f})", fontsize=10); ax[0].grid(alpha=.2)
ax[1].hist(finalv, bins=30, color="#1f77ff", alpha=.8); ax[1].axvline(0.7, color="r", ls="--"); ax[1].set_xlabel("CRAVE 末值(终点可达)"); ax[1].set_title(f"末值分布: 未完成(<0.7) {incomplete.mean():.0%}", fontsize=10); ax[1].grid(alpha=.2)
ax[2].hist(rp_, bins=40, alpha=.5, color="#2ca02c", density=True, label=f"AE-pos 帧 残差(均{rp_.mean():.2f})")
ax[2].hist(rn_, bins=40, alpha=.5, color="#d62728", density=True, label=f"AE-neg 帧 残差(均{rn_.mean():.2f})")
ax[2].axvline(thr, color="k", ls="--", lw=1, label=f"OOD阈{thr:.2f}"); ax[2].set_xlabel("OOD 残差(到最近 milestone)"); ax[2].set_title("② 残差能否区分 AE-neg", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=.2)
fig.suptitle("B2 验证: 终点可达性 + OOD残差 → 弱失败信号(零GPU训练, 补 CRAVE 的 neg)", fontsize=12)
fig.tight_layout(); out = viz_dir() / "crave_b2_failure_signal.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); print("B2_DONE", flush=True)
