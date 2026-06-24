"""三档 positive/normal/negative 分析 + "遥操专家数据 neg 应该多吗"。
CRAVE 离散阶梯天然三分: 推进(pos)/平台停留(normal)/回落(neg); AE 标量中段是噪声无结构。
数据: smooth800(806 专家base) + dagger(311 含策略尝试+纠错)。对照 base-only vs dagger-only 的 neg 率。
"""
import glob, json
import numpy as np, pandas as pd

from crave.config import REPO

# TODO(crave-lib): mv_value_full (.npy value cache) + *_awbc datasets are not in the
# dataset registry; kept as explicit paths.
MV = REPO / "temp/mv_value_full"; AW = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc"
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
csAW = json.load(open(AW / "meta/info.json"))["chunks_size"]; csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
W = 50; EPS = 0.02   # |adv|<EPS = normal(平台/无 milestone 变化)


def adv(v, w=W):
    a = np.zeros(len(v))
    for i in range(len(v)): a[i] = v[min(i + w, len(v) - 1)] - v[i]
    return np.clip(a, -1, 1)


# 判断每个 ep 是 base(专家) 还是 dagger: 用 episodes.jsonl 的 task 串或长度? 这里用 dagger 标志列
# A_smooth800_dagger_all 的 parquet 可能有标识; 退而用 episode_index 顺序(前 806 base, 后 dagger)近似
eps = sorted(int(p.stem[2:]) for p in MV.glob("ep*.npy"))
# 读 dagger 标识: A_smooth800_dagger_all 若有 is_dagger 列最好; 否则按已知 806 base 分界
try:
    pq0 = sorted(glob.glob(str(DS / "data/**/*.parquet"), recursive=True))[0]
    cols0 = pd.read_parquet(pq0).columns.tolist()
    dagger_col = next((c for c in cols0 if "dagger" in c.lower()), None)
except Exception:
    dagger_col = None
print(f"dagger 标识列: {dagger_col}", flush=True)

C_all, A_all, isdag = [], [], []
for e in eps:
    cv = np.load(MV / f"ep{e}.npy").astype(float)
    pqa = AW / "data" / f"chunk-{e//csAW:03d}" / f"episode_{e:06d}.parquet"
    if not pqa.exists(): continue
    aa = pd.read_parquet(pqa, columns=["absolute_advantage"])["absolute_advantage"].to_numpy().astype(float)
    n = min(len(cv), len(aa)); C_all.append(adv(cv[:n])); A_all.append(aa[:n])
    dg = 0
    if dagger_col:
        pqd = DS / "data" / f"chunk-{e//csDS:03d}" / f"episode_{e:06d}.parquet"
        if pqd.exists(): dg = int(pd.read_parquet(pqd, columns=[dagger_col])[dagger_col].to_numpy().astype(float).mean() > 0.5)
    isdag.append(np.full(n, dg))
C = np.concatenate(C_all); A = np.concatenate(A_all); D = np.concatenate(isdag) if dagger_col else np.zeros(len(C))


def three(a, eps=EPS):
    return float((a > eps).mean()), float((np.abs(a) <= eps).mean()), float((a < -eps).mean())


cp, cn, cneg = three(C); ap, an, aneg = three(A)
print(f"\n=== 三档(ε={EPS}) ===", flush=True)
print(f"CRAVE: positive {cp:.0%} / normal {cn:.0%} / negative {cneg:.0%}", flush=True)
print(f"AE   : positive {ap:.0%} / normal {an:.0%} / negative {aneg:.0%}", flush=True)
print(f"\n=== neg 率(遥操专家数据应少) ===", flush=True)
print(f"CRAVE neg(adv<0): 全 {(C<0).mean():.0%}", flush=True)
print(f"AE    neg(adv<0): 全 {(A<0).mean():.0%}", flush=True)
if dagger_col and D.sum() > 0 and (D == 0).sum() > 0:
    print(f"  base(专家) neg: CRAVE {(C[D==0]<0).mean():.0%} / AE {(A[D==0]<0).mean():.0%}", flush=True)
    print(f"  dagger(含失败) neg: CRAVE {(C[D==1]<0).mean():.0%} / AE {(A[D==1]<0).mean():.0%}", flush=True)
# AE 的 normal 中段是否有结构: 看 AE normal 帧的 CRAVE adv 分布(若 AE normal ≈ CRAVE 各档混杂 → AE 中段无结构)
ae_normal = np.abs(A) <= EPS
print(f"\nAE 'normal'({an:.0%}) 那些帧里, CRAVE 判: pos {three(C[ae_normal])[0]:.0%}/normal {three(C[ae_normal])[1]:.0%}/neg {three(C[ae_normal])[2]:.0%}", flush=True)
print(f"CRAVE 'normal'({cn:.0%}) 那些帧里, AE 判: pos {three(A[np.abs(C)<=EPS])[0]:.0%}/normal {three(A[np.abs(C)<=EPS])[1]:.0%}/neg {three(A[np.abs(C)<=EPS])[2]:.0%}", flush=True)
print("THREE_LEVEL_DONE", flush=True)
