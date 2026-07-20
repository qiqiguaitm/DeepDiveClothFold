#!/usr/bin/env python
"""对齐特征上的完整 UR-VC 分析(修 dino_sub20 错对齐后的定稿版).
A) UR-VC 基线 vs 时间标签 vs (参照 CRAVE v1 0.943): corr/mono/end/覆盖率, rho 三档
B) 时序之战 S1 变速 / S2 长回退: τ=0.3 vs 无带
用法: python urvc_aligned_analysis.py dino|siglip2
"""
import glob, os, sys
import numpy as np, pandas as pd

ENC = sys.argv[1]
FEAT = f"/vePFS/tim/tmp/claude-1000/-vePFS-tim-workspace-deepdive-kai0/52e86d52-cd8c-4dfd-9952-1594aae894a2/scratchpad/kai0_aligned/{ENC}"
GT = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage/data/chunk-000"
TAU = 0.30

featL, gtL = [], []
for f in sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4])):
    e = int(os.path.basename(f)[2:-4])
    d = np.load(f); p, fidx = d["pooled"].astype(np.float32), d["fidx"]
    p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    df = pd.read_parquet(f"{GT}/episode_{e:06d}.parquet", columns=["stage_progress_gt"])
    assert fidx[-1] < len(df) + 20, f"ep{e}: fidx {fidx[-1]} vs parquet {len(df)}"
    featL.append(p); gtL.append(df["stage_progress_gt"].values[np.minimum(fidx, len(df)-1)].astype(np.float32))
ne = len(featL)
print(f"[{ENC}] eps={ne} frames={sum(len(p) for p in featL)}", flush=True)

def urvc(F, W, tau, rho):
    preds, covs = [], []
    for a in range(ne):
        acc = np.zeros(len(F[a])); cnt = np.zeros(len(F[a]), np.int32)
        for b in range(ne):
            if a == b: continue
            C = F[a] @ F[b].T
            C = np.where(np.abs(W[a][:, None] - W[b][None, :]) <= tau, C, -2.0)
            j = C.argmax(1); c = C[np.arange(len(j)), j]
            ok = c >= (rho if rho is not None else -1.5)
            acc[ok] += W[b][j[ok]]; cnt[ok] += 1
        preds.append(np.where(cnt > 0, acc/np.maximum(cnt, 1), W[a]).astype(np.float32))
        covs.append((cnt > 0).mean())
    return preds, np.mean(covs)

def metrics(pred, gt):
    cs = [np.corrcoef(p, t)[0,1] for p,t in zip(pred,gt) if np.std(p)>1e-6 and np.std(t)>1e-6]
    return np.median(cs), np.mean([(np.diff(p)>=-1e-6).mean() for p in pred]), np.mean([p[-1] for p in pred])

# ===== A) 基线 =====
W0 = [np.arange(len(p), dtype=np.float32)/max(len(p)-1,1) for p in featL]
print(f"\n===== A) UR-VC 基线({ENC}, 对齐特征) =====")
print(f"{'方法':<22} {'corr(med)':>9} {'mono':>6} {'end':>6} {'覆盖率':>7}")
c,m,e = metrics(W0, gtL); print(f"{'时间标签 g':<22} {c:>9.3f} {m:>6.3f} {e:>6.3f} {'—':>7}")
for rho in [None, 0.90, 0.95]:
    P, cov = urvc(featL, W0, TAU, rho)
    c,m,e = metrics(P, gtL)
    print(f"{'UR-VC rho='+str(rho if rho is not None else '无'):<22} {c:>9.3f} {m:>6.3f} {e:>6.3f} {cov:>6.1%}")
print("参照: CRAVE v1 收口 corr 0.943 / mono 0.981 / end 0.999(3055ep 30Hz 口径)")

# ===== B) 时序之战 =====
def build(scn):
    F,T,W,MOD,RIDX = [],[],[],[],[]
    for i,(f,t) in enumerate(zip(featL, gtL)):
        n = len(f)
        if i%2==1 and scn=="S1":
            k=int(n*0.35); r=np.concatenate([np.repeat(np.arange(k),5), np.arange(k,n)]); mod=True
        elif i%2==1 and scn=="S2":
            a,b,c2=int(n*0.05),int(n*0.45),int(n*0.75)
            r=np.concatenate([np.arange(c2), np.arange(a,b), np.arange(c2,n)]); mod=True
        else:
            r=np.arange(n); mod=False
        F.append(f[r]); T.append(t[r]); W.append(np.arange(len(r),dtype=np.float32)/max(len(r)-1,1))
        MOD.append(mod); RIDX.append(r)
    return F,T,W,MOD,RIDX

for scn, name in [("S1","S1 变速(前35%慢5x)"),("S2","S2 长回退(75%处重做40%)")]:
    F,T,W,MOD,RIDX = build(scn)
    global featL_saved; featL_saved = featL
    ne_l = ne
    rows = {"时间标签(wall)": W}
    for nm, tau in [("UR-VC τ=0.3", TAU), ("无带检索 τ=∞", 10.0)]:
        preds = []
        for a in range(ne):
            acc=np.zeros(len(F[a])); cnt=np.zeros(len(F[a]),np.int32)
            for b in range(ne):
                if a==b: continue
                C = F[a] @ F[b].T
                C = np.where(np.abs(W[a][:,None]-W[b][None,:])<=tau, C, -2.0)
                j = C.argmax(1); ok = C[np.arange(len(j)), j] > -1.5
                acc[ok] += W[b][j[ok]]; cnt[ok] += 1
            preds.append(np.where(cnt>0, acc/np.maximum(cnt,1), W[a]).astype(np.float32))
        rows[nm] = preds
    print(f"\n===== B) {name} | 被改ep={sum(MOD)}/{ne} =====")
    hdr = f"{'方法':<16} {'corr被改ep':>10}" + ("  回退捕获率" if scn=="S2" else "")
    print(hdr)
    for nm, P in rows.items():
        cs = [np.corrcoef(P[i],T[i])[0,1] for i in range(ne) if MOD[i] and np.std(T[i])>1e-6]
        extra = ""
        if scn=="S2":
            caps=[]
            for i in range(ne):
                if not MOD[i]: continue
                n0=len(featL[i]); a,b,c2=int(n0*0.05),int(n0*0.45),int(n0*0.75)
                r=np.arange(c2, c2+(b-a))
                caps.append((P[i][r] < P[i][c2-1]-0.05).mean())
            extra=f"  {np.mean(caps):>9.1%}"
        print(f"{nm:<16} {np.median(cs):>10.3f}"+extra)
