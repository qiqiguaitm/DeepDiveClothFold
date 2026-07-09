#!/usr/bin/env python
"""编解码器依赖分析:CRAVE value 在多大程度依赖具体编码器?能否用 encoder-free proprio 逃逸?
在同帧对齐的缓存特征上(dinov2-large / dinov3-h / proprio-only / dinov3h+proprio)建 milestone + 全 Viterbi value,
比较 (a) value 跨编码器一致性(corr) 与 (b) 分区一致性(ARI) —— 定位依赖所在 + 展示逃逸路线。
Run: PYTHONPATH=crave/src:crave/experiments CUDA_VISIBLE_DEVICES=1 python crave/experiments/encoder_escape_analysis.py
"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"crave/experiments")
from crave.render import setup_mpl
from crave.utils import mkp
from crave.config import resolve_dataset
from crave.data import kai0
from milestone_select import build_milestones_uniform
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score
plt=setup_mpl()
REPO=Path("/home/tim/workspace/deepdive_kai0")
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-8)
def load_enc(idxp, sharddir, dim):
    zf=np.load(idxp); E,FR,T=zf["E"],zf["FR"],zf["T"]; N=int(zf["n"])
    feat=np.zeros((N,dim),np.float16); valid=np.zeros(N,bool)
    for f in sorted(glob.glob(str(sharddir/"shard_*.npz"))):
        z=np.load(f); feat[z["gidx"]]=z["feat"]; valid[z["gidx"]]=z["valid"]
    return E,FR,T,feat,valid
E,FR,T,d2,val2=load_enc(REPO/"temp/crave_full/index_dino.npz",REPO/"temp/crave_full/dino",1024)
_,_,_,d3,val3=load_enc(REPO/"temp/crave_full_dinov3h/index.npz",REPO/"temp/crave_full_dinov3h",1280)
vi=np.where(val2&val3)[0]; Ev,FRv,Tv=E[vi],FR[vi],T[vi]; ne=len(set(Ev.tolist()))
cfg=resolve_dataset("kai0_base"); cs=kai0.chunks_size(cfg.root); DS=Path(cfg.root)
# proprio 3Hz
P=np.zeros((len(vi),28),np.float32); ep_list=sorted(set(Ev.tolist()))
for e in ep_list:
    loc=np.where(Ev==e)[0]; st=np.stack(pd.read_parquet(DS/f"data/chunk-{e//cs:03d}/episode_{e:06d}.parquet",columns=["observation.state"])["observation.state"].to_numpy())
    pm=mkp(st); P[loc]=pm[np.minimum(FRv[loc],len(pm)-1)]
PMU,PSD=P.mean(0),P.std(0)+1e-8; Pn=l2((P-PMU)/PSD)
img2=l2(d2[vi].astype(np.float32)); img3=l2(d3[vi].astype(np.float32))
variants={
 "dinov2L+prop": np.concatenate([img2,Pn],1),
 "dinov3H+prop": np.concatenate([img3,Pn],1),
 "proprio-only": Pn,
 "dinov3H-only": img3,
}
print(f"{len(vi)} frames, {ne} eps. building milestones per variant...",flush=True)
MS={}
for name,F in variants.items():
    cen,lab,order,Pord,M,nf=build_milestones_uniform(F,Ev,Tv,ne); MS[name]=(cen[order].astype(np.float32),np.asarray(Pord,float),M)
    print(f"  {name}: M={M}",flush=True)
def full_vit(Fq,C,Pord,lam=80.0):
    Ps=np.sort(Pord); Cs=C[np.argsort(Pord)]; bins=np.unique(np.concatenate([[0.0],Ps,[1.0]])); cb=[int(np.searchsorted(bins,p)) for p in Ps]
    d=np.linalg.norm(Fq[:,None]-Cs[None],axis=2); n=len(Fq); em=np.full((n,len(bins)),1e3)
    for ci in range(len(Cs)): em[:,cb[ci]]=np.minimum(em[:,cb[ci]],d[:,ci])
    nb=len(bins); pen=lam*np.abs(bins[:,None]-bins[None]); cost=np.full(nb,1e9); cost[0]=em[0,0]; BP=np.zeros((n,nb),int)
    for j in range(1,n):
        tr=cost[None,:]+pen; k=tr.argmin(1); cost=em[j]+tr[np.arange(nb),k]; BP[j]=k
    cost[nb-1]-=2; s=int(cost.argmin()); path=np.zeros(n,int); path[-1]=s
    for j in range(n-2,-1,-1): s=BP[j+1][s]; path[j]=s
    return bins[path]
def nearest_ms(Fq,C,Pord):
    Cs=C[np.argsort(Pord)]; return np.linalg.norm(Fq[:,None]-Cs[None],axis=2).argmin(1)
rng=np.random.RandomState(0); samp=rng.choice(ep_list,150,replace=False)
names=list(variants); vals={n:{} for n in names}; labs={n:[] for n in names}
Fbyv={n:variants[n] for n in names}
for e in samp:
    loc=np.where(Ev==e)[0]; o=np.argsort(FRv[loc]); loc=loc[o]
    for n in names:
        Fq=Fbyv[n][loc]; vals[n][e]=full_vit(Fq,*MS[n][:2]); labs[n].append(nearest_ms(Fq,*MS[n][:2]))
# value corr matrix (mean per-ep pairwise pearson) + partition ARI matrix
K=len(names); VC=np.eye(K); AR=np.eye(K)
for i in range(K):
    for j in range(i+1,K):
        cs=[pearsonr(vals[names[i]][e],vals[names[j]][e])[0] for e in samp if vals[names[i]][e].std()>1e-6 and vals[names[j]][e].std()>1e-6]
        VC[i,j]=VC[j,i]=np.nanmean(cs)
        AR[i,j]=AR[j,i]=adjusted_rand_score(np.concatenate(labs[names[i]]),np.concatenate(labs[names[j]]))
print("\n=== VALUE corr (mean per-ep pearson) ===",flush=True)
print("        "+" ".join(f"{n[:9]:>9}" for n in names),flush=True)
for i,n in enumerate(names): print(f"{n[:9]:>9} "+" ".join(f"{VC[i,j]:9.3f}" for j in range(K)),flush=True)
print("\n=== PARTITION ARI (per-frame milestone labels) ===",flush=True)
print("        "+" ".join(f"{n[:9]:>9}" for n in names),flush=True)
for i,n in enumerate(names): print(f"{n[:9]:>9} "+" ".join(f"{AR[i,j]:9.3f}" for j in range(K)),flush=True)
np.savez(REPO/"temp/encoder_escape.npz",VC=VC,AR=AR,names=names)
# fig
fig,ax=plt.subplots(1,2,figsize=(15,6.2))
for a,(M,ttl,fmt) in zip(ax,[(VC,"VALUE 跨编码器一致 (per-ep corr)","%.3f"),(AR,"PARTITION 一致 (ARI)","%.3f")]):
    im=a.imshow(M,vmin=0,vmax=1,cmap="viridis"); a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xticklabels(names,rotation=30,ha="right",fontsize=9); a.set_yticklabels(names,fontsize=9)
    for i in range(K):
        for j in range(K): a.text(j,i,fmt%M[i,j],ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
    a.set_title(ttl,fontsize=11); fig.colorbar(im,ax=a,fraction=.046)
fig.suptitle("CRAVE 编码器依赖定位:value 几乎不依赖编码器(含 proprio-only 逃逸) vs 分区强依赖",fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); out="crave/docs/visualization/encoders/encoder_escape_value_vs_partition.png"; Path(out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=120); print("SAVED",out,flush=True)
