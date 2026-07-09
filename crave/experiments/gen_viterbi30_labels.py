#!/usr/bin/env python
"""全量 3055 ep 的正式 CRAVE 标签:完整对称 Viterbi(无 smooth) @30Hz native · joint img⊕proprio。
读 temp/crave_30hz_feat_v2/ep{e}.npy (crop224, shard 同空间) + parquet proprio,
投到 joint milestones(temp/crave_joint_milestones.npz),dpHB λ=80 全局 Viterbi → 阶梯 value。
存 temp/crave_ae_labels/vit30_sym/ep{e}.npy(=A 标签;B=cummax 由 write 脚本派生)。
Run: PYTHONPATH=crave/src /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/gen_viterbi30_labels.py
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from crave.utils import mkp
from crave.config import resolve_dataset
from crave.data import kai0

REPO=Path("/home/tim/workspace/deepdive_kai0"); FEAT=REPO/"temp/crave_30hz_feat_v2"
OUT=REPO/"temp/crave_ae_labels/vit30_sym"; OUT.mkdir(parents=True,exist_ok=True)
z=np.load(REPO/"temp/crave_joint_milestones.npz"); C=z["C"]; Pord=z["Pord"]; PMU=z["PMU"]; PSD=z["PSD"]
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-8)
Ps=np.sort(Pord); Cs=C[np.argsort(Pord)]; bins=np.unique(np.concatenate([[0.0],Ps,[1.0]])); cb=[int(np.searchsorted(bins,p)) for p in Ps]
cfg=resolve_dataset("kai0_base"); cs=kai0.chunks_size(cfg.root); DS=Path(cfg.root)
def smkp(e): return mkp(np.stack(pd.read_parquet(DS/f"data/chunk-{e//cs:03d}/episode_{e:06d}.parquet",columns=["observation.state"])["observation.state"].to_numpy()))
def full_vit(F,lam=80.0):
    d=np.linalg.norm(F[:,None]-Cs[None],axis=2); n=len(F); em=np.full((n,len(bins)),1e3)
    for ci in range(len(Cs)): em[:,cb[ci]]=np.minimum(em[:,cb[ci]],d[:,ci])
    nb=len(bins); pen=lam*np.abs(bins[:,None]-bins[None]); cost=np.full(nb,1e9); cost[0]=em[0,0]; BP=np.zeros((n,nb),int)
    for j in range(1,n):
        tr=cost[None,:]+pen; k=tr.argmin(1); cost=em[j]+tr[np.arange(nb),k]; BP[j]=k
    cost[nb-1]-=2; s=int(cost.argmin()); path=np.zeros(n,int); path[-1]=s
    for j in range(n-2,-1,-1): s=BP[j+1][s]; path[j]=s
    return bins[path]
def main():
    eps=sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy"))
    print(f"{len(eps)} eps have 30Hz feat",flush=True); dd_all=[]
    for i,e in enumerate(eps):
        if (OUT/f"ep{e}.npy").exists(): continue
        img=l2(np.load(FEAT/f"ep{e}.npy").astype(np.float32)); pm=smkp(e); n=min(len(img),len(pm))
        F=np.concatenate([img[:n], l2((pm[:n]-PMU)/PSD)],1)
        v=full_vit(F).astype(np.float32); np.save(OUT/f"ep{e}.npy",v)
        dd_all.append(float((np.maximum.accumulate(v)-v).max()))
        if i%200==0: print(f"  {i}/{len(eps)} ep{e} n={n} v {v.min():.2f}->{v.max():.2f}",flush=True)
    if dd_all: print(f"done. re-grasp 回撤 mean={np.mean(dd_all):.3f} frac>0.05={np.mean(np.array(dd_all)>0.05):.2%}",flush=True)
if __name__=="__main__": main()
