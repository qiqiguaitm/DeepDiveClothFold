import json, numpy as np, pandas as pd, av, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
REPO=Path("/vePFS/tim/workspace/deepdive_kai0");DS=REPO/"kai0/data/Task_A/vis_base/v3/2026-05-26-v3"
ARM=REPO/"temp/tcc_vis0526_armmask/feat_cache";RAW=REPO/"temp/tcc_vis0526_raw/feat_cache"
cs=json.load(open(DS/"meta/info.json")).get("chunks_size",1000);TEST=7
def lpst(e,n):
    pq=DS/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
all_eps=sorted(int(p.stem[2:]) for p in ARM.glob("ep*.npz"));mined=[e for e in all_eps if e!=TEST]
def loadep(e):
    a=np.load(ARM/f"ep{e}.npz")["f"];r=np.load(RAW/f"ep{e}.npz")["f"];n=min(len(a),len(r));return a[:n],r[:n],lpst(e,n),n
def mkp(s):return np.concatenate([s,np.vstack([np.zeros((1,s.shape[1])),np.diff(s,axis=0)])],1)
Sall=[loadep(e)[2] for e in mined];Pm=mkp(np.concatenate(Sall));PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a,r,st):
    an=a/np.linalg.norm(a,axis=1,keepdims=True);rn=r/np.linalg.norm(r,axis=1,keepdims=True)
    Pn=((mkp(st)-PMU)/PSD);Pn/=np.linalg.norm(Pn,axis=1,keepdims=True);return np.concatenate([rn,an,Pn],1)
A,R,S,T,E,FR=[],[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep(e);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e));FR.append(np.arange(n)*10)
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E);FR=np.concatenate(FR);G=emb(A,R,S)
km=KMeans(96,n_init=2,random_state=0).fit(G);lab=km.labels_
n_ep=len(set(E.tolist()));cov=np.array([len(set(E[lab==c].tolist()))/n_ep for c in range(96)])
def gr(idx):
    o=[];s=None;pv=None
    for i in idx:
        if pv is None or i!=pv+1:
            if s is not None:o.append((s,pv))
            s=i
        pv=i
    if s is not None:o.append((s,pv))
    return [x for x in o if x[1]-x[0]>=1]
msC=np.argsort(cov)[-20:].tolist();Pk={}
for c in msC:
    fe=[]
    for e in sorted(set(E.tolist())):
        m=np.where(E==e)[0];rs=gr(m[lab[m]==c].tolist())
        if rs:fe.append(T[rs[0][0]])
    Pk[c]=float(np.median(fe)) if fe else .5
# ep7 三路分段匹配 (raw[:384] / arm[384:768] / prop[768:]) — 验证视觉vs本体分歧
order=sorted(msC,key=lambda c:Pk[c]);Cm=km.cluster_centers_[order];Pord=np.array([Pk[c] for c in order])
a7,r7,st7,n7=loadep(TEST);F7=emb(a7,r7,st7)
def segm(lo,hi):
    d=np.linalg.norm(F7[:,None,lo:hi]-Cm[None,:,lo:hi],axis=2);return d.argmin(1)
mraw=segm(0,384);marm=segm(384,768);mprop=segm(768,F7.shape[1])
print("ep7开头 三路各自匹配 milestone 的 P (视觉vs本体分歧=别名信号):")
ndiv=0
for t in range(min(35,n7)):
    pr,pa,pp=Pord[mraw[t]],Pord[mprop[t]],Pord[mprop[t]]
    praw,parm,pp=Pord[mraw[t]],Pord[marm[t]],Pord[mprop[t]]
    vis=max(praw,parm);spread=vis-pp
    div="  <<< 视觉高/本体低=别名" if spread>0.4 else ""
    if spread>0.4:ndiv+=1
    print(f"  t={t/3:.1f}s  raw P{praw:.2f}  arm P{parm:.2f}  prop P{pp:.2f}  (视觉-本体={spread:+.2f}){div}")
print(f"\n开头35帧中 {ndiv} 帧 视觉-本体分歧>0.4 (=多路分歧可检测的别名误判)")
print("DONE_MP")
