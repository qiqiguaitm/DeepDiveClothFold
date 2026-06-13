import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
REPO=Path("/vePFS/tim/workspace/deepdive_kai0")
DS=REPO/"kai0/data/Task_A/self_built/A_new_smooth_800/base"
ARM=REPO/"temp/tcc_smooth800_armmask/feat_cache";RAW=REPO/"temp/tcc_smooth800_raw/feat_cache"
cs=json.load(open(DS/"meta/info.json")).get("chunks_size",1000);TEST=0
def lpst(e,n):
    pq=DS/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
rawset=set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
all_eps=sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset and e!=TEST)
mined=sorted(np.random.RandomState(0).permutation(all_eps)[:500].tolist())
print(f"raw提取 {len(rawset)} ep; 挖掘 {len(mined)} ep (ep0 held-out)")
def loadep(e):
    a=np.load(ARM/f"ep{e}.npz")["f"];r=np.load(RAW/f"ep{e}.npz")["f"];n=min(len(a),len(r));return a[:n],r[:n],lpst(e,n),n
def mkp(s):return np.concatenate([s,np.vstack([np.zeros((1,s.shape[1])),np.diff(s,axis=0)])],1)
Sall=[loadep(e)[2] for e in mined];Pm=mkp(np.concatenate(Sall));PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a,r,st):
    an=a/np.linalg.norm(a,axis=1,keepdims=True);rn=r/np.linalg.norm(r,axis=1,keepdims=True)
    Pn=((mkp(st)-PMU)/PSD);Pn/=np.linalg.norm(Pn,axis=1,keepdims=True);return np.concatenate([rn,an,Pn],1)
A,R,S,T,E,FR,SP,EP=[],[],[],[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep(e);g=emb(a,r,st);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e));FR.append(np.arange(n)*10);SP.append(g[:2]);EP.append(g[-2:])
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E);FR=np.concatenate(FR);G=emb(A,R,S)
km=KMeans(96,n_init=2,random_state=0).fit(G);lab=km.labels_;allC=km.cluster_centers_
N=len(set(E.tolist()));tpos=np.array([T[lab==c].mean() if (lab==c).any() else .5 for c in range(96)])
Pstart={}
for e in sorted(set(E.tolist())):
    m=np.where(E==e)[0][:3];nn=np.linalg.norm(G[m][:,None]-allC[None],axis=2).argmin(1);Pstart[e]=float(np.median(tpos[nn]))
cov_n=np.array([min(1,(len(set(E[lab==c].tolist()))+sum(1 for e in Pstart if Pstart[e]>tpos[c]+0.1))/N) for c in range(96)])
bk=np.linspace(0,1,11);sel=[]
for b in range(10):
    inb=[c for c in range(96) if bk[b]<=tpos[c]<bk[b+1]]
    if inb:sel+=sorted(inb,key=lambda c:-cov_n[c])[:2]
sel=sorted(set(sel),key=lambda c:tpos[c])
def gr(idx):
    o=[];s=None;pv=None
    for i in idx:
        if pv is None or i!=pv+1:
            if s is not None:o.append((s,pv))
            s=i
        pv=i
    if s is not None:o.append((s,pv))
    return [x for x in o if x[1]-x[0]>=1]
Pk={}
for c in sel:
    fe=[]
    for e in sorted(set(E.tolist())):
        m=np.where(E==e)[0];rs=gr(m[lab[m]==c].tolist())
        if rs:fe.append(T[rs[0][0]])
    Pk[c]=float(np.median(fe)) if fe else float(tpos[c])
order=sorted(sel,key=lambda c:Pk[c]);C=allC[order];Pord=np.array([Pk[c] for c in order])
print("V2.4 milestones:",len(order),"前段(P<0.5):",sum(1 for c in order if Pk[c]<0.5))
startK=KMeans(8,n_init=2,random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK=KMeans(8,n_init=2,random_state=0).fit(np.concatenate(EP)).cluster_centers_
NB=21;bins=np.linspace(0,1,NB);cb=[[int(np.argmin(abs(bins-Pk[c])))] for c in order]
def dpHB(emit,lam=8.0):
    pen=lam*np.abs(bins[:,None]-bins[None]);NF=len(emit);cost=np.full(NB,1e9);cost[0]=emit[0,0];bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen;k=tr.argmin(1);cost=emit[j]+tr[np.arange(NB),k];bp[j]=k
    cost[NB-1]-=2;path=np.zeros(NF,int);path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1):path[j]=bp[j+1,path[j+1]]
    return bins[path]
def med(a,w):
    h=w//2;return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
def value(a,r,st):
    Fq=emb(a,r,st);nq=len(Fq);d=np.linalg.norm(Fq[:,None]-C[None],axis=2);em=np.full((nq,NB),1e3)
    for ci in range(len(order)):
        for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d[:,ci])
    ds=np.linalg.norm(Fq[:,None]-startK[None],axis=2).min(1);de=np.linalg.norm(Fq[:,None]-endK[None],axis=2).min(1)
    tn=np.arange(nq)/nq;em[:,0]=np.minimum(em[:,0],np.where(tn<0.3,ds,ds+(tn-0.3)*6));em[:,NB-1]=np.minimum(em[:,NB-1],np.where(tn>0.6,de,de+(0.6-tn)*6))
    return med(dpHB(em),9)
a0,r0,st0,n0=loadep(TEST);v0=value(a0,r0,st0)
NF=len(pd.read_parquet(DS/"data/chunk-000/episode_000000.parquet"))
v645=np.repeat(v0,10)[:NF]
if len(v645)<NF:v645=np.concatenate([v645,np.full(NF-len(v645),v645[-1])])
np.save("/vePFS/tim/viva_work/ep0_value_v24.npy",v645)
print(f"ep0 V2.4 value: {n0}帧→{NF}帧, range[{v0.min():.2f},{v0.max():.2f}] 开头max={v0[:3].max():.2f} 末={v0[-3:].mean():.2f}")
print("SAVED ep0_value_v24.npy")
