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
A,R,S,T,E,FR,SP,EP=[],[],[],[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep(e);g=emb(a,r,st);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e));FR.append(np.arange(n)*10);SP.append(g[:2]);EP.append(g[-2:])
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E);FR=np.concatenate(FR);G=emb(A,R,S)
km=KMeans(96,n_init=2,random_state=0).fit(G);lab=km.labels_;allC=km.cluster_centers_
N=len(set(E.tolist()));cov=np.array([len(set(E[lab==c].tolist()))/N for c in range(96)])
tpos=np.array([T[lab==c].mean() if (lab==c).any() else .5 for c in range(96)])
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
print("V2.4 milestones:",[(f"c{c}",round(Pk[c],2),f"{cov_n[c]:.0%}") for c in order])
print(f"前段(P<0.5): {sum(1 for c in order if Pk[c]<0.5)}/{len(order)}")
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
a7,r7,st7,n7=loadep(TEST);v7=value(a7,r7,st7)
print(f"ep7 V2.4: 开头10s(前30帧)max={v7[:30].max():.2f} 末段={v7[-10:].mean():.2f} 全程[{v7.min():.2f},{v7.max():.2f}]")
# 前段误判通病: 测5个其他ep开头10s max
import random
others=random.Random(1).sample([e for e in mined],5)
print("其他5ep 开头10s value max (越低越好,抓取段不该高):")
for e in others:
    aa,rr,ss,nn=loadep(e);vv=value(aa,rr,ss);print(f"  ep{e}: 开头max={vv[:30].max():.2f}")
def camp(e):return DS/"videos"/f"chunk-{e//cs:03d}"/"observation.images.top_head"/f"episode_{e:06d}.mp4"
def grab(e,fr):
    c=av.open(str(camp(e)))
    for i,f in enumerate(c.decode(video=0)):
        if i==fr:c.close();return f.to_ndarray(format="rgb24")
    c.close();return None
nc=len(order);ncol=(nc+3)//4;fig,axes=plt.subplots(4,ncol,figsize=(4*ncol,12))
for ax,c in zip(axes.flat,order):
    m=np.where(lab==c)[0];dd=np.linalg.norm(G[m]-allC[c],axis=1);rep=m[np.argmin(dd)]
    img=grab(int(E[rep]),int(FR[rep]))
    if img is not None:ax.imshow(img)
    ax.axis("off");ax.set_title(f"c{c} P={Pk[c]:.2f} {cov_n[c]:.0%}",fontsize=8)
for ax in axes.flat[nc:]:ax.axis("off")
fig.suptitle(f"V2.4 milestones ({nc}个, 增分子cov修正+进度均匀分桶) — 进度均匀填补前段",fontsize=12)
fig.tight_layout();fig.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/vis0526_v24_milestones.png",dpi=110)
fig,ax=plt.subplots(figsize=(12,4));x=np.arange(n7)/3
ax.plot(x,v7,color="#2ca02c",lw=2,label="V2.4 (增分子+分桶)")
ax.axhline(0.9,color="r",ls=":",lw=1,label="旧版ep7误判线0.9")
ax.set_xlabel("seconds");ax.set_ylabel("V");ax.set_ylim(-.05,1.05);ax.legend(fontsize=9);ax.grid(alpha=.3)
ax.set_title(f"ep7 V2.4: 开头抓取段 max={v7[:30].max():.2f} (旧版跳到0.95)",fontsize=11)
fig.tight_layout();fig.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/vis0526_ep7_v24_value.png",dpi=120)
print("DONE_V24")
