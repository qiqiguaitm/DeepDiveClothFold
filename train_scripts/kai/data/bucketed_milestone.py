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
# ep7: 到20个milestone的距离 + DP value
order=sorted(msC,key=lambda c:Pk[c]);C=km.cluster_centers_[order];Pord=np.array([Pk[c] for c in order])
a7,r7,st7,n7=loadep(TEST);F7=emb(a7,r7,st7)
d7m=np.linalg.norm(F7[:,None]-C[None],axis=2);match=d7m.argmin(1)
NB=21;bins=np.linspace(0,1,NB);cb=[[int(np.argmin(abs(bins-Pk[c])))] for c in order]
em=np.full((n7,NB),1e3)
for ci in range(len(order)):
    for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d7m[:,ci])
def dpHB(emit,lam=8.0):
    pen=lam*np.abs(bins[:,None]-bins[None]);NF=len(emit);cost=np.full(NB,1e9);cost[0]=emit[0,0];bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen;k=tr.argmin(1);cost=emit[j]+tr[np.arange(NB),k];bp[j]=k
    cost[NB-1]-=2;path=np.zeros(NF,int);path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1):path[j]=bp[j+1,path[j+1]]
    return bins[path]
vdp=dpHB(em)
print("ep7 前段 每帧匹配milestone(到20簇最近) + DP value:")
for t in range(min(40,n7)):
    mi=match[t];print(f"  t={t/3:.1f}s c{order[mi]:02d} P={Pord[mi]:.2f} cov={cov[order[mi]]:.0%} | DPv={vdp[t]:.2f}")
allC=km.cluster_centers_
# 快速上升: DP value 第一次>=0.8 的帧, 看它匹配的milestone
rise=np.where(vdp>=0.8)[0]
tt=int(rise[0]) if len(rise) else int(np.argmax(vdp))
tc=order[match[tt]];tP=Pord[match[tt]]
print(f"\n>>> DP value 首次到{vdp[tt]:.2f} 在 t={tt/3:.1f}s, 该帧匹配 milestone c{tc} P={tP:.2f} cov={cov[tc]:.0%}")
m=np.where(lab==tc)[0];dd=np.linalg.norm(G[m]-allC[tc],axis=1);reps=m[np.argsort(dd)[:4]]
def camp(e):return DS/"videos"/f"chunk-{e//cs:03d}"/"observation.images.top_head"/f"episode_{e:06d}.mp4"
def grab(e,fr):
    c=av.open(str(camp(e)))
    for i,f in enumerate(c.decode(video=0)):
        if i==fr:c.close();return f.to_ndarray(format="rgb24")
    c.close();return None
fig,axes=plt.subplots(2,4,figsize=(15,7))
ep7_hit=[t for t in range(n7) if match[t]==match[tt]][:4]
for ax,t in zip(axes[0],ep7_hit):
    img=grab(TEST,t*10)
    if img is not None:ax.imshow(img)
    ax.axis("off");ax.set_title(f"ep7 t={t/3:.1f}s DPv={vdp[t]:.2f}",fontsize=8)
for ax in axes[0][len(ep7_hit):]:ax.axis("off")
for ax,i in zip(axes[1],reps):
    img=grab(int(E[i]),int(FR[i]))
    if img is not None:ax.imshow(img)
    ax.axis("off");ax.set_title(f"c{tc}代表 ep{int(E[i])} f{int(FR[i])}",fontsize=8)
fig.suptitle(f"ep7快速上升匹配簇 c{tc} (P={tP:.2f} cov={cov[tc]:.0%}): 上=ep7命中画面(抓起) 下=该簇典型代表帧 — 误判?",fontsize=11)
fig.tight_layout();fig.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/ep7_rising_cluster_check.png",dpi=115)
print("DONE_DIAG")
# 20 milestone簇代表帧网格 (按P_k排序)
import av
# 进度分桶重选 milestone (按tpos均匀分桶, 填前段空洞)
tpos=np.array([T[lab==c].mean() if (lab==c).any() else 0.5 for c in range(96)])
bk=np.linspace(0,1,11);sel=[]
for b in range(10):
    inb=[c for c in range(96) if bk[b]<=tpos[c]<bk[b+1]]
    if inb: sel+=sorted(inb,key=lambda c:-cov[c])[:2]
sel=sorted(set(sel),key=lambda c:tpos[c])
def gr2(idx):
    o=[];s=None;pv=None
    for i in idx:
        if pv is None or i!=pv+1:
            if s is not None:o.append((s,pv))
            s=i
        pv=i
    if s is not None:o.append((s,pv))
    return [x for x in o if x[1]-x[0]>=1]
for c in sel:
    if c in Pk:continue
    fe=[]
    for e in sorted(set(E.tolist())):
        m=np.where(E==e)[0];rs=gr2(m[lab[m]==c].tolist())
        if rs:fe.append(T[rs[0][0]])
    Pk[c]=float(np.median(fe)) if fe else float(tpos[c])
order=sel;allC=km.cluster_centers_
print("分桶重选 milestone (按tpos):",[(f"c{c}",round(Pk[c],2),f"{cov[c]:.0%}",f"t{tpos[c]:.2f}") for c in order])
def camp(e):return DS/"videos"/f"chunk-{e//cs:03d}"/"observation.images.top_head"/f"episode_{e:06d}.mp4"
def grab(e,fr):
    c=av.open(str(camp(e)))
    for i,f in enumerate(c.decode(video=0)):
        if i==fr:c.close();return f.to_ndarray(format="rgb24")
    c.close();return None
fig,axes=plt.subplots(4,6,figsize=(20,12))
for ax,c in zip(axes.flat,order):
    m=np.where(lab==c)[0];dd=np.linalg.norm(G[m]-allC[c],axis=1);rep=m[np.argmin(dd)]
    img=grab(int(E[rep]),int(FR[rep]))
    if img is not None:ax.imshow(img)
    ax.axis("off");ax.set_title(f"c{c:02d}  P={Pk[c]:.2f}  cov={cov[c]:.0%}",fontsize=9)
fig.suptitle("vis 5-26: BUCKETED milestones (tpos-uniform, fills early gap) — representative frame of each. 看是否清晰递进/有视觉别名",fontsize=12)
fig.tight_layout();fig.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/vis0526_bucketed_milestones.png",dpi=110)
print("bucketed_sel:",[(f"c{c}",round(Pk[c],2),f"{cov[c]:.0%}") for c in order])
print("DONE_20")
