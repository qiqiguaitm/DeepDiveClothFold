import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=Path("/vePFS/tim/workspace/deepdive_kai0")
ds=REPO/"kai0/data/Task_A/kai0_advantage";cs=json.load(open(ds/"meta/info.json")).get("chunks_size",1000)
cache=REPO/"temp/tcc_kai0_armmask/feat_cache"
zp=np.load(REPO/"temp/recurrence_v0_kai0/embeddings.npz");EVAL=sorted(set(zp["ep_ids"].tolist()))
all_eps=sorted(int(p.stem[2:]) for p in cache.glob("ep*.npz"))
pool=np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()
def lpst(e,n):
    pq=ds/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
def gt_(e,n):
    pq=ds/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    g=pd.read_parquet(pq,columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy();return g[np.minimum(np.arange(n)*10,len(g)-1)]
def mkp(s):return np.concatenate([s,np.vstack([np.zeros((1,s.shape[1])),np.diff(s,axis=0)])],1)
def load(eps):
    A,S,T,E=[],[],[],[]
    for e in eps:
        f=np.load(cache/f"ep{e}.npz")["f"];n=len(f);A.append(f);S.append(lpst(e,n));T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e))
    return np.concatenate(A),np.concatenate(S),np.concatenate(T),np.concatenate(E)
Am,Sm,Tm,Em=load(pool[:500]);Pm=mkp(Sm);PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a,st):
    an=a/np.linalg.norm(a,axis=1,keepdims=True);Pn=((mkp(st)-PMU)/PSD);Pn/=np.linalg.norm(Pn,axis=1,keepdims=True);return np.concatenate([an,Pn],1)
G=emb(Am,Sm);km=KMeans(96,n_init=2,random_state=0).fit(G);lab=km.labels_
n_ep=len(set(Em.tolist()));cov=np.array([len(set(Em[lab==c].tolist()))/n_ep for c in range(96)])
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
    for e in sorted(set(Em.tolist())):
        m=np.where(Em==e)[0];rs=gr(m[lab[m]==c].tolist())
        if rs:fe.append(Tm[rs[0][0]])
    Pk[c]=float(np.median(fe)) if fe else .5
order=sorted(msC,key=lambda c:Pk[c]);C=km.cluster_centers_[order];Pord=np.array([Pk[c] for c in order])
NB=21;bins=np.linspace(0,1,NB);cb=[[int(np.argmin(abs(bins-Pk[c])))] for c in order]
def dpV(emit,lam=8.0):
    pen=lam*np.abs(bins[:,None]-bins[None]);NF=len(emit);cost=emit[0].copy();bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen;k=tr.argmin(1);cost=emit[j]+tr[np.arange(NB),k];bp[j]=k
    path=np.zeros(NF,int);path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1):path[j]=bp[j+1,path[j+1]]
    return bins[path]
def med(a,w):
    h=w//2;return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
def vals(a,st):
    Fq=emb(a,st);nq=len(Fq);d=np.linalg.norm(Fq[:,None]-C[None],axis=2)
    em=np.full((nq,NB),1e3)
    for ci in range(len(order)):
        for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d[:,ci])
    vdp=med(dpV(em),9)
    vint_raw=np.zeros(nq)
    for t in range(nq):
        g=vdp[t]
        below=[i for i in range(len(order)) if Pord[i]<=g];above=[i for i in range(len(order)) if Pord[i]>g]
        pi=below[-1] if below else 0; ni=above[0] if above else len(order)-1
        if pi==ni: vint_raw[t]=Pord[pi];continue
        w=d[t,pi]/(d[t,pi]+d[t,ni]+1e-9);vint_raw[t]=Pord[pi]+w*(Pord[ni]-Pord[pi])
    # V2.4 测地: 沿本ep轨迹累积特征位移
    cumd=np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(Fq,axis=0),axis=1))])
    hitf={}
    for i in range(len(order)):
        idx=np.where(vdp>=Pord[i]-0.025)[0];hitf[i]=int(idx[0]) if len(idx) else None
    vgeo=np.zeros(nq)
    for t in range(nq):
        g=vdp[t]
        below=[i for i in range(len(order)) if Pord[i]<=g];above=[i for i in range(len(order)) if Pord[i]>g]
        pi=below[-1] if below else 0; ni=above[0] if above else len(order)-1
        tp=hitf.get(pi);tn=hitf.get(ni)
        if pi==ni or tp is None or tn is None or tn<=tp:
            vgeo[t]=Pord[pi];continue
        frac=np.clip((cumd[min(t,nq-1)]-cumd[tp])/(cumd[tn]-cumd[tp]+1e-9),0,1)
        vgeo[t]=Pord[pi]+frac*(Pord[ni]-Pord[pi])
    return vdp, med(vint_raw,5), med(vgeo,5)
STATS=[];sel=EVAL[:6]
fig,axes=plt.subplots(2,3,figsize=(16,8))
for ax,e in zip(axes.flat,sel):
    a=np.load(cache/f"ep{e}.npz")["f"];n=len(a);st=lpst(e,n);gt=gt_(e,n);x=np.arange(n)/3
    vdp,veuc,vgeo=vals(a,st)
    ax.plot(x,gt,"k-",lw=2,label="GT (linear)")
    ax.step(x,vdp,color="#888",lw=1.3,where="post",label="DP (mono step)")
    ax.plot(x,veuc,color="#d62728",lw=1.5,label="interp EUCLIDEAN")
    ax.plot(x,vgeo,color="#2ca02c",lw=1.9,label="interp GEODESIC (V2.4)")
    from scipy.stats import kendalltau as ktau
    STATS.append((abs(veuc-gt).mean(),abs(vgeo-gt).mean(),ktau(veuc,gt)[0],ktau(vgeo,gt)[0]))
    ax.set_title(f"kai0 ep{e}",fontsize=9);ax.set_ylim(-.05,1.05);ax.grid(alpha=.3);ax.set_xlabel("s")
    ax.legend(fontsize=6.5,loc="upper left")
STATS_arr=np.array(STATS);print(f"EUCLIDEAN: MAE={STATS_arr[:,0].mean():.3f} tau={STATS_arr[:,2].mean():.3f}");print(f"GEODESIC : MAE={STATS_arr[:,1].mean():.3f} tau={STATS_arr[:,3].mean():.3f}")
fig.suptitle("V2.4 geodesic interpolation (along-trajectory cumulative displacement) vs Euclidean vs DP vs GT",fontsize=12)
fig.tight_layout();fig.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/v24_geodesic_interp_kai0.png",dpi=120)
print("DONE_VIZ")
