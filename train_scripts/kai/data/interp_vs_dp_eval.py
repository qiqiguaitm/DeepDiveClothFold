import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import kendalltau, pearsonr
REPO=Path("/vePFS/tim/workspace/deepdive_kai0")
ds=REPO/"kai0/data/Task_A/kai0_advantage";cs=json.load(open(ds/"meta/info.json")).get("chunks_size",1000)
cache=REPO/"temp/tcc_kai0_armmask/feat_cache"
zp=np.load(REPO/"temp/recurrence_v0_kai0/embeddings.npz");EVAL=sorted(set(zp["ep_ids"].tolist()))
all_eps=sorted(int(p.stem[2:]) for p in cache.glob("ep*.npz"))
pool=np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()
def lpst(e,n):
    pq=ds/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
def gtav(e,n):
    pq=ds/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    df=pd.read_parquet(pq,columns=["stage_progress_gt","absolute_value"])
    idx=np.minimum(np.arange(n)*10,len(df)-1)
    return df["stage_progress_gt"].to_numpy()[idx], df["absolute_value"].to_numpy()[idx]
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
def values(a,st):
    Fq=emb(a,st);nq=len(Fq);d=np.linalg.norm(Fq[:,None]-C[None],axis=2)  # (nq,20)
    em=np.full((nq,NB),1e3)
    for ci in range(len(order)):
        for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d[:,ci])
    vdp=med(dpV(em),9)
    # 用户双锚距离插值: 用vdp粗定位前后milestone, 特征距离插值
    vint=np.zeros(nq)
    for t in range(nq):
        g=vdp[t]
        below=[i for i in range(len(order)) if Pord[i]<=g];above=[i for i in range(len(order)) if Pord[i]>g]
        pi=below[-1] if below else 0; ni=above[0] if above else len(order)-1
        if pi==ni: vint[t]=Pord[pi];continue
        dp_=d[t,pi];dn_=d[t,ni];w=dp_/(dp_+dn_+1e-9)
        vint[t]=Pord[pi]+w*(Pord[ni]-Pord[pi])
    return vdp, med(vint,5)
res={"DP(现状)":[[],[],[]],"双锚距离插值(用户)":[[],[],[]],"pi0-AE(监督)":[[],[],[]]}
for e in EVAL:
    a=np.load(cache/f"ep{e}.npz")["f"];n=len(a);st=lpst(e,n);gt,av=gtav(e,n)
    if gt.std()<1e-6:continue
    vdp,vint=values(a,st)
    for nm,v in [("DP(现状)",vdp),("双锚距离插值(用户)",vint),("pi0-AE(监督)",av)]:
        res[nm][0].append(np.abs(v-gt).mean());res[nm][1].append(pearsonr(v,gt)[0]);res[nm][2].append(kendalltau(v,gt)[0])
print("kai0 held-out 50 GT ep — value vs stage_progress_gt:")
print(f"{'方法':<22}{'MAE':>8}{'Pearson':>9}{'τ':>8}")
for nm,(m,p,t) in res.items():
    print(f"{nm:<22}{np.nanmean(m):>8.3f}{np.nanmean(p):>9.3f}{np.nanmean(t):>8.3f}")
# 平滑度: 相邻帧value差的std (越小越平滑)
print("\n平滑度(相邻帧ΔV的std, 越小越连续):")
sm={"DP":[],"插值":[]}
for e in EVAL[:20]:
    a=np.load(cache/f"ep{e}.npz")["f"];n=len(a);st=lpst(e,n)
    vdp,vint=values(a,st)
    sm["DP"].append(np.diff(vdp).std());sm["插值"].append(np.diff(vint).std())
print(f"  DP={np.mean(sm['DP']):.4f}  插值={np.mean(sm['插值']):.4f}")
print("DONE_INT")
