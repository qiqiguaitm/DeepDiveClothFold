import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import kendalltau, pearsonr
REPO=Path("/vePFS/tim/workspace/deepdive_kai0");DS=REPO/"kai0/data/Task_A/kai0_advantage"
ARM=REPO/"temp/tcc_kai0_armmask/feat_cache";RAW=REPO/"temp/tcc_kai0_raw/feat_cache"
cs=json.load(open(DS/"meta/info.json")).get("chunks_size",1000)
zp=np.load(REPO/"temp/recurrence_v0_kai0/embeddings.npz");EVAL=sorted(set(zp["ep_ids"].tolist()))
rawset=set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
pool=np.random.RandomState(0).permutation([e for e in rawset if e not in set(EVAL)]).tolist()
mined=sorted(pool[:500])
def lpst(e,n):
    pq=DS/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
def gt_(e,n):
    pq=DS/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    g=pd.read_parquet(pq,columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy();return g[np.minimum(np.arange(n)*10,len(g)-1)]
def loadep(e):
    a=np.load(ARM/f"ep{e}.npz")["f"];r=np.load(RAW/f"ep{e}.npz")["f"];n=min(len(a),len(r));return a[:n],r[:n],lpst(e,n),n
def mkp(s):return np.concatenate([s,np.vstack([np.zeros((1,s.shape[1])),np.diff(s,axis=0)])],1)
Sall=[loadep(e)[2] for e in mined];Pm=mkp(np.concatenate(Sall));PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a,r,st):
    an=a/np.linalg.norm(a,axis=1,keepdims=True);rn=r/np.linalg.norm(r,axis=1,keepdims=True)
    Pn=((mkp(st)-PMU)/PSD);Pn/=np.linalg.norm(Pn,axis=1,keepdims=True);return np.concatenate([rn,an,Pn],1)
A,R,S,T,E,SP,EP=[],[],[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep(e);g=emb(a,r,st);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e));SP.append(g[:2]);EP.append(g[-2:])
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E);G=emb(A,R,S)
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
print(f"V2.4 kai0 milestones: {len(order)}个, 前段(P<0.5) {sum(1 for c in order if Pk[c]<0.5)}/{len(order)}")
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
    vdp=med(dpHB(em),9)
    Fqn=Fq/np.linalg.norm(Fq,axis=1,keepdims=True);Cn=C/np.linalg.norm(C,axis=1,keepdims=True);cos=Fqn@Cn.T
    vref=np.zeros(nq)
    for t in range(nq):
        g=vdp[t];bl=[i for i in range(len(order)) if Pord[i]<=g];ab=[i for i in range(len(order)) if Pord[i]>g]
        pi=bl[-1] if bl else 0;ni=ab[0] if ab else len(order)-1
        if pi==ni:vref[t]=Pord[pi];continue
        sp,sn=max(cos[t,pi],0),max(cos[t,ni],0);w=(sn+1e-6)/(sp+sn+2e-6);vref[t]=Pord[pi]+w*(Pord[ni]-Pord[pi])
    vref=med(vref,5)
    w=np.exp(cos/0.08);w/=w.sum(1,keepdims=True);vsoft=med(w@Pord,5)
    return vdp,vref,vsoft
R={"DP阶梯":[[],[],[],[]],"段内相似度细化":[[],[],[],[]],"软相似度加权":[[],[],[],[]]}
for e in EVAL:
    if e not in rawset:continue
    a,r,st,n=loadep(e);gt=gt_(e,n)
    if gt.std()<1e-6:continue
    vdp,vref,vsoft=value(a,r,st)
    for nm,v in [("DP阶梯",vdp),("段内相似度细化",vref),("软相似度加权",vsoft)]:
        R[nm][0].append(np.abs(v-gt).mean());R[nm][1].append(pearsonr(v,gt)[0]);R[nm][2].append(kendalltau(v,gt)[0]);R[nm][3].append(np.diff(v).std())
print("V2.4 kai0 held-out GT — 段内细化 vs DP阶梯:")
print(f"{'方法':<18}{'MAE':>8}{'Pearson':>9}{'tau':>8}{'平滑ΔVstd':>11}")
for nm,(m,p,t,sm) in R.items():
    print(f"{nm:<18}{np.nanmean(m):>8.3f}{np.nanmean(p):>9.3f}{np.nanmean(t):>8.3f}{np.nanmean(sm):>11.4f}")
print("(对照 pi0-AE监督 MAE0.054/Pearson0.971; 之前欧氏双锚 §4.4.15 tau0.807<DP)")
print("KAI0_V24_GT_DONE")
