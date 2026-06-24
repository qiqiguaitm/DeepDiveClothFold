import json, colorsys, numpy as np, pandas as pd, av, cv2, torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
REPO=Path("/vePFS/tim/workspace/deepdive_kai0"); VB=REPO/"kai0/data/Task_A/vis_base/v3"
PROTO=np.load(REPO/"temp/armmask/arm_prototypes.npz")["proto"];THR=0.6;P=16
dev="cuda";proc=AutoImageProcessor.from_pretrained("facebook/dinov2-small")
enc=AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval();proto_t=torch.from_numpy(PROTO).float().to(dev)
# ---- 挖掘集 vis0520 三路 (现成缓存) ----
DS0=VB/"2026-05-20-v3";cs=json.load(open(DS0/"meta/info.json")).get("chunks_size",1000)
ARM=REPO/"temp/tcc_vis0520_armmask/feat_cache";RAW=REPO/"temp/tcc_vis0520_raw/feat_cache"
def lpst0(e,n):
    pq=DS0/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
mined=sorted(int(p.stem[2:]) for p in ARM.glob("ep*.npz"))
def mkp(s):return np.concatenate([s,np.vstack([np.zeros((1,s.shape[1])),np.diff(s,axis=0)])],1)
def loadep0(e):
    a=np.load(ARM/f"ep{e}.npz")["f"];r=np.load(RAW/f"ep{e}.npz")["f"];n=min(len(a),len(r));return a[:n],r[:n],lpst0(e,n),n
Sall=[loadep0(e)[2] for e in mined];Pm=mkp(np.concatenate(Sall));PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a,r,st):
    an=a/np.linalg.norm(a,axis=1,keepdims=True);rn=r/np.linalg.norm(r,axis=1,keepdims=True)
    Pn=((mkp(st)-PMU)/PSD);Pn/=np.linalg.norm(Pn,axis=1,keepdims=True);return np.concatenate([rn,an,Pn],1)
A,R,S,T,E=[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep0(e);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e))
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E)
G=emb(A,R,S);km=KMeans(96,n_init=2,random_state=0).fit(G);lab=km.labels_
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
ms=np.argsort(cov)[-20:].tolist();modes={}
for c in ms:
    starts=[]
    for e in sorted(set(E.tolist())):
        m=np.where(E==e)[0]
        for a,b in gr(m[lab[m]==c].tolist()):starts.append(T[a])
    X=np.array(starts).reshape(-1,1)
    if len(X)<8:modes[c]=[float(np.median(starts))];continue
    g1=GaussianMixture(1,random_state=0).fit(X);g2=GaussianMixture(2,random_state=0).fit(X)
    modes[c]=sorted(g2.means_.ravel().tolist()) if g1.bic(X)-g2.bic(X)>10 else [float(np.median(starts))]
C=km.cluster_centers_[ms];NB=21;bins=np.linspace(0,1,NB);cb={ci:[int(np.argmin(abs(bins-m))) for m in modes[c]] for ci,c in enumerate(ms)}
def dpV(emit,lam=8.0):
    pen=lam*np.abs(bins[:,None]-bins[None]);NF=len(emit);cost=np.full(NB,1e9);cost[0]=emit[0,0];bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen;k=tr.argmin(1);cost=emit[j]+tr[np.arange(NB),k];bp[j]=k
    cost[NB-1]-=2.0;path=np.zeros(NF,int);path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1):path[j]=bp[j+1,path[j+1]]
    return bins[path]
def med(a,w):
    h=w//2;return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
def extract_both(DS,e,cs2):
    mp4=DS/"videos"/f"chunk-{e//cs2:03d}"/"observation.images.top_head"/f"episode_{e:06d}.mp4"
    imgs=[];c=av.open(str(mp4))
    for i,f in enumerate(c.decode(video=0)):
        if i%10:continue
        a=f.to_ndarray(format="rgb24");h,w=a.shape[:2];s=224/min(h,w);a=cv2.resize(a,(round(w*s),round(h*s)));hh,ww=a.shape[:2]
        imgs.append(a[(hh-224)//2:(hh+224)//2,(ww-224)//2:(ww+224)//2])
    c.close();arm=[];raw=[]
    with torch.no_grad():
        for b in range(0,len(imgs),32):
            batch=imgs[b:b+32];px=proc(images=batch,return_tensors="pt").to(dev);toks=enc(**px).last_hidden_state[:,1:]
            raw.append(torch.nn.functional.normalize(toks.mean(1),dim=-1).cpu().numpy())
            tn=torch.nn.functional.normalize(toks,dim=-1);sim=(tn@proto_t.T).max(-1).values;om=[]
            for im in batch:
                rgb=im.reshape(P,14,P,14,3).mean((1,3))/255.0
                hsv=np.array([[colorsys.rgb_to_hsv(*rgb[i,j]) for j in range(P)] for i in range(P)])
                om.append(((hsv[...,0]>0.02)&(hsv[...,0]<0.12)&(hsv[...,1]>0.4)&(hsv[...,2]>0.25)).reshape(-1))
            om=torch.from_numpy(np.stack(om)).to(dev);keep=(~((sim>THR)|om)).float().unsqueeze(-1)
            arm.append(torch.nn.functional.normalize((toks*keep).sum(1)/keep.sum(1).clamp(min=8),dim=-1).cpu().numpy())
    return np.concatenate(arm),np.concatenate(raw)
def value(DS,e,cs2):
    arm,raw=extract_both(DS,e,cs2);n=min(len(arm),len(raw))
    pq=DS/"data"/f"chunk-{e//cs2:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy())[np.minimum(np.arange(n)*10,9999999)]
    st=st[:n] if len(st)>=n else np.pad(st,((0,n-len(st)),(0,0)),mode='edge')
    Fq=emb(arm[:n],raw[:n],st);d=np.linalg.norm(Fq[:,None]-C[None],axis=2);em=np.full((n,NB),1e3)
    for ci in range(len(ms)):
        for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d[:,ci])
    return med(dpV(em),9)
tests=[("2026-04-23-v3",5),("2026-04-23-v3",15),("2026-04-29-v3",10),("2026-04-29-v3",60),
       ("2026-05-06-v3",10),("2026-05-06-v3",60),("2026-05-10-v3",10),("2026-05-10-v3",60),
       ("2026-05-19-v3",10),("2026-05-19-v3",60),("2026-05-22-v3",10),("2026-05-22-v3",60),
       ("2026-05-27-v3",10),("2026-05-27-v3",60),("2026-05-28-v3",10),("2026-05-28-v3",80)]
fig,axes=plt.subplots(4,4,figsize=(16,11))
for ax,(date,e) in zip(axes.flat,tests):
    try:
        DS=VB/date;cs2=json.load(open(DS/"meta/info.json")).get("chunks_size",1000)
        V=value(DS,e,cs2);x=np.arange(len(V))/3
        ok=(V[:5].mean()<0.25) and (V[-5:].mean()>0.7)
        ax.plot(x,V,color="#2ca02c" if ok else "#d62728",lw=1.8)
        ax.set_title(f"{date[5:10]} ep{e}  init{V[:5].mean():.2f} end{V[-5:].mean():.2f} {'OK' if ok else 'BAD'}",fontsize=8)
        print(f"{date[5:10]} ep{e}: init={V[:5].mean():.2f} end={V[-5:].mean():.2f} {'OK' if ok else 'BAD'}")
    except Exception as ex:
        ax.set_title(f"{date[5:10]} ep{e} ERR {str(ex)[:20]}",fontsize=7);print(f"{date} ep{e} ERR {ex}")
    ax.set_ylim(-.05,1.05);ax.grid(alpha=.3)
fig.suptitle("V2.3 robustness — cross-date sampling (mined on 5-20, applied to 8 dates): value curves",fontsize=12)
fig.tight_layout();fig.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/v23_crossday_robustness.png",dpi=120)
print("DONE_CD")
