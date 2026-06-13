import json, numpy as np, pandas as pd, av, cv2
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=Path("/vePFS/tim/workspace/deepdive_kai0")
DS=REPO/"kai0/data/Task_A/vis_base/v3/2026-05-20-v3"
ARM=REPO/"temp/tcc_vis0520_armmask/feat_cache"; RAW=REPO/"temp/tcc_vis0520_raw/feat_cache"
cs=json.load(open(DS/"meta/info.json")).get("chunks_size",1000)
def lpst(e,n):
    pq=DS/"data"/f"chunk-{e//cs:03d}"/f"episode_{e:06d}.parquet"
    st=np.stack(pd.read_parquet(pq,columns=["observation.state"])["observation.state"].to_numpy());return st[np.minimum(np.arange(n)*10,len(st)-1)]
all_eps=sorted(int(p.stem[2:]) for p in ARM.glob("ep*.npz"))
TEST=[37]; mined=[e for e in all_eps if e not in TEST]
def loadep(e):
    a=np.load(ARM/f"ep{e}.npz")["f"];r=np.load(RAW/f"ep{e}.npz")["f"];n=min(len(a),len(r))
    return a[:n],r[:n],lpst(e,n),n
def mkp(s):return np.concatenate([s,np.vstack([np.zeros((1,s.shape[1])),np.diff(s,axis=0)])],1)
Sall=[loadep(e)[2] for e in mined];Pm=mkp(np.concatenate(Sall));PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a,r,st):
    an=a/np.linalg.norm(a,axis=1,keepdims=True);rn=r/np.linalg.norm(r,axis=1,keepdims=True)
    Pn=((mkp(st)-PMU)/PSD);Pn/=np.linalg.norm(Pn,axis=1,keepdims=True)
    return np.concatenate([rn,an,Pn],1)  # raw ⊕ armmask ⊕ proprio
A,R,S,T,E=[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep(e);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e))
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E)
G=emb(A,R,S)
km=KMeans(n_clusters=96,n_init=2,random_state=0).fit(G);lab=km.labels_
n_ep=len(set(E.tolist()));cov=np.array([len(set(E[lab==c].tolist()))/n_ep for c in range(96)])
def gr(idx):
    out=[];s=None;pv=None
    for i in idx:
        if pv is None or i!=pv+1:
            if s is not None:out.append((s,pv))
            s=i
        pv=i
    if s is not None:out.append((s,pv))
    return [x for x in out if x[1]-x[0]>=1]
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
C=km.cluster_centers_[ms];NB=21;bins=np.linspace(0,1,NB)
cb={ci:[int(np.argmin(abs(bins-m))) for m in modes[c]] for ci,c in enumerate(ms)}
def dp_hardbound(emit,lam=8.0):
    pen=lam*np.abs(bins[:,None]-bins[None]);NF=len(emit)
    cost=np.full(NB,1e9);cost[0]=emit[0,0]  # 硬边界 V[0]=0: 首帧只允许bin0
    bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen;k=tr.argmin(1);cost=emit[j]+tr[np.arange(NB),k];bp[j]=k
    cost[NB-1]+=-2.0  # 末帧偏好bin20(轻奖励完成态)
    path=np.zeros(NF,int);path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1):path[j]=bp[j+1,path[j+1]]
    return bins[path]
def med(a,w):
    h=w//2;return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
def value_of(a,r,st):
    Fq=emb(a,r,st);nq=len(Fq)
    d=np.linalg.norm(Fq[:,None]-C[None],axis=2);em=np.full((nq,NB),1e3)
    for ci in range(len(ms)):
        for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d[:,ci])
    return med(dp_hardbound(em),9)
# 渲染 ep37
e=37;a,r,st,n=loadep(e);V3hz=value_of(a,r,st)
cam=DS/"videos"/f"chunk-{e//cs:03d}"/"observation.images.top_head"/f"episode_{e:06d}.mp4"
cc=av.open(str(cam));NF=min(n*10,cc.streams.video[0].frames or n*10);cc.close()
V=np.repeat(V3hz,10)[:NF]
print(f"V2.3 ep37: range[{V.min():.2f},{V.max():.2f}] init={V[:90].mean():.2f} end={V[-90:].mean():.2f}")
fig,ax=plt.subplots(figsize=(8.5,5.6),dpi=100);x=np.arange(NF)/30
ax.plot(x,V,color="#2ca02c",lw=2.3)
ax.set_xlabel("seconds");ax.set_ylabel("V (0=start, 1=folded)");ax.set_ylim(-.05,1.08);ax.set_xlim(0,NF/30);ax.grid(alpha=.3)
ax.set_title("vis 5-20 ep37 (ORANGE cloth) — V2.3 robust (raw⊕armmask⊕proprio + hard V0=0)",fontsize=9.5)
fig.tight_layout();fig.canvas.draw();bg=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();Hf,Wf=bg.shape[:2]
def px(xd,yd):p=ax.transData.transform((xd,yd));return int(round(p[0])),int(round(Hf-p[1]))
def stream():
    c=av.open(str(cam))
    for fr in c.decode(video=0):
        aa=fr.to_ndarray(format="rgb24");h,w=aa.shape[:2];yield cv2.resize(aa,(int(w*Hf/h),Hf))
    c.close()
out="/vePFS/tim/workspace/deepdive_kai0/temp/vis0520_ep37_v23_sync.mp4"
oc=av.open(out,mode="w");stm=oc.add_stream("libx264",rate=30);stm.options={"crf":"23","preset":"veryfast"};first=True;t=0
for left in stream():
    if t>=NF:break
    if first:Wl=left.shape[1];W=(Wl+Wf)//2*2;H=Hf//2*2;stm.width,stm.height,stm.pix_fmt=W,H,"yuv420p";first=False
    cv=np.concatenate([left,bg],1);col=px(t/30,0)[0]+Wl;y0=px(t/30,0)[1];y1=px(t/30,1.0)[1]
    if 0<=col<cv.shape[1]:cv[min(y1,y0):max(y1,y0),max(0,col-1):col+1]=[0,0,0]
    dx,dy=px(t/30,V[t]);dx+=Wl;cv[max(0,dy-4):dy+4,max(0,dx-4):dx+4]=[44,160,44]
    vf=av.VideoFrame.from_ndarray(np.ascontiguousarray(cv[:H,:W]),format="rgb24")
    for pk in stm.encode(vf):oc.mux(pk)
    t+=1
for pk in stm.encode():oc.mux(pk)
oc.close();print(f"SAVED {out} {W}x{H} {t}f");print("DONE_V23")
