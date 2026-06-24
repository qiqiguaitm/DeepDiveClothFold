import json, numpy as np, pandas as pd, av, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
REPO=Path("/vePFS/tim/workspace/deepdive_kai0")
DS=REPO/"kai0/data/Task_A/vis_base/v3/2026-05-26-v3"
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
A,R,S,T,E,SP,EP=[],[],[],[],[],[],[]
for e in mined:
    a,r,st,n=loadep(e);g=emb(a,r,st);A.append(a);R.append(r);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e));SP.append(g[:2]);EP.append(g[-2:])
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E);G=emb(A,R,S)
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
order=sorted(msC,key=lambda c:Pk[c]);C=km.cluster_centers_[order];Pord=np.array([Pk[c] for c in order])
startK=KMeans(8,n_init=2,random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK=KMeans(8,n_init=2,random_state=0).fit(np.concatenate(EP)).cluster_centers_
NB=21;bins=np.linspace(0,1,NB);cb=[[int(np.argmin(abs(bins-Pk[c])))] for c in order]
def dpHB(emit,lam=8.0):
    pen=lam*np.abs(bins[:,None]-bins[None]);NF=len(emit);cost=np.full(NB,1e9);cost[0]=emit[0,0];bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen;k=tr.argmin(1);cost=emit[j]+tr[np.arange(NB),k];bp[j]=k
    cost[NB-1]-=2.0;path=np.zeros(NF,int);path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1):path[j]=bp[j+1,path[j+1]]
    return bins[path]
def med(a,w):
    h=w//2;return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
def four(a,r,st):
    Fq=emb(a,r,st);nq=len(Fq);d=np.linalg.norm(Fq[:,None]-C[None],axis=2)
    em=np.full((nq,NB),1e3)
    for ci in range(len(order)):
        for b in cb[ci]:em[:,b]=np.minimum(em[:,b],d[:,ci])
    ds=np.linalg.norm(Fq[:,None]-startK[None],axis=2).min(1);de=np.linalg.norm(Fq[:,None]-endK[None],axis=2).min(1)
    tn=np.arange(nq)/nq;em[:,0]=np.minimum(em[:,0],np.where(tn<0.3,ds,ds+(tn-0.3)*6));em[:,NB-1]=np.minimum(em[:,NB-1],np.where(tn>0.6,de,de+(0.6-tn)*6))
    vdp=med(dpHB(em),9)
    # 双锚欧氏 / 测地
    cumd=np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(Fq,axis=0),axis=1))])
    hitf={i:(np.where(vdp>=Pord[i]-0.025)[0][0] if (vdp>=Pord[i]-0.025).any() else None) for i in range(len(order))}
    veuc=np.zeros(nq);vgeo=np.zeros(nq)
    for t in range(nq):
        g=vdp[t];bl=[i for i in range(len(order)) if Pord[i]<=g];ab=[i for i in range(len(order)) if Pord[i]>g]
        pi=bl[-1] if bl else 0;ni=ab[0] if ab else len(order)-1
        if pi==ni:veuc[t]=Pord[pi];vgeo[t]=Pord[pi];continue
        w=d[t,pi]/(d[t,pi]+d[t,ni]+1e-9);veuc[t]=Pord[pi]+w*(Pord[ni]-Pord[pi])
        tp,tnn=hitf.get(pi),hitf.get(ni)
        if tp is None or tnn is None or tnn<=tp:vgeo[t]=Pord[pi]
        else:vgeo[t]=Pord[pi]+np.clip((cumd[t]-cumd[tp])/(cumd[tnn]-cumd[tp]+1e-9),0,1)*(Pord[ni]-Pord[pi])
    # 簇内2-NN
    t2=np.argsort(d,axis=1)[:,:2];vref=np.zeros(nq)
    for t in range(nq):
        c1,c2=t2[t];w1=1/(d[t,c1]+1e-6);w2=1/(d[t,c2]+1e-6);vref[t]=(Pord[c1]*w1+Pord[c2]*w2)/(w1+w2)
    return vdp, med(veuc,5), med(vgeo,5), med(vref,5)
a,r,st,n=loadep(TEST);vdp,veuc,vgeo,vref=four(a,r,st)
cam=DS/"videos"/f"chunk-{TEST//cs:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4"
cc=av.open(str(cam));NF=cc.streams.video[0].frames or n*10;cc.close();NF=min(NF,n*10)
def up(v):return np.repeat(v,10)[:NF]
V={"DP (V2.3 main)":up(vdp),"double-anchor Euclidean":up(veuc),"geodesic":up(vgeo),"cluster 2-NN":up(vref)}
cols=["#2ca02c","#d62728","#1f77b4","#ff7f0e"]
print(f"ep7 NF={NF}; ranges:",{k:(round(v.min(),2),round(v.max(),2)) for k,v in V.items()})
MAX=360
def stream():
    c=av.open(str(cam))
    for f in c.decode(video=0):
        s=min(1.0,MAX/max(f.height,f.width));g=f.reformat(width=int(f.width*s)//2*2,height=int(f.height*s)//2*2,format="rgb24") if s<1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
f0=next(stream())
fig=plt.figure(figsize=(15,7));gs=fig.add_gridspec(4,2,width_ratios=[1,1.5],hspace=0.45,wspace=0.12)
axc=fig.add_subplot(gs[:,0]);axc.axis("off");im=axc.imshow(f0);axc.set_title("vis 5-26 ep7",fontsize=10)
x=np.arange(NF)/30;axes=[];curs=[];dots=[]
for i,(nm,v) in enumerate(V.items()):
    ax=fig.add_subplot(gs[i,1]);ax.plot(x,v,color=cols[i],lw=1.6);ax.set_ylim(-.05,1.08);ax.set_ylabel(nm,fontsize=7.5);ax.grid(alpha=.3)
    if i<3:ax.set_xticklabels([])
    cu=ax.axvline(0,color="k",lw=1);dt,=ax.plot([0],[v[0]],"o",color=cols[i],ms=6,mec="k")
    axes.append(ax);curs.append(cu);dots.append(dt)
axes[-1].set_xlabel("seconds")
fig.suptitle("vis 5-26 ep7: four value methods vs live frame — which tracks真实进度?",fontsize=11)
def render(img,t):
    im.set_data(img)
    for i,(nm,v) in enumerate(V.items()):
        curs[i].set_xdata([t/30,t/30]);dots[i].set_data([t/30],[v[t]])
    fig.canvas.draw();return np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[...,:3])
g0=render(f0,0);H,Wd=g0.shape[:2];H-=H%2;Wd-=Wd%2
oc=av.open("/vePFS/tim/workspace/deepdive_kai0/temp/vis0526_ep7_4method_sync.mp4",mode="w");stm=oc.add_stream("libx264",rate=30);stm.width,stm.height,stm.pix_fmt=Wd,H,"yuv420p";stm.options={"crf":"23"}
t=0
for img in stream():
    if t>=NF:break
    vf=av.VideoFrame.from_ndarray(render(img,t)[:H,:Wd],format="rgb24")
    for pk in stm.encode(vf):oc.mux(pk)
    t+=1
for pk in stm.encode():oc.mux(pk)
oc.close();print(f"SAVED vis0526_ep7_4method_sync.mp4 {Wd}x{H} {t}f");print("DONE_4M")
