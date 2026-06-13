import json, numpy as np, pandas as pd, av, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
REPO=Path("/vePFS/tim/workspace/deepdive_kai0");DS=REPO/"kai0/data/Task_A/vis_base/v3/2026-05-26-v3"
ARM=REPO/"temp/tcc_vis0526_armmask/feat_cache";RAW=REPO/"temp/tcc_vis0526_raw/feat_cache"
cs=json.load(open(DS/"meta/info.json")).get("chunks_size",1000);TEST=-1
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
# autonomy rollout 应用 V2.4
AROLL=REPO/"temp/tcc_autonomy_armmask/feat_cache";RROLL=REPO/"temp/tcc_autonomy_raw/feat_cache"
ROLLDS=REPO/"temp/autonomy"
aR=np.load(AROLL/"ep0.npz")["f"];rR=np.load(RROLL/"ep0.npz")["f"];nR=min(len(aR),len(rR))
stR=np.stack(pd.read_parquet(ROLLDS/"data/chunk-000/episode_000000.parquet",columns=["observation.state"])["observation.state"].to_numpy())
stR=stR[np.minimum(np.arange(nR)*10,len(stR)-1)]
vR=value(aR[:nR],rR[:nR],stR)
print(f"rollout V2.4: range[{vR.min():.2f},{vR.max():.2f}] 三轮均分:")
for s_ in range(3):
    seg=vR[nR*s_//3:nR*(s_+1)//3];print(f"  round{s_+1}: {seg[0]:.2f}->{seg[-1]:.2f} (min{seg.min():.2f})")
import matplotlib.pyplot as plt
# milestone 命中(右上散点诊断): autonomy 每帧最近 milestone + margin
gRf=emb(aR[:nR],rR[:nR],stR);Dr=np.linalg.norm(gRf[:,None]-C[None],axis=2);labR=Dr.argmin(1)
dsrt=np.sort(Dr,axis=1);margR=dsrt[:,0]/dsrt[:,1]
cam=ROLLDS/"videos/chunk-000/top_head/episode_000000.mp4"
cobj=av.open(str(cam));LV=cobj.streams.video[0].frames or 7676;FPS=float(cobj.streams.video[0].average_rate or 30);cobj.close()
NF=LV;V=np.repeat(vR,10)[:NF]
if len(V)<NF:V=np.concatenate([V,np.full(NF-len(V),V[-1])])
lab30=np.repeat(labR,10)[:NF];marg30=np.repeat(margR,10)[:NF]
L=NF;x=np.arange(L)/FPS
MAXSIDE=460
def stream_cam(path):
    c=av.open(str(path))
    for f in c.decode(video=0):
        s=min(1.0,MAXSIDE/max(f.height,f.width))
        g=f.reformat(width=int(f.width*s)//2*2,height=int(f.height*s)//2*2,format="rgb24") if s<1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
import cv2
mscol=matplotlib.colormaps["tab20"]
# ---- 背景 panel: matplotlib 只画一次(右侧两行) ----
PFIG=plt.figure(figsize=(10,7),dpi=100)
gs=PFIG.add_gridspec(2,1,hspace=0.30)
ax_l=PFIG.add_subplot(gs[0])
for k in range(len(order)):
    hit=np.where((lab30==k)&(marg30<=0.8))[0]
    if len(hit):ax_l.scatter(hit/FPS,np.full(len(hit),Pord[k]),s=6,color=mscol(k%20),alpha=.6)
for b in (L/3,2*L/3):ax_l.axvline(b/FPS,color="orange",ls=":",lw=1.2)
ax_l.set_ylim(-0.03,1.03);ax_l.set_xlim(0,L/FPS);ax_l.set_ylabel("milestone P_k")
ax_l.set_title("milestone 命中 (置信 margin<=0.8, 颜色=阶段; 橙=轮次边界)",fontsize=9);ax_l.grid(alpha=.2)
ax_v=PFIG.add_subplot(gs[1],sharex=ax_l)
ax_v.plot(x,V[:L],color="#2ca02c",lw=2,label="V2.4 零训练 milestone-value")
for b in (L/3,2*L/3):ax_v.axvline(b/FPS,color="orange",ls=":",lw=1.2)
ax_v.text(2,1.04,"round1",fontsize=9,color="gray");ax_v.text(L/3/FPS+2,1.04,"round2",fontsize=9,color="gray");ax_v.text(2*L/3/FPS+2,1.04,"round3",fontsize=9,color="gray")
ax_v.set_xlabel("seconds");ax_v.set_ylabel("V");ax_v.set_ylim(-.05,1.13);ax_v.legend(fontsize=8,loc="lower right");ax_v.grid(alpha=.3)
ax_v.set_title("V2.4 value: 轮次边界退步到 0 + 每轮恢复爬升, round3→1.0",fontsize=9)
PFIG.suptitle("autonomy 真机 rollout — V2.4 零训练 milestone-value 同步对齐",fontsize=12)
PFIG.canvas.draw()
PANEL=np.asarray(PFIG.canvas.buffer_rgba())[...,:3].copy();Hp,Wp=PANEL.shape[:2]
PFIG.savefig(REPO/"docs/visualization/cross_episode_recurrence_value/rollout_v24_sync_preview.png",dpi=110)
# ---- 线性像素映射(避免逐帧 matplotlib) ----
def pmap(ax):
    bb=ax.get_position();xlo,xhi=ax.get_xlim();ylo,yhi=ax.get_ylim();return bb.x0,bb.x1,bb.y0,bb.y1,xlo,xhi,ylo,yhi
MV=pmap(ax_v);ML=pmap(ax_l)
def xpx(sec):
    x0,x1,_,_,xlo,xhi,_,_=MV;return int(round((x0+(sec-xlo)/(xhi-xlo)*(x1-x0))*Wp))
def yspan(m):
    _,_,y0,y1,_,_,_,_=m;return int(round((1-y1)*Hp)),int(round((1-y0)*Hp))
LT,LB=yspan(ML);VT,VB=yspan(MV)
def valpx(sec,val):
    x0,x1,y0,y1,xlo,xhi,ylo,yhi=MV
    return (int(round((x0+(sec-xlo)/(xhi-xlo)*(x1-x0))*Wp)),int(round((1-(y0+(val-ylo)/(yhi-ylo)*(y1-y0)))*Hp)))
# ---- camera 尺寸 ----
f0=next(stream_cam(cam));ch,cw=f0.shape[:2];csc=Hp/ch;cw2=int(round(cw*csc))//2*2
Wtot=(cw2+Wp)//2*2;Htot=Hp//2*2
OUT="/vePFS/tim/workspace/deepdive_kai0/temp/rollout_v24_sync.mp4"
oc=av.open(OUT,mode="w")
stv=oc.add_stream("libx264",rate=int(round(FPS)));ENC="libx264(veryfast)";opts={"preset":"veryfast","crf":"23"}
stv.width,stv.height,stv.pix_fmt=Wtot,Htot,"yuv420p";stv.options=opts
print(f"encoder={ENC} canvas={Wtot}x{Htot} frames={L}")
def compose(camimg,t):
    panel=PANEL.copy();px=xpx(t/FPS)
    cv2.line(panel,(px,LT),(px,LB),(110,110,110),2);cv2.line(panel,(px,VT),(px,VB),(110,110,110),2)
    vx,vy=valpx(t/FPS,float(V[min(t,L-1)]));cv2.circle(panel,(vx,vy),7,(44,160,46),-1);cv2.circle(panel,(vx,vy),7,(0,0,0),1)
    cam2=cv2.resize(camimg,(cw2,Hp));canv=np.zeros((Hp,cw2+Wp,3),np.uint8);canv[:,:cw2]=cam2;canv[:,cw2:]=panel
    return np.ascontiguousarray(canv[:Htot,:Wtot])
import time as _t;t0=_t.time();t=0
for camimg in stream_cam(cam):
    if t>=L:break
    vf=av.VideoFrame.from_ndarray(compose(camimg,t),format="rgb24")
    for pkt in stv.encode(vf):oc.mux(pkt)
    t+=1
    if t%1500==0:print(f"  {t}/{L} ({t/(_t.time()-t0):.0f} fps)")
for pkt in stv.encode():oc.mux(pkt)
oc.close()
print(f"SAVED {OUT} {Wtot}x{Htot} {t}f @ {FPS}fps enc={ENC} 用时{_t.time()-t0:.0f}s ({t/(_t.time()-t0):.0f} render-fps)")
print("ROLLOUT_V24_VIDEO_DONE")
