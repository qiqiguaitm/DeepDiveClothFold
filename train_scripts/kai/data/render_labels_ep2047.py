"""kai0_base ep2047: 在三方value基础上加两条 正/负标签带 —— TCC连续DP 的 advantage(ΔV)符号
vs AWBC-AE 的 absolute_advantage 符号(AE直接输出的相对value)。画面+3曲线+2标签带 同步mp4。
"""
import json
from pathlib import Path
import numpy as np, av, pandas as pd, matplotlib, os
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
_sh=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"]=["SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
R=Path("/vePFS/tim/workspace/deepdive_kai0"); BASE=R/"kai0/data/Task_A/kai0_base"; FR=R/"temp/tcc_e2e_frames/kai0base"
Q5=R/"kai0/data/Task_A/self_built/advantage_q5"; TEST=2047; W=50
csB=json.load(open(BASE/"meta/info.json"))["chunks_size"]; csQ=json.load(open(Q5/"meta/info.json"))["chunks_size"]
s=np.load(R/"temp/_solve_ep2047_30hz.npz"); crave=s["crave"]; ae=s["ae"]
# TCC 连续(DP, 同 render_3way)
z=np.load(R/"temp/_sim_ep2047.npz"); sim=z["sim"].astype(np.float32); n=sim.shape[0]
eps=sorted(int(p.stem[2:]) for p in FR.glob("ep*.npz")); REFS=[e for e in eps if e!=TEST][:30]
bankt=np.concatenate([np.arange(L)/max(1,L-1) for L in [len(np.load(FR/f"ep{e}.npz")["frames"]) for e in REFS]])
NB=201; bins=np.linspace(0,1,NB); binid=np.clip((bankt*(NB-1)).round().astype(int),0,NB-1)
simb=np.full((n,NB),-9.0)
for b in range(NB):
    c=np.where(binid==b)[0]
    if len(c): simb[:,b]=sim[:,c].max(1)
rmax=simb.max(1,keepdims=True); rmin=np.where(simb>-8,simb,np.inf).min(1,keepdims=True)
emit=np.where(simb>-8,(rmax-simb)/(rmax-rmin+1e-6),1.0)
def dp(em,lam=0.2):
    pen=lam*np.abs(bins[:,None]-bins[None]); cost=np.full(NB,1e9); cost[0]=em[0,0]; bp=np.zeros((n,NB),int)
    for j in range(1,n):
        tr=cost[None,:]+pen; k=tr.argmin(1); cost=em[j]+tr[np.arange(NB),k]; bp[j]=k
    cost[NB-1]-=2; path=np.zeros(n,int); path[-1]=cost.argmin()
    for j in range(n-2,-1,-1): path[j]=bp[j+1,path[j+1]]
    return path
path=dp(emit); Wb=8; tcc=np.zeros(n)
for i in range(n):
    lo=max(0,path[i]-Wb); hi=min(NB,path[i]+Wb+1); ss=simb[i,lo:hi].copy()
    if (ss<=-8).all(): tcc[i]=bins[path[i]]; continue
    ss[ss<=-8]=ss[ss>-8].min(); w=np.exp((ss-ss.max())/0.03); w/=w.sum(); tcc[i]=(w*bins[lo:hi]).sum()
def med(a,w=9): h=w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
tcc=med(tcc)
# AE absolute_advantage (AE 直接输出的相对value, 用于正负标签)
ae_adv=pd.read_parquet(Q5/"data"/f"chunk-{TEST//csQ:03d}"/f"episode_{TEST:06d}.parquet")["absolute_advantage"].to_numpy().astype(float)
NF=min(n,len(crave),len(ae),len(ae_adv)); tcc,crave,ae,ae_adv=tcc[:NF],crave[:NF],ae[:NF],ae_adv[:NF]
# 相对value(advantage = V(t+W)-V(t)); negative = adv < -eps (排除平台≈0)
def adv(v): return np.array([v[min(i+W,len(v)-1)]-v[i] for i in range(len(v))])
EPS=1e-3
tcc_adv=adv(tcc); neg_tcc=tcc_adv<-EPS; neg_ae=ae_adv<-EPS
print(f"ep{TEST}: TCC neg%{neg_tcc.mean()*100:.0f} | AE neg%{neg_ae.mean()*100:.0f}",flush=True)

VID=BASE/"videos"/f"chunk-{TEST//csB:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4"; OUT=R/"temp/labels_ep2047_sync.mp4"
MAXSIDE=440
def stream(p):
    c=av.open(str(p))
    for f in c.decode(video=0):
        sc=min(1.0,MAXSIDE/max(f.height,f.width)); g=f.reformat(width=int(f.width*sc)//2*2,height=int(f.height*sc)//2*2,format="rgb24") if sc<1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
f0=next(stream(VID)); x=np.arange(NF)
fig=plt.figure(figsize=(11,9.6)); gs=fig.add_gridspec(3,1,height_ratios=[1.4,1.0,1.0],hspace=0.30)
axc=fig.add_subplot(gs[0]); axc.axis("off"); im=axc.imshow(f0); ttl=axc.set_title("",fontsize=11)
# 面板1: TCC 连续 value, negative 段红色标记
ax1=fig.add_subplot(gs[1])
ax1.plot(x,tcc,color="#2ca02c",lw=2.0)
ax1.fill_between(x,0,1.1,where=neg_tcc,color="#d62728",alpha=0.22,step="mid")
cur1=ax1.axvline(0,color="k",lw=1.1); dot1,=ax1.plot([0],[tcc[0]],"o",color="#2ca02c",ms=7,mec="k")
ax1.set_xlim(0,NF); ax1.set_ylim(0,1.1); ax1.set_ylabel("TCC 连续 value"); ax1.grid(alpha=.25)
ax1.set_title(f"TCC 连续(DP): 相对value<0(negative)红色标记 — neg {neg_tcc.mean()*100:.0f}%(稀疏=真退步)",fontsize=10)
# 面板2: AWBC-AE value, negative 段红色标记
ax2=fig.add_subplot(gs[2],sharex=ax1)
ax2.plot(x,ae,color="#1f77b4",lw=1.6)
ax2.fill_between(x,0,1.1,where=neg_ae,color="#d62728",alpha=0.22,step="mid")
cur2=ax2.axvline(0,color="k",lw=1.1); dot2,=ax2.plot([0],[ae[0]],"o",color="#1f77b4",ms=7,mec="k")
ax2.set_xlim(0,NF); ax2.set_ylim(0,1.1); ax2.set_ylabel("AWBC-AE value"); ax2.set_xlabel("frame(30Hz)"); ax2.grid(alpha=.25)
ax2.set_title(f"AWBC pi0-AE: absolute_advantage<0(negative)红色标记 — neg {neg_ae.mean()*100:.0f}%(密集碎裂=噪声)",fontsize=10)
fig.suptitle(f"kai0_base ep{TEST}: TCC连续 vs AWBC-AE 的 negative 标签对比 × 画面同步",fontsize=12,y=0.98)
oc=av.open(str(OUT),mode="w"); st=oc.add_stream("libx264",rate=30); done=None; t=0
for img in stream(VID):
    if t>=NF: break
    im.set_data(img)
    ttl.set_text(f"frame {t}/{NF}  TCC={'NEG' if neg_tcc[t] else 'pos'}  AE={'NEG' if neg_ae[t] else 'pos'}")
    for cu in (cur1,cur2): cu.set_xdata([t,t])
    dot1.set_data([t],[tcc[t]]); dot2.set_data([t],[ae[t]])
    fig.canvas.draw()
    arr=np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[...,:3])
    if done is None: H,Wd=arr.shape[:2]; H-=H%2; Wd-=Wd%2; st.width,st.height,st.pix_fmt=Wd,H,"yuv420p"; st.options={"crf":"21"}; done=True
    for pkt in st.encode(av.VideoFrame.from_ndarray(arr[:H,:Wd],format="rgb24")): oc.mux(pkt)
    t+=1
    if t%600==0: print(f"  {t}/{NF}",flush=True)
for pkt in st.encode(): oc.mux(pkt)
oc.close(); print(f"SAVED {OUT} {Wd}x{H} {t}f",flush=True); print("DONE")
