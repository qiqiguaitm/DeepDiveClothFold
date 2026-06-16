"""ep2047 最细粒度连续value × 视频同步: 上=top_head帧, 下=连续value曲线+游标。
从 _sim_ep2047.npz 重算 NB=1001 连续value(CPU), 渲 mp4。
"""
import numpy as np, av, matplotlib, os
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from pathlib import Path
_sh=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"]=["SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
R=Path("/vePFS/tim/workspace/deepdive_kai0"); FR=R/"temp/tcc_e2e_frames/kai0base"; BASE=R/"kai0/data/Task_A/kai0_base"; TEST=2047
import json; csB=json.load(open(BASE/"meta/info.json"))["chunks_size"]
z=np.load(R/"temp/_sim_ep2047.npz"); sim=z["sim"].astype(np.float32); n=sim.shape[0]
eps=sorted(int(p.stem[2:]) for p in FR.glob("ep*.npz")); REFS=[e for e in eps if e!=TEST][:30]
bankt=np.concatenate([np.arange(L)/max(1,L-1) for L in [len(np.load(FR/f"ep{e}.npz")["frames"]) for e in REFS]])
NB=1001; bins=np.linspace(0,1,NB); binid=np.clip((bankt*(NB-1)).round().astype(int),0,NB-1)
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
path=dp(emit); Wb=40; cont=np.zeros(n)
for i in range(n):
    lo=max(0,path[i]-Wb); hi=min(NB,path[i]+Wb+1); s=simb[i,lo:hi].copy()
    if (s<=-8).all(): cont[i]=bins[path[i]]; continue
    s[s<=-8]=s[s>-8].min(); w=np.exp((s-s.max())/0.03); w/=w.sum(); cont[i]=(w*bins[lo:hi]).sum()
def med(a,w=3): h=w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
cont=med(cont,3); print("cont len",n,"end",round(float(cont[-1]),2),flush=True)

VID=BASE/"videos"/f"chunk-{TEST//csB:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4"; OUT=R/"temp/finest_value_ep2047_sync.mp4"
MAXSIDE=480
def stream(p):
    c=av.open(str(p))
    for f in c.decode(video=0):
        s=min(1.0,MAXSIDE/max(f.height,f.width)); g=f.reformat(width=int(f.width*s)//2*2,height=int(f.height*s)//2*2,format="rgb24") if s<1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
f0=next(stream(VID)); x=np.arange(n)
fig=plt.figure(figsize=(11,8.4)); gs=fig.add_gridspec(2,1,height_ratios=[1.5,1.0],hspace=0.18)
axc=fig.add_subplot(gs[0]); axc.axis("off"); im=axc.imshow(f0); ttl=axc.set_title("",fontsize=11)
axv=fig.add_subplot(gs[1]); axv.plot(x,cont,color="#2ca02c",lw=1.6); axv.axhline(1,color="#ddd",ls=":",lw=1)
cur=axv.axvline(0,color="k",lw=1.3); dot,=axv.plot([0],[cont[0]],"o",color="#2ca02c",ms=8,mec="k")
axv.set_xlim(0,n); axv.set_ylim(-0.02,1.05); axv.set_xlabel("frame(30Hz)"); axv.set_ylabel("value"); axv.grid(alpha=.25)
fig.suptitle(f"kai0_base ep{TEST}: 最细粒度连续value (端到端TCC+细binDP, 286级) × 视频同步",fontsize=12,y=0.97)
oc=av.open(str(OUT),mode="w"); st=oc.add_stream("libx264",rate=30); done=None; t=0
for img in stream(VID):
    if t>=n: break
    im.set_data(img); ttl.set_text(f"frame {t}/{n}  value={cont[t]:.3f}")
    cur.set_xdata([t,t]); dot.set_data([t],[cont[t]]); fig.canvas.draw()
    arr=np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[...,:3])
    if done is None: H,W=arr.shape[:2]; H-=H%2; W-=W%2; st.width,st.height,st.pix_fmt=W,H,"yuv420p"; st.options={"crf":"21"}; done=True
    for pkt in st.encode(av.VideoFrame.from_ndarray(arr[:H,:W],format="rgb24")): oc.mux(pkt)
    t+=1
    if t%600==0: print(f"  {t}/{n}",flush=True)
for pkt in st.encode(): oc.mux(pkt)
oc.close(); print(f"SAVED {OUT} {W}x{H} {t}f",flush=True); print("DONE")
