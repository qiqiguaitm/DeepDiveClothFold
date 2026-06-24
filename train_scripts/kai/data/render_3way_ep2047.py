"""kai0_base ep2047: 三方 value × 画面 同步视频 — TCC连续(DP修复) / milestone离散阶梯 / AWBC-AE。
TCC连续从 _sim_ep2047.npz 重算(细binDP+子bin软期望); 离散crave+ae 取自 _solve_ep2047_30hz.npz。
"""
import json
from pathlib import Path
import numpy as np, av, matplotlib, os
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
_sh=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"]=["SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
R=Path("/vePFS/tim/workspace/deepdive_kai0"); BASE=R/"kai0/data/Task_A/kai0_base"; FR=R/"temp/tcc_e2e_frames/kai0base"; TEST=2047
csB=json.load(open(BASE/"meta/info.json"))["chunks_size"]
s=np.load(R/"temp/_solve_ep2047_30hz.npz"); crave=s["crave"]; ae=s["ae"]
# --- TCC 连续: 从 sim 场重算 细binDP + 子bin软期望 (=图53/最终连续) ---
z=np.load(R/"temp/_sim_ep2047.npz"); sim=z["sim"].astype(np.float32); n=sim.shape[0]
eps=sorted(int(p.stem[2:]) for p in FR.glob("ep*.npz")); REFS=[e for e in eps if e!=TEST][:30]
bankt=np.concatenate([np.arange(L)/max(1,L-1) for L in [len(np.load(FR/f"ep{e}.npz")["frames"]) for e in REFS]])
assert len(bankt)==sim.shape[1]
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
NF=min(n,len(crave),len(ae)); tcc,crave,ae=tcc[:NF],crave[:NF],ae[:NF]
print(f"3way ep{TEST}: NF={NF} tcc_end{tcc[-1]:.2f} crave_end{crave[-1]:.2f} ae_end{ae[-1]:.2f}",flush=True)

VID=BASE/"videos"/f"chunk-{TEST//csB:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4"; OUT=R/"temp/3way_ep2047_sync.mp4"
MAXSIDE=470
def stream(p):
    c=av.open(str(p))
    for f in c.decode(video=0):
        sc=min(1.0,MAXSIDE/max(f.height,f.width)); g=f.reformat(width=int(f.width*sc)//2*2,height=int(f.height*sc)//2*2,format="rgb24") if sc<1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
f0=next(stream(VID)); x=np.arange(NF)
fig=plt.figure(figsize=(11,8.6)); gs=fig.add_gridspec(2,1,height_ratios=[1.5,1.0],hspace=0.2)
axc=fig.add_subplot(gs[0]); axc.axis("off"); im=axc.imshow(f0); ttl=axc.set_title("",fontsize=11)
axv=fig.add_subplot(gs[1])
axv.step(x,crave,where="post",color="#1f77b4",lw=1.6,alpha=.75,label="milestone 离散阶梯(CRAVE)")
axv.plot(x,tcc,color="#2ca02c",lw=2.2,label="TCC 连续(DP)")
axv.plot(x,ae,color="#d62728",lw=1.5,alpha=.85,label="AWBC pi0-AE 监督")
axv.axhline(1,color="#ddd",ls=":",lw=1)
cur=axv.axvline(0,color="k",lw=1.3)
dd,=axv.plot([0],[crave[0]],"s",color="#1f77b4",ms=6,mec="k"); dt,=axv.plot([0],[tcc[0]],"o",color="#2ca02c",ms=8,mec="k"); da,=axv.plot([0],[ae[0]],"^",color="#d62728",ms=7,mec="k")
axv.set_xlim(0,NF); axv.set_ylim(-0.05,1.12); axv.set_xlabel("frame(30Hz)"); axv.set_ylabel("value"); axv.grid(alpha=.25); axv.legend(fontsize=9,loc="upper left")
fig.suptitle(f"kai0_base ep{TEST}: TCC连续 vs milestone离散 vs AWBC-AE × 画面同步",fontsize=12,y=0.97)
oc=av.open(str(OUT),mode="w"); st=oc.add_stream("libx264",rate=30); done=None; t=0
for img in stream(VID):
    if t>=NF: break
    im.set_data(img); ttl.set_text(f"frame {t}/{NF}  TCC={tcc[t]:.2f}  离散={crave[t]:.2f}  AE={ae[t]:.2f}")
    cur.set_xdata([t,t]); dd.set_data([t],[crave[t]]); dt.set_data([t],[tcc[t]]); da.set_data([t],[ae[t]]); fig.canvas.draw()
    arr=np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[...,:3])
    if done is None: H,W=arr.shape[:2]; H-=H%2; W-=W%2; st.width,st.height,st.pix_fmt=W,H,"yuv420p"; st.options={"crf":"21"}; done=True
    for pkt in st.encode(av.VideoFrame.from_ndarray(arr[:H,:W],format="rgb24")): oc.mux(pkt)
    t+=1
    if t%600==0: print(f"  {t}/{NF}",flush=True)
for pkt in st.encode(): oc.mux(pkt)
oc.close(); print(f"SAVED {OUT} {W}x{H} {t}f",flush=True); print("DONE")
