"""把时序证据累积(Viterbi-DP)搬进端到端TCC的readout: 在 query×进度 相似度场上跑DP,
替代逐帧独立argmax。验证: 能否像CRAVE一样消fold凹口、同时保留持续真回退。复用 _e2e_kai0base_model.pt。
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, av, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from transformers import AutoModel
_sh=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"]=["SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
R=Path("/vePFS/tim/workspace/deepdive_kai0"); BASE=R/"kai0/data/Task_A/kai0_base"
FRD=R/"temp/tcc_e2e_frames/kai0base"; csB=json.load(open(BASE/"meta/info.json"))["chunks_size"]; TEST=2047; dev="cuda"
IMEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev); ISTD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
eps=sorted(int(p.stem[2:]) for p in FRD.glob("ep*.npz")); TRAIN=[e for e in eps if e!=TEST]
def prop(e,n,stride=10):
    st=np.stack(pd.read_parquet(BASE/"data"/f"chunk-{e//csB:03d}"/f"episode_{e:06d}.parquet",columns=["observation.state"])["observation.state"].to_numpy())
    st=st[np.minimum(np.arange(n)*stride,len(st)-1)] if stride>1 else st[:n]
    return np.concatenate([st,np.vstack([np.zeros((1,14)),np.diff(st,axis=0)])],1).astype(np.float32)
IMG,PRr={},{}
for e in TRAIN: IMG[e]=np.load(FRD/f"ep{e}.npz")["frames"]; PRr[e]=prop(e,len(IMG[e]))
allp=np.concatenate([PRr[e] for e in TRAIN]); MU,SD=allp.mean(0),allp.std(0)+1e-8
def pn(p): q=(p-MU)/SD; return (q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-9)).astype(np.float32)
for e in PRr: PRr[e]=pn(PRr[e])
bb=AutoModel.from_pretrained("facebook/dinov2-small").to(dev); head=nn.Sequential(nn.Linear(412,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,128)).to(dev)
ck=torch.load(R/"temp/_e2e_kai0base_model.pt"); bb.load_state_dict(ck["bb"]); head.load_state_dict(ck["head"]); bb.eval(); head.eval()
@torch.no_grad()
def embed(fr,pr):
    o=[]
    for b in range(0,len(fr),128):
        x=torch.from_numpy(fr[b:b+128]).to(dev).permute(0,3,1,2).float()/255.; x=(x-IMEAN)/ISTD
        with torch.autocast("cuda",dtype=torch.bfloat16): v=bb(x).last_hidden_state[:,1:].mean(1).float()
        v=v/(v.norm(dim=-1,keepdim=True)+1e-9); o.append(head(torch.cat([v,torch.from_numpy(pr[b:b+128]).to(dev)],-1)).cpu().numpy())
    z=np.concatenate(o); return z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-9)
# 参考帧库: 嵌入 + 归一化时间
REFS=TRAIN[:30]; bank=[]; bankt=[]
for e in REFS:
    z=embed(IMG[e],PRr[e]); bank.append(z); bankt.append(np.arange(len(z))/max(1,len(z)-1))
bank=np.concatenate(bank); bankt=np.concatenate(bankt)
# query: ep2047 全2629帧
def crop(im): s=224/min(im.shape[:2]); g=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s))); h,w=g.shape[:2]; return g[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224]
c=av.open(str(BASE/"videos"/f"chunk-{TEST//csB:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4")); frs=np.stack([crop(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]).astype(np.uint8); c.close()
zq=embed(frs, pn(prop(TEST,len(frs),stride=1)[:len(frs)])); n=len(zq)
# 相似度场 -> emit[i,bin]: query i 对"时间落在bin b的参考帧"的最佳相似度
NB=21; bins=np.linspace(0,1,NB); binid=np.clip((bankt*(NB-1)).round().astype(int),0,NB-1)
sim=zq@bank.T                                   # (n, bank)
simbin=np.full((n,NB),-1.0)
for b in range(NB):
    cols=np.where(binid==b)[0]
    if len(cols): simbin[:,b]=sim[:,cols].max(1)
# emit 逐帧归一化: 最优bin代价0, 其余按相对差距(消除TCC相似度场过平的问题)
rmax=simbin.max(1,keepdims=True); rmin=simbin.min(1,keepdims=True)
emit=(rmax-simbin)/(rmax-rmin+1e-6)              # ∈[0,1] 每帧, 缺失bin→1(高代价)
def dpHB(em,lam=0.5):
    pen=lam*np.abs(bins[:,None]-bins[None]); NF=len(em)
    cost=np.full(NB,1e9); cost[0]=em[0,0]; bp=np.zeros((NF,NB),int)
    for j in range(1,NF):
        tr=cost[None,:]+pen; k=tr.argmin(1); cost=em[j]+tr[np.arange(NB),k]; bp[j]=k
    cost[NB-1]-=2; path=np.zeros(NF,int); path[-1]=cost.argmin()
    for j in range(NF-2,-1,-1): path[j]=bp[j+1,path[j+1]]
    return bins[path]
def med(a,w=27): h=w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
v_dp=med(dpHB(emit),9)
# 对照: 现有 argmax-median readout
amax=np.median(np.stack([ (np.arange(len(bankt))[binid==binid]) for _ in [0]]),0) if False else None
v_argmax=med(bankt[sim.argmax(1)])
def mono(v): return np.mean(np.diff(v)>=-1e-6)
def dens(v,W=50): a=np.array([v[min(i+W,len(v)-1)]-v[i] for i in range(len(v))]); return np.mean(np.abs(np.clip(a,-1,1))>1e-3)
def dip(v): return v[int(.75*n):].min()
def rough(v): return np.mean(np.abs(np.diff(v,2)))
for nm,v in [("argmax(现有)",v_argmax),("DP-readout(时序证据)",v_dp)]:
    print(f"{nm:<18} end{v[-1]:.2f} 单调{mono(v):.0%} adv密度{dens(v):.0%} 末段最低{dip(v):.2f} 抖{rough(v):.4f}")
x=np.arange(n)
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(x,v_argmax,color="#bbb",lw=1.5,label=f"现有 argmax-readout (单调{mono(v_argmax):.0%} fold凹口→{dip(v_argmax):.2f})")
ax.plot(x,v_dp,color="#d62728",lw=2.2,label=f"DP-readout 时序证据累积 (单调{mono(v_dp):.0%} adv{dens(v_dp):.0%} 末段→{dip(v_dp):.2f})")
ax.axhline(1,color="#ddd",ls=":",lw=1); ax.set_xlim(0,n); ax.set_ylim(-0.05,1.12)
ax.set_xlabel("frame(30Hz)"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.legend(fontsize=9,loc="upper left")
ax.set_title("端到端TCC内置时序证据累积: 在TCC相似度场上跑Viterbi-DP替代逐帧argmax (消fold凹口/保留持续回退)",fontsize=11)
out=R/"docs/visualization/cross_episode_recurrence_value/e2e_dp_readout_ep2047.png"
fig.tight_layout(); fig.savefig(out,dpi=125); print("SAVED",out); print("DONE")
