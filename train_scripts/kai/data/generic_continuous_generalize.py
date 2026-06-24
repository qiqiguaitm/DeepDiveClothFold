"""跨数据集泛化: 连续 value 方法(CRAVE离散 + frozen-TCC + 细binDP连续)在任意 feat_cache 上,
挑最长 episode 输出 离散vs连续 value。用法: python generic_continuous_generalize.py --feat <cache> --tag <name>
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, matplotlib, torch, torch.nn as nn
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdf5_v24_eval import build_model, loadep, mkp
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss
_sh=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"]=["SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
np.random.seed(0); torch.manual_seed(0)
ap=argparse.ArgumentParser(); ap.add_argument("--feat",required=True); ap.add_argument("--tag",required=True)
a=ap.parse_args(); FC=Path(a.feat)
eps=sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
lens={e:loadep(FC,e)[3] for e in eps}; TEST=max(lens,key=lambda e:lens[e])
print(f"[{a.tag}] {len(eps)} eps; 最长 ep{TEST} ({lens[TEST]}帧@3Hz≈{lens[TEST]*10/30:.0f}s)",flush=True)
# 离散 CRAVE
value,_=build_model(FC,eps,eps)
aa,rr,st,nq=loadep(FC,TEST); disc=value(aa,rr,st)
# emb 复刻 + frozen-TCC
Sall=[loadep(FC,e)[2] for e in eps]; Pm=mkp(np.concatenate(Sall)); PMU,PSD=Pm.mean(0),Pm.std(0)+1e-8
def emb(a_,r_,s_):
    an=a_/np.linalg.norm(a_,axis=1,keepdims=True); rn=r_/np.linalg.norm(r_,axis=1,keepdims=True)
    Pn=((mkp(s_)-PMU)/PSD); Pn/=np.linalg.norm(Pn,axis=1,keepdims=True)
    return np.concatenate([rn,an,Pn],1).astype(np.float32)
Gd={e:emb(*loadep(FC,e)[:3]) for e in eps}
class Head(nn.Module):
    def __init__(s,d): super().__init__(); s.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,128))
    def forward(s,x): return s.net(x)
head=Head(Gd[eps[0]].shape[1]); opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-5)
for step in range(1200):
    bes=list(np.random.choice(eps,min(8,len(eps)),replace=False)); E_,I_,L_=[],[],[]
    for e in bes:
        f=Gd[e]; m=len(f); ix=np.sort(np.random.choice(m,size=32,replace=m<32))
        E_.append(head(torch.from_numpy(f[ix]))); I_.append(torch.from_numpy(ix).long()); L_.append(m)
    loss=compute_tcc_loss(embs=torch.stack(E_),idxs=torch.stack(I_),seq_lens=torch.tensor(L_),stochastic_matching=False,
        normalize_embeddings=True,loss_type="regression_mse",similarity_type="l2",num_cycles=20,cycle_length=2,
        temperature=0.1,label_smoothing=0.1,variance_lambda=0.001,huber_delta=0.1,normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step()
head.eval()
def hemb(x):
    with torch.no_grad(): z=head(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32))).numpy()
    return z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-9)
REFS=[e for e in eps if e!=TEST][:30]
bank=np.concatenate([hemb(Gd[e]) for e in REFS]); bankt=np.concatenate([np.arange(len(Gd[e]))/max(1,len(Gd[e])-1) for e in REFS])
zq=hemb(emb(aa,rr,st)); sim=zq@bank.T
NB=201; bins=np.linspace(0,1,NB); binid=np.clip((bankt*(NB-1)).round().astype(int),0,NB-1)
simb=np.full((nq,NB),-9.0)
for b in range(NB):
    c=np.where(binid==b)[0]
    if len(c): simb[:,b]=sim[:,c].max(1)
rmax=simb.max(1,keepdims=True); rmin=np.where(simb>-8,simb,np.inf).min(1,keepdims=True)
emit=np.where(simb>-8,(rmax-simb)/(rmax-rmin+1e-6),1.0)
def dp(em,lam=0.2):
    pen=lam*np.abs(bins[:,None]-bins[None]); cost=np.full(NB,1e9); cost[0]=em[0,0]; bp=np.zeros((nq,NB),int)
    for j in range(1,nq):
        tr=cost[None,:]+pen; k=tr.argmin(1); cost=em[j]+tr[np.arange(NB),k]; bp[j]=k
    cost[NB-1]-=2; path=np.zeros(nq,int); path[-1]=cost.argmin()
    for j in range(nq-2,-1,-1): path[j]=bp[j+1,path[j+1]]
    return path
path=dp(emit); Wb=8; cont=np.zeros(nq)
for i in range(nq):
    lo=max(0,path[i]-Wb); hi=min(NB,path[i]+Wb+1); s=simb[i,lo:hi].copy()
    if (s<=-8).all(): cont[i]=bins[path[i]]; continue
    s[s<=-8]=s[s>-8].min(); w=np.exp((s-s.max())/0.03); w/=w.sum(); cont[i]=(w*bins[lo:hi]).sum()
def med(x,w=5): h=w//2; return np.array([np.median(x[max(0,j-h):j+h+1]) for j in range(len(x))])
cont=med(cont)
def mono(v): return np.mean(np.diff(v)>=-1e-6)
def dens(v,W=15): A=np.array([v[min(i+W,len(v)-1)]-v[i] for i in range(len(v))]); return np.mean(np.abs(np.clip(A,-1,1))>1e-3)
from scipy.stats import pearsonr
ct=np.arange(nq)/(nq-1)
print(f"[{a.tag}] 连续: end{cont[-1]:.2f} 单调{mono(cont):.0%} adv{dens(cont):.0%} corr(连续,时间){pearsonr(cont,ct)[0]:.3f}; 离散 end{disc[-1]:.2f} 单调{mono(disc):.0%}",flush=True)
x=np.arange(nq)*10
fig,ax=plt.subplots(figsize=(12,4.4))
ax.step(x,disc,where="post",color="#1f77b4",lw=1.5,alpha=.7,label=f"离散CRAVE (end{disc[-1]:.2f} 单调{mono(disc):.0%})")
ax.plot(x,cont,color="#2ca02c",lw=2.0,label=f"连续 TCC+DP (end{cont[-1]:.2f} 单调{mono(cont):.0%} adv{dens(cont):.0%} corr时间{pearsonr(cont,ct)[0]:.2f})")
ax.axhline(1,color="#ddd",ls=":",lw=1); ax.set_xlim(0,x[-1]); ax.set_ylim(-0.05,1.1)
ax.set_xlabel("frame(30Hz)"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.legend(fontsize=9,loc="upper left")
ax.set_title(f"跨数据泛化 [{a.tag}] 最长 ep{TEST}({lens[TEST]*10}f): 离散CRAVE vs 连续TCC+DP",fontsize=11)
out=Path("docs/visualization/cross_episode_recurrence_value")/f"generalize_continuous_{a.tag}.png"
fig.tight_layout(); fig.savefig(out,dpi=125); print("SAVED",out,flush=True)
np.savez(f"temp/_gen_{a.tag}.npz", cont=cont, disc=disc, test=TEST, nframes=lens[TEST]); print("DONE")
