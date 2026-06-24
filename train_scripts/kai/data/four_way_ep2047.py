"""ep2047 四方连续 value 对比(30Hz): frozen-TCC / 端到端-TCC / AWBC-AE / 簇空间距离(修bug)。
距离法 bug 修复: 距离集补上 start原型(P=0)/end原型(P=1)(原 IDW 缺端点锚→末帧首尾混淆判低)。
frozen/AE 来自 _solve_ep2047_30hz.npz, e2e 来自 _e2e30_ep2047.npz, 距离法用 30Hz 特征现算。
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdf5_v24_eval import loadep, mkp
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0)
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = REPO / "temp/crave_kai0bd/feat_cache"; EP30 = REPO / "temp/_ep2047_30hz"; TEST = 2047
eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))

# ---- 复刻 V2.4 挖矿(同 build_model)取 milestone 质心 C/Pk + 端点原型 ----
Sall = [loadep(FC, e)[2] for e in eps]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
def emb(a_, r_, s_):
    an = a_/np.linalg.norm(a_, axis=1, keepdims=True); rn = r_/np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(s_)-PMU)/PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)
A,R,S,T,E,SP,EP_ = [],[],[],[],[],[],[]
for e in eps:
    aa,rr,st,n = loadep(FC, e); g = emb(aa,rr,st)
    A.append(aa);R.append(rr);S.append(st);T.append(np.arange(n)/max(1,n-1));E.append(np.full(n,e));SP.append(g[:2]);EP_.append(g[-2:])
A=np.concatenate(A);R=np.concatenate(R);S=np.concatenate(S);T=np.concatenate(T);E=np.concatenate(E)
G=emb(A,R,S); km=KMeans(96,n_init=2,random_state=0).fit(G); lab=km.labels_; allC=km.cluster_centers_
N=len(set(E.tolist())); tpos=np.array([T[lab==c].mean() if (lab==c).any() else .5 for c in range(96)])
Pstart={}
for e in sorted(set(E.tolist())):
    m=np.where(E==e)[0][:3]; nnz=np.linalg.norm(G[m][:,None]-allC[None],axis=2).argmin(1); Pstart[e]=float(np.median(tpos[nnz]))
cov_n=np.array([min(1,(len(set(E[lab==c].tolist()))+sum(1 for e in Pstart if Pstart[e]>tpos[c]+0.1))/N) for c in range(96)])
bk=np.linspace(0,1,11); sel=[]
for b in range(10):
    inb=[c for c in range(96) if bk[b]<=tpos[c]<bk[b+1]]
    if inb: sel+=sorted(inb,key=lambda c:-cov_n[c])[:2]
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
order=sorted(sel,key=lambda c:Pk[c]); Cms=allC[order]; Pk_ord=np.array([Pk[c] for c in order])
startK=KMeans(8,n_init=2,random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK=KMeans(8,n_init=2,random_state=0).fit(np.concatenate(EP_)).cluster_centers_

# ---- 距离法(修bug): 锚集 = start原型(P=0) + milestone(Pk) + end原型(P=1) ----
d30=np.load(EP30/f"ep{TEST}.npz"); a30,r30,s30=d30["armmask"],d30["raw"],d30["state"]
n30=min(len(a30),len(r30),len(s30)); F30=emb(a30[:n30],r30[:n30],s30[:n30])
# 锚中心与进度值
anchorsC=np.concatenate([startK, Cms, endK]); anchorsP=np.concatenate([np.zeros(len(startK)), Pk_ord, np.ones(len(endK))])
d=np.linalg.norm(F30[:,None]-anchorsC[None],axis=2)              # (n,锚)
tau=0.35*np.median(np.sort(d,axis=1)[:,1])  # 自适应温度(锐)
w=np.exp(-(d - d.min(1,keepdims=True))/ (tau+1e-6)); w/=w.sum(1,keepdims=True)
v_dist_raw=(w*anchorsP[None]).sum(1)
def med(a,w=27): h=w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
v_dist=med(v_dist_raw)

# ---- 载入其余三方(30Hz) ----
s=np.load(REPO/"temp/_solve_ep2047_30hz.npz"); frozen=s["tcc"]; ae=s["ae"]
e2e=np.load(REPO/"temp/_e2e30_ep2047.npz")["e2e"]
NF=min(len(v_dist),len(frozen),len(ae),len(e2e)); v_dist,frozen,ae,e2e=v_dist[:NF],frozen[:NF],ae[:NF],e2e[:NF]; x=np.arange(NF)
W=50
def mono(v): return np.mean(np.diff(v)>=-1e-6)
def aden(v): a=np.array([v[min(i+W,len(v)-1)]-v[i] for i in range(len(v))]); return np.mean(np.abs(np.clip(a,-1,1))>1e-3)
def rough(v): return np.mean(np.abs(np.diff(v,2)))
M={"端到端TCC":e2e,"frozen-TCC":frozen,"簇距离(修bug)":v_dist,"AWBC pi0-AE":ae}
print("方法            end   单调   adv密度  抖动")
for k,v in M.items(): print(f"{k:<14}{v[-1]:.2f}  {mono(v):.0%}   {aden(v):.0%}    {rough(v):.4f}")

col={"端到端TCC":"#2ca02c","frozen-TCC":"#888","簇距离(修bug)":"#ff7f0e","AWBC pi0-AE":"#d62728"}
fig,ax=plt.subplots(figsize=(13,5.2))
for k,v in M.items():
    ax.plot(x,v,color=col[k],lw=2.0 if "端到端" in k else 1.6,label=f"{k} (end{v[-1]:.2f} 单调{mono(v):.0%} adv{aden(v):.0%} 抖{rough(v):.4f})")
ax.axhline(1,color="#bbb",ls=":",lw=1); ax.set_xlim(0,NF); ax.set_ylim(-0.05,1.13)
ax.set_xlabel("frame (30Hz)"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.legend(fontsize=9.5,loc="upper left")
ax.set_title(f"kai0_base ep{TEST} 四方连续 value 对比(30Hz): 端到端TCC / frozen-TCC / 簇距离(修bug) / AWBC-AE",fontsize=12)
out=REPO/"docs/visualization/cross_episode_recurrence_value/four_way_ep2047.png"
fig.tight_layout(); fig.savefig(out,dpi=125); print("SAVED",out); print("DONE")
