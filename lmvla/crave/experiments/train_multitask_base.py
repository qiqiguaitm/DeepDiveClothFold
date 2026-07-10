import numpy as np, time, torch, torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); DEV='cuda:0'; rng=np.random.RandomState(0); torch.manual_seed(0)
# (名, base bank, fps, task_group, cap_eps)  原始频率 native; 均衡 kai/xvla cap 1000
DS=[('kai','kai_dinov3base',30.,0,1000),('vis','vis_dinov3base',30.,0,10000),
    ('xvla','xvla_dinov3base_full',30.,1,1000),('coffee','coffee_dinov3base',50.,2,10000)]
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
def load_bank(bank):
    d=REPO/'temp'/bank; idx=np.load(d/'index.npz'); E=idx['E'];FR=idx['FR'];N=len(E);feat=np.zeros((N,768),np.float16)
    for sh in sorted(d.glob('shard_*.npz')):
        s=np.load(sh); g=s['gidx']; v=s['valid'] if 'valid' in s else np.ones(len(g),bool); feat[g[v]]=s['feat'][v]
    return E,FR,feat
def cc(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else np.nan
print('加载 base bank + cap...',flush=True); t0=time.time()
RAW={}; sub=[]
for name,bank,fps,tg,cap in DS:
    E,FR,feat=load_bank(bank); eps=sorted(np.unique(E).tolist())
    if len(eps)>cap: eps=[eps[i] for i in sorted(rng.choice(len(eps),cap,replace=False))]
    keep=np.isin(E,eps); E=E[keep]; FR=FR[keep]; feat=feat[keep]
    # per-ep T (归一时间)
    T=np.zeros(len(E),np.float32)
    for e in eps:
        m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
    RAW[name]=(E,T,feat,tg,fps,eps); ss=rng.choice(len(feat),min(20000,len(feat)),replace=False); sub.append(l2(feat[ss].astype(np.float32)))
    print(f'  {name}: {len(eps)} eps {len(E)} frames ({time.time()-t0:.0f}s)',flush=True)
pca=PCA(128,random_state=0).fit(np.concatenate(sub)); pm=pca.mean_.astype(np.float32); pc=pca.components_.astype(np.float32)
print(f'shared PCA768→128 ({time.time()-t0:.0f}s)',flush=True)
def daw(F,C,P,lam):  # 双锚 Viterbi(PCA128 空间, F/C 已L2)
    sC=l2(F[:3].mean(0)[None])[0]; eC=l2(F[-3:].mean(0)[None])[0]; C2=np.vstack([C,sC,eC]); Pp=np.concatenate([P,[0.],[1.]])
    bins=np.unique(np.concatenate([[0.],Pp,[1.]])); nb=len(bins); cb=[int(np.searchsorted(bins,v)) for v in Pp]; pen=lam*np.abs(bins[:,None]-bins[None])
    de=np.linalg.norm(F[:,None]-C2[None],axis=2); em=np.full((len(F),nb),1e3)
    for ti in range(len(Pp)): em[:,cb[ti]]=np.minimum(em[:,cb[ti]],de[:,ti])
    cost=np.full(nb,1e9); cost[0]=em[0,0]; BP=np.zeros((len(F),nb),int)
    for j in range(1,len(F)):
        tr=cost[None,:]+pen; kk=tr.argmin(1); cost=em[j]+tr[np.arange(nb),kk]; BP[j]=kk
    si=nb-1; path=np.zeros(len(F),int); path[-1]=si
    for j in range(len(F)-2,-1,-1): si=BP[j+1][si]; path[j]=si
    step=bins[path]; segs=[]; a=0
    for t in range(1,len(step)):
        if step[t]!=step[t-1]: segs.append((a,t-1,step[t-1])); a=t
    segs.append((a,len(step)-1,step[-1])); reps=[]
    for i0,i1,val in segs:
        cand=[ti for ti in range(len(Pp)) if abs(Pp[ti]-val)<1e-9]; fr=np.arange(i0,i1+1); bd=1e18; bf=i0
        for ti in cand:
            dd=np.linalg.norm(F[fr]-C2[ti],axis=1); k=int(dd.argmin())
            if dd[k]<bd: bd=dd[k]; bf=fr[k]
        reps.append((bf,float(val)))
    if reps[0][0]!=0: reps=[(0,float(step[0]))]+reps
    if reps[-1][0]!=len(step)-1: reps=reps+[(len(step)-1,float(step[-1]))]
    rf=np.array([r[0] for r in reps]); rv=np.array([r[1] for r in reps]); keep=np.concatenate([[True],np.diff(rf)>0])
    return np.interp(np.arange(len(step)),rf[keep],rv[keep]).astype(np.float32)
# 内联 milestone + teacher per 数据集
DATA={}
for name,(E,T,feat,tg,fps,eps) in RAW.items():
    F128=l2((l2(feat.astype(np.float32))-pm)@pc.T); NC=len(eps)
    fit=rng.choice(len(F128),min(80000,len(F128)),replace=False)
    bg=BayesianGaussianMixture(n_components=40,covariance_type='diag',weight_concentration_prior=1e-2,max_iter=120,random_state=0).fit(F128[fit])
    labs=bg.predict(F128); C=[]; P=[]
    for k in range(40):
        m=labs==k
        if m.sum()<20: continue
        if len(set(E[m].tolist()))/NC>=0.5: C.append(F128[m].mean(0)); P.append(float(np.median(T[m])))
    C=l2(np.array(C,np.float32)); P=np.array(P); lam=16.*fps/3.
    recs=[]; corrs=[]
    for e in eps:
        m=np.where(E==e)[0]; o=m[np.argsort(FR[E==e] if False else np.arange(len(m)))]  # E already sorted per ep? rebuild
    # 重新按 ep 收集(FR 排序)
    for e in eps:
        idx=np.where(E==e)[0]; f=F128[idx]; t=T[idx]
        v=daw(f,C,P,lam); recs.append([f,v,t,tg]); corrs.append(cc(v,t))
    DATA[name]=recs; print(f'{name}: M={len(C)} milestones, teacher-vs-T corr={np.nanmean(corrs):.3f} ({time.time()-t0:.0f}s)',flush=True)
# split + train
TR={};EV={}
for n in DATA:
    r=DATA[n]; rng.shuffle(r); k=int(len(r)*0.85); TR[n]=r[:k]; EV[n]=r[k:]
print('train/eval:',{n:(len(TR[n]),len(EV[n])) for n in DATA},flush=True)
NT=3
class G(nn.Module):
    def __init__(s,h=256,L=2,emb=0):
        super().__init__(); s.emb=nn.Embedding(NT,emb) if emb>0 else None; s.g=nn.GRU(128+(emb if emb>0 else 0),h,L,batch_first=True); s.head=nn.Sequential(nn.Linear(h,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x,tk,ln):
        if s.emb is not None: x=torch.cat([x,s.emb(tk)[:,None,:].expand(-1,x.shape[1],-1)],-1)
        p=nn.utils.rnn.pack_padded_sequence(x,ln.cpu(),batch_first=True,enforce_sorted=False); o,_=s.g(p); o,_=nn.utils.rnn.pad_packed_sequence(o,batch_first=True); return torch.sigmoid(s.head(o)).squeeze(-1)
def batches(pool,bs=24):
    ix=sorted(range(len(pool)),key=lambda i:len(pool[i][0]))
    for k in range(0,len(ix),bs):
        gp=[pool[i] for i in ix[k:k+bs]]; L=max(len(a[0]) for a in gp); B=len(gp)
        X=np.zeros((B,L,128),np.float32);Y=np.zeros((B,L),np.float32);M=np.zeros((B,L),np.float32);tk=np.zeros(B,int);ln=np.zeros(B,int)
        for b,(f,v,t,tg) in enumerate(gp): n=len(f);X[b,:n]=f;Y[b,:n]=v;M[b,:n]=1;tk[b]=tg;ln[b]=n
        yield torch.tensor(X,device=DEV),torch.tensor(Y,device=DEV),torch.tensor(M,device=DEV),torch.tensor(tk,device=DEV),torch.tensor(ln)
@torch.no_grad()
def ev(net):
    net.eval(); o={}
    for n in DATA:
        c=[np.corrcoef(net(torch.tensor(f[None],device=DEV),torch.tensor([tg],device=DEV),torch.tensor([len(f)]))[0].cpu().numpy(),v)[0,1] for f,v,t,tg in EV[n]]
        o[n]=np.nanmean(c)
    return o
def train(h,L,emb,ep=40,cap=250):
    net=G(h,L,emb).to(DEV); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep)
    gr={0:TR['kai']+TR['vis'],1:TR['xvla'],2:TR['coffee']}
    for e in range(ep):
        net.train(); ep_=[]
        for g in gr: s=gr[g]; ii=rng.choice(len(s),min(cap,len(s)),replace=len(s)<cap); ep_+=[s[i] for i in ii]
        for X,Y,M,tk,ln in batches(ep_):
            pr=net(X,tk,ln); loss=(((pr-Y)**2)*M).sum()/M.sum(); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net,sum(p.numel() for p in net.parameters())
print('\n=== DINOv3-BASE 原始频率 · 3-task均衡 · 参数扫描 ===',flush=True)
for tag,(h,L) in [('h128/L2',(128,2)),('h256/L2',(256,2)),('h256/L3',(256,3)),('h384/L2',(384,2))]:
    net,p=train(h,L,0); r=ev(net)
    print(f'{tag:>9}({p/1e6:.2f}M): '+' '.join(f'{n}={r[n]:.3f}' for n in DATA)+f' mean={np.mean(list(r.values())):.3f}',flush=True)
print('DONE',flush=True)
