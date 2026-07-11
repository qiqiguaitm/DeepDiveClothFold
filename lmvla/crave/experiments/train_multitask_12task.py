import numpy as np, time, torch, torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from crave.render import setup_mpl
plt=setup_mpl()
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); DEV='cuda:0'; rng=np.random.RandomState(0); torch.manual_seed(0)
DS=[('kai','kai_dinov3base',30.,0,1000),('vis','vis_dinov3base',30.,0,10000),('xvla','xvla_dinov3base_full',30.,1,1000),('coffee','coffee_dinov3base',50.,2,10000),
('cups_open','aloha_static_cups_open_dinov3base',50.,2,10000),('candy','aloha_static_candy_dinov3base',50.,2,10000),('screw_driver','aloha_static_screw_driver_dinov3base',50.,2,10000),('vinh_cup','aloha_static_vinh_cup_dinov3base',50.,2,10000),('ziploc_slide','aloha_static_ziploc_slide_dinov3base',50.,2,10000),('coffee_new','aloha_static_coffee_new_dinov3base',50.,2,10000),('pro_pencil','aloha_static_pro_pencil_dinov3base',50.,2,10000),('vinh_cup_left','aloha_static_vinh_cup_left_dinov3base',50.,2,10000)]
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
def load_bank(bank):
    d=REPO/'lmvla/crave/data'/bank; idx=np.load(d/'index.npz'); E=idx['E'];FR=idx['FR'];N=len(E);feat=np.zeros((N,768),np.float16)
    for sh in sorted(d.glob('shard_*.npz')):
        s=np.load(sh); g=s['gidx']; v=s['valid'] if 'valid' in s else np.ones(len(g),bool); feat[g[v]]=s['feat'][v]
    return E,FR,feat
def cc(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else np.nan
def daw(F,C,P,lam):
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
print('prep...',flush=True); t0=time.time(); RAW={}; sub=[]
for name,bank,fps,tg,cap in DS:
    E,FR,feat=load_bank(bank); eps=sorted(np.unique(E).tolist())
    if len(eps)>cap: eps=[eps[i] for i in sorted(rng.choice(len(eps),cap,replace=False))]
    keep=np.isin(E,eps); E=E[keep];FR=FR[keep];feat=feat[keep]; T=np.zeros(len(E),np.float32)
    for e in eps: m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
    RAW[name]=(E,FR,T,feat,tg,fps,eps); ss=rng.choice(len(feat),min(20000,len(feat)),replace=False); sub.append(l2(feat[ss].astype(np.float32)))
pca=PCA(128,random_state=0).fit(np.concatenate(sub)); pm=pca.mean_.astype(np.float32); pcp=pca.components_.astype(np.float32)
DATA={}; MS={}
for name,(E,FR,T,feat,tg,fps,eps) in RAW.items():
    F128=l2((l2(feat.astype(np.float32))-pm)@pcp.T); NC=len(eps)
    fit=rng.choice(len(F128),min(80000,len(F128)),replace=False)
    bg=BayesianGaussianMixture(n_components=40,covariance_type="diag",weight_concentration_prior=1e-2,max_iter=120,random_state=0).fit(F128[fit]); labs=bg.predict(F128)
    C=[];P=[]
    for k in range(40):
        m=labs==k
        if m.sum()>=20 and len(set(E[m].tolist()))/NC>=0.5: C.append(F128[m].mean(0)); P.append(float(np.median(T[m])))
    C=l2(np.array(C,np.float32)); P=np.array(P); lam=16.*fps/3.; recs=[]
    for e in eps:
        idx=np.where(E==e)[0]; f=F128[idx]; t=T[idx]; recs.append([f,daw(f,C,P,lam),t,tg,int(e)])
    DATA[name]=recs; print(f'{name} M={len(C)} ({time.time()-t0:.0f}s)',flush=True)
TR={};EV={}
for n in DATA: r=DATA[n]; rng.shuffle(r); k=int(len(r)*0.85); TR[n]=r[:k]; EV[n]=r[k:]
class G(nn.Module):
    def __init__(s,h=256): super().__init__(); s.g=nn.GRU(128,h,2,batch_first=True); s.head=nn.Sequential(nn.Linear(h,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x,ln): p=nn.utils.rnn.pack_padded_sequence(x,ln.cpu(),batch_first=True,enforce_sorted=False); o,_=s.g(p); o,_=nn.utils.rnn.pad_packed_sequence(o,batch_first=True); return torch.sigmoid(s.head(o)).squeeze(-1)
net=G(256).to(DEV); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40)
gr={0:TR['kai']+TR['vis'],1:TR['xvla'],2:sum([TR[n] for n in DATA if n not in ('kai','vis','xvla')],[])}
for e in range(40):
    net.train(); pool=[]
    for g in gr: s=gr[g]; ii=rng.choice(len(s),min(250,len(s)),replace=len(s)<250); pool+=[s[i] for i in ii]
    pool=sorted(pool,key=lambda a:len(a[0]))
    for k in range(0,len(pool),24):
        gp=pool[k:k+24]; L=max(len(a[0]) for a in gp); B=len(gp); X=np.zeros((B,L,128),np.float32);Y=np.zeros((B,L),np.float32);M=np.zeros((B,L),np.float32);ln=np.zeros(B,int)
        for b,(f,v,t,tg,e2) in enumerate(gp): n=len(f);X[b,:n]=f;Y[b,:n]=v;M[b,:n]=1;ln[b]=n
        pr=net(torch.tensor(X,device=DEV),torch.tensor(ln)); loss=((( pr-torch.tensor(Y,device=DEV))**2)*torch.tensor(M,device=DEV)).sum()/torch.tensor(M,device=DEV).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
torch.save(net.state_dict(),REPO/'lmvla/crave/data/base_multitask_h256.pt')
net.eval()
@torch.no_grad()
def pred(f, warm=20):
    fw=np.concatenate([f[:1].repeat(warm,0),f])
    return net(torch.tensor(fw[None],device=DEV),torch.tensor([len(fw)]))[0].cpu().numpy()[warm:]
# per-task corr 全表
print("=== 12-task per-dataset held-out corr ===",flush=True)
allcorr={}
for n in DATA:
    c=[cc(pred(f),v) for f,v,t,tg,e in EV[n]]; allcorr[n]=np.nanmean(c); print(f"  {n}: {allcorr[n]:.3f} ",flush=True)
print(f"  >>> 12-task mean={np.mean(list(allcorr.values())):.3f}",flush=True)
# 渲 aloha 9 任务(coffee+8) 各1留出ep
alods=['coffee','cups_open','candy','screw_driver','vinh_cup','ziploc_slide','coffee_new','pro_pencil','vinh_cup_left']
fig,axes=plt.subplots(3,3,figsize=(14,9)); axes=axes.flatten()
for ax,dn in zip(axes,alods):
    f,v,t,tg,e=EV[dn][0]; p=pred(f)
    ax.plot(t,color='#e8830c',lw=1.1,alpha=.6,label='time'); ax.plot(v,color='#2ca02c',lw=1.5,label='teacher'); ax.plot(p,color='#1f77ff',lw=1.9,label='value model')
    ax.set_title(f'{dn} ep{e} corr={cc(p,v):.3f}',fontsize=9); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
fig.suptitle('DINOv3-base 12-task multitask value - 9 ALOHA tasks (held-out)',fontsize=12); fig.tight_layout()
fig.savefig(REPO/"temp/base_aloha9.png",dpi=110,bbox_inches='tight'); print('SAVED base_aloha9.png',flush=True)
for dsname,ncol in [("xvla",6),("coffee",8)]:
    ev=EV[dsname][:ncol]; nr=2; nc=(len(ev)+1)//2
    fig,axes=plt.subplots(nr,nc,figsize=(3.2*nc,6)); axes=np.atleast_1d(axes).flatten()
    for ax,(f,v,t,tg,e) in zip(axes,ev):
        p=pred(f); ax.plot(t,color='#e8830c',lw=1.2,alpha=.7,label='norm time'); ax.plot(v,color='#2ca02c',lw=1.6,label='CRAVE teacher'); ax.plot(p,color='#1f77ff',lw=2,label='base value model')
        ax.set_title(f'{dsname} ep{e} corr={cc(p,v):.3f}',fontsize=9); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    for ax in axes[len(ev):]: ax.axis('off')
    fig.suptitle(f'DINOv3-base multitask value - {dsname} held-out (causal, zero-future)',fontsize=12); fig.tight_layout()
    outp=REPO/f'temp/base_{dsname}.png'; fig.savefig(outp,dpi=115,bbox_inches='tight'); print('SAVED',outp.name,flush=True)
print('DONE',flush=True)
