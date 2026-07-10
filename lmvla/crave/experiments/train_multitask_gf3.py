#!/usr/bin/env python
"""Multitask CRAVE value — gf3 8卡版(域重组 + 均衡采样 + 参数量扫描).

在 gf3(/vePFS-North-E/vis_robot/workspace/deepdive_kai0, fastwam env, crave/src)上跑。
本地运行需改: REPO 路径、G=recurrence_graphs 路径(本地在 lmvla/lmwm/data/...)、
             DSETS 里 xvla bank='xvla_dinov3h_full'(1532ep, 两套数据已就绪)。

设计:
  - 4 数据集(kai/vis/coffee/xvla)DINOv3-H 1280D → 共享 PCA128;teacher=各自 recurrence_graph 双锚Viterbi+polyline。
  - 域重组: kai+vis=task0(同SOP同臂)/ xvla=task1 / coffee=task2;均衡采样(kai/xvla cap 1000)。
  - 参数量扫描 h128/256/384/512 + L3 → 证实容量非瓶颈(0.21M–2.63M mean 0.954–0.960, h512最差)。
实测见 docs/multitask_value.md §3.5。
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=crave/src python temp/mt_gf3_v2.py
"""
import numpy as np, time, torch, torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); G=REPO/'lmwm/data/recurrence_graphs'; DEV='cuda:0'
rng=np.random.RandomState(0); torch.manual_seed(0)
# (名, bank, graph, fps, task_group, cap)  均衡: kai/xvla cap 1000, vis/coffee 全
DSETS=[('kai','crave_full_dinov3h','kai0base_dinov3h',3.0,0,1000),('vis','vis_dinov3h','vis_dinov3h',30.0,0,10000),
       ('xvla','xvla_dinov3h_full','xvla_dinov3h',30.0,1,1000),('coffee','coffee_dinov3h','coffee_dinov3h',50.0,2,10000)]
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
def load_bank(bank):
    d=REPO/'temp'/bank; idx=np.load(d/'index.npz'); E=idx['E'];FR=idx['FR'];T=idx['T'];N=len(E);feat=np.zeros((N,1280),np.float16)
    for sh in sorted(d.glob('shard_*.npz')):
        s=np.load(sh); g=s['gidx']; v=s['valid'] if 'valid' in s else np.ones(len(g),bool); feat[g[v]]=s['feat'][v]
    return E,FR,T,feat
def daw(F,Ct,pord,lam):
    Fn=l2(F);Ctn=l2(Ct);vals=np.asarray(pord,float);sC=l2(Fn[:3].mean(0)[None])[0];eC=l2(Fn[-3:].mean(0)[None])[0]
    C2=np.vstack([Ctn,sC,eC]);P=np.concatenate([vals,[0.],[1.]]);bins=np.unique(np.concatenate([[0.],P,[1.]]));nb=len(bins)
    cb=[int(np.searchsorted(bins,v)) for v in P];pen=lam*np.abs(bins[:,None]-bins[None])
    de=np.linalg.norm(Fn[:,None]-C2[None],axis=2);em=np.full((len(Fn),nb),1e3)
    for ti in range(len(P)): em[:,cb[ti]]=np.minimum(em[:,cb[ti]],de[:,ti])
    cost=np.full(nb,1e9);cost[0]=em[0,0];BP=np.zeros((len(Fn),nb),int)
    for j in range(1,len(Fn)):
        tr=cost[None,:]+pen;kk=tr.argmin(1);cost=em[j]+tr[np.arange(nb),kk];BP[j]=kk
    si=nb-1;path=np.zeros(len(Fn),int);path[-1]=si
    for j in range(len(Fn)-2,-1,-1): si=BP[j+1][si];path[j]=si
    step=bins[path];segs=[];a=0
    for t in range(1,len(step)):
        if step[t]!=step[t-1]: segs.append((a,t-1,step[t-1]));a=t
    segs.append((a,len(step)-1,step[-1]));reps=[]
    for i0,i1,val in segs:
        cand=[ti for ti in range(len(P)) if abs(P[ti]-val)<1e-9];fr=np.arange(i0,i1+1);bd=1e18;bf=i0
        for ti in cand:
            dd=np.linalg.norm(Fn[fr]-C2[ti],axis=1);k=int(dd.argmin())
            if dd[k]<bd: bd=dd[k];bf=fr[k]
        reps.append((bf,float(val)))
    if reps[0][0]!=0: reps=[(0,float(step[0]))]+reps
    if reps[-1][0]!=len(step)-1: reps=reps+[(len(step)-1,float(step[-1]))]
    rf=np.array([r[0] for r in reps]);rv=np.array([r[1] for r in reps]);keep=np.concatenate([[True],np.diff(rf)>0])
    return np.interp(np.arange(len(step)),rf[keep],rv[keep]).astype(np.float32)
def cc(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else np.nan
print('加载+teacher(均衡cap)...',flush=True);t0=time.time();DATA={};sub=[]
for name,bank,gg,fps,tg,cap in DSETS:
    E,FR,T,feat=load_bank(bank);z=np.load(G/gg/'recurrence_graph.npz',allow_pickle=True);Ct=z['prototype_table'].astype(np.float32);pord=z['pord']
    lam=16.0*fps/3.0;eps=sorted(np.unique(E).tolist())
    if len(eps)>cap: eps=[eps[i] for i in sorted(rng.choice(len(eps),cap,replace=False))]  # cap before teacher
    recs=[];ss=rng.choice(len(feat),min(20000,len(feat)),replace=False);sub.append(l2(feat[ss].astype(np.float32)))
    for e in eps:
        loc=np.where(E==e)[0];o=np.argsort(FR[loc]);gi=loc[o];F=feat[gi].astype(np.float32);t=T[gi].astype(np.float32)
        if len(F)<8: continue
        recs.append([F,daw(F,Ct,pord,lam),t,tg])
    DATA[name]=recs;print(f'  {name}: {len(recs)} eps ({time.time()-t0:.0f}s)',flush=True)
pca=PCA(128,random_state=0).fit(np.concatenate(sub));pm=pca.mean_.astype(np.float32);pc=pca.components_.astype(np.float32)
for name in DATA:
    for r in DATA[name]: r[0]=l2((l2(r[0])-pm)@pc.T).astype(np.float32)
print(f'PCA fitted ({time.time()-t0:.0f}s)',flush=True)
TR={};EV={}
for name in DATA:
    r=DATA[name];rng.shuffle(r);k=int(len(r)*0.85);TR[name]=r[:k];EV[name]=r[k:]
print('train/eval:',{n:(len(TR[n]),len(EV[n])) for n in DATA},flush=True)
NT=3
class GRUv(nn.Module):
    def __init__(s,h=256,L=2,emb=0):
        super().__init__();s.emb=nn.Embedding(NT,emb) if emb>0 else None;din=128+(emb if emb>0 else 0)
        s.g=nn.GRU(din,h,L,batch_first=True);s.head=nn.Sequential(nn.Linear(h,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x,task,ln):
        if s.emb is not None: x=torch.cat([x,s.emb(task)[:,None,:].expand(-1,x.shape[1],-1)],-1)
        p=nn.utils.rnn.pack_padded_sequence(x,ln.cpu(),batch_first=True,enforce_sorted=False);o,_=s.g(p);o,_=nn.utils.rnn.pad_packed_sequence(o,batch_first=True)
        return torch.sigmoid(s.head(o)).squeeze(-1)
def batches(pool,bs=24):
    idx=sorted(range(len(pool)),key=lambda i:len(pool[i][0]))
    for k in range(0,len(idx),bs):
        gp=[pool[i] for i in idx[k:k+bs]];L=max(len(a[0]) for a in gp);B=len(gp)
        X=np.zeros((B,L,128),np.float32);Y=np.zeros((B,L),np.float32);M=np.zeros((B,L),np.float32);tk=np.zeros(B,int);ln=np.zeros(B,int)
        for b,(f,v,t,tg) in enumerate(gp):
            n=len(f);X[b,:n]=f;Y[b,:n]=v;M[b,:n]=1;tk[b]=tg;ln[b]=n
        yield torch.tensor(X,device=DEV),torch.tensor(Y,device=DEV),torch.tensor(M,device=DEV),torch.tensor(tk,device=DEV),torch.tensor(ln)
@torch.no_grad()
def ev(net):
    net.eval();out={}
    for n in DATA:
        c=[]
        for f,v,t,tg in EV[n]:
            pr=net(torch.tensor(f[None],device=DEV),torch.tensor([tg],device=DEV),torch.tensor([len(f)]))[0].cpu().numpy();c.append(cc(pr,v))
        out[n]=np.nanmean(c)
    return out
def train(h,L=2,emb=0,ep=45,cap=250):
    net=GRUv(h,L,emb).to(DEV);opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep)
    groups={0:TR['kai']+TR['vis'],1:TR['xvla'],2:TR['coffee']}
    for e in range(ep):
        net.train();epool=[]
        for g in groups: src=groups[g];ii=rng.choice(len(src),min(cap,len(src)),replace=len(src)<cap);epool+=[src[i] for i in ii]
        for X,Y,M,tk,ln in batches(epool):
            pr=net(X,tk,ln);loss=(((pr-Y)**2)*M).sum()/M.sum();opt.zero_grad();loss.backward();opt.step()
        sch.step()
    return net,sum(p.numel() for p in net.parameters())
print('\n=== 均衡数据(kai1000/vis289/xvla1000/coffee50) · 参数量扫描 ===',flush=True)
for tag,(h,L) in [('h128/L2',(128,2)),('h256/L2',(256,2)),('h384/L2',(384,2)),('h512/L2',(512,2)),('h256/L3',(256,3))]:
    net,p=train(h,L); r=ev(net)
    print(f'{tag:>9} ({p/1e6:.2f}M): '+' '.join(f'{n}={r[n]:.3f}' for n in DATA)+f' mean={np.mean(list(r.values())):.3f}',flush=True)
print('DONE',flush=True)
