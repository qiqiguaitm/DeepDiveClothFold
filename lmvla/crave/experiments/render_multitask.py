import numpy as np, torch, torch.nn as nn
from pathlib import Path
from crave.render import setup_mpl
plt=setup_mpl()
OUT=Path('/home/tim/workspace/deepdive_kai0/temp/multitask_cache'); DEV='cuda:0'; rng=np.random.RandomState(0)
DS=['kai','vis','coffee','xvla']; NT=4
def cc(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else np.nan
DATA={d:[] for d in DS}
for di,d in enumerate(DS):
    z=np.load(OUT/f'{d}.npz'); 
    for e in z['ep']: DATA[d].append((z[f'f{e}'],z[f'v{e}'],z[f't{e}'],di,int(e)))
    rng.shuffle(DATA[d])
EV={d:DATA[d][int(len(DATA[d])*0.85):] for d in DS}
class GRUv(nn.Module):
    def __init__(s,h=256,L=2,emb=32):
        super().__init__(); s.emb=nn.Embedding(NT,emb); s.g=nn.GRU(128+emb,h,L,batch_first=True); s.head=nn.Sequential(nn.Linear(h,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x,task):
        e=s.emb(task)[:,None,:].expand(-1,x.shape[1],-1); o,_=s.g(torch.cat([x,e],-1)); return torch.sigmoid(s.head(o)).squeeze(-1)
net=GRUv().to(DEV); net.load_state_dict(torch.load('/home/tim/workspace/deepdive_kai0/temp/crave_multitask_gru.pt')); net.eval()
fig=plt.figure(figsize=(17,8)); gs=fig.add_gridspec(2,4,height_ratios=[1,1.1])
# 上排: bar 对比
cfgs=['单任务×4','共享\n无task','共享\n+task','共享\n大']
data=np.array([[0.972,0.981,0.970,0.979],[0.956,0.964,0.980,0.973],[0.958,0.978,0.969,0.923],[0.955,0.973,0.975,0.947]])
axb=fig.add_subplot(gs[0,:2]); x=np.arange(NT); w=0.2; cols=['#888','#1f77ff','#2ca02c','#d62728']
for i in range(4): axb.bar(x+(i-1.5)*w,data[i],w,label=cfgs[i].replace('\n',''),color=cols[i])
axb.set_xticks(x); axb.set_xticklabels(DS); axb.set_ylim(0.9,1.0); axb.set_ylabel('progress corr vs teacher'); axb.legend(fontsize=8,ncol=2); axb.set_title('per-task corr · 4配置'); axb.grid(axis='y',alpha=.3)
axm=fig.add_subplot(gs[0,2:]); means=[0.976,0.968,0.957,0.962]; params=[0.68,0.72,0.75,1.57]
axm.plot(params,means,'o-',color='#1f77ff',ms=9)
for p,m,c in zip(params,means,cfgs): axm.annotate(c.replace('\n',''),(p,m),fontsize=8,xytext=(3,3),textcoords='offset points')
axm.set_xlabel('参数量(M)'); axm.set_ylabel('mean corr(4任务)'); axm.set_title('参数量 vs 性能: 大模型反而更差(小任务过拟合)'); axm.grid(alpha=.3)
# 下排: 每任务一条留出曲线
for j,d in enumerate(DS):
    f,v,t,ti,e=EV[d][0]
    with torch.no_grad(): pr=net(torch.tensor(f[None],device=DEV),torch.tensor([ti],device=DEV))[0].cpu().numpy()
    ax=fig.add_subplot(gs[1,j]); xx=np.arange(len(v))
    ax.plot(xx,t,color='#e8830c',lw=1.3,alpha=.7,label='归一时间'); ax.plot(xx,v,color='#2ca02c',lw=1.6,label='CRAVE teacher')
    ax.plot(xx,pr,color='#1f77ff',lw=2,label='共享模型(在线)')
    ax.set_title(f'{d} ep{e}(留出) corr={cc(pr,v):.3f}',fontsize=10); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25)
    if j==0: ax.legend(fontsize=8)
fig.suptitle('单个共享 0.72M value 模型跨 4 任务(kai/vis/coffee/xvla, xvla全168ep) · 一个模型全搞定',fontsize=13)
fig.tight_layout(); fig.savefig('lmvla/crave/docs/visualization/online_value/multitask_4task.png',dpi=115,bbox_inches='tight'); print('SAVED multitask_4task.png')
