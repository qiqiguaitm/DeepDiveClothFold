import json, numpy as np, cv2, av, h5py, torch, torch.nn as nn
from pathlib import Path
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); DEV='cuda:0'; rng=np.random.RandomState(0); RES=128
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
class Dec(nn.Module):
    def __init__(s,d=768):
        super().__init__(); s.fc=nn.Linear(d,512*4*4)
        def blk(i,o): return nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.SiLU())
        s.up=nn.Sequential(blk(512,256),blk(256,128),blk(128,64),blk(64,32),blk(32,32)); s.out=nn.Conv2d(32,3,3,1,1)
    def forward(s,x): return torch.sigmoid(s.out(s.up(s.fc(x).view(-1,512,4,4))))
def milestones(feat,E,T,NC):
    bg=BayesianGaussianMixture(n_components=40,covariance_type='diag',weight_concentration_prior=1e-2,max_iter=120,random_state=0).fit(feat[rng.choice(len(feat),min(40000,len(feat)),replace=False)]); labs=bg.predict(feat); ms=[]
    for k in range(40):
        m=labs==k
        if m.sum()>=20 and len(set(E[m].tolist()))/NC>=0.5: ms.append((float(np.median(T[m])), l2(feat[m].mean(0)[None])[0]))
    ms.sort()
    return [ms[i] for i in np.linspace(0,len(ms)-1,min(8,len(ms))).astype(int)] if ms else []
# ===== COFFEE =====
z=np.load(REPO/'lmvla/crave/data/coffee_dinov3base/index.npz'); s=np.load(REPO/'lmvla/crave/data/coffee_dinov3base/shard_0.npz')
E=z['E'];FR=z['FR']; feat=l2(s['feat'].astype(np.float32)); N=len(feat); eps=sorted(np.unique(E).tolist()); NC=len(eps)
T=np.zeros(N,np.float32)
for e in eps: m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
imgs=np.zeros((N,RES,RES,3),np.uint8); cont=av.open(str(REPO/'temp/aloha_static_coffee/videos/observation.images.cam_high/chunk-000/file-000.mp4')); i=0
for fr in cont.decode(video=0):
    if i>=N: break
    imgs[i]=cv2.resize(fr.to_ndarray(format='rgb24'),(RES,RES)); i+=1
cont.close()
net=Dec().to(DEV); net.load_state_dict(torch.load(REPO/'lmvla/crave/data/base_decoder_coffee.pt')); net.eval()
ms=milestones(feat,E,T,NC); fig,ax=plt.subplots(2,len(ms),figsize=(2.3*len(ms),5))
for j,(pv,cen) in enumerate(ms):
    with torch.no_grad(): dec=net(torch.tensor(cen[None],device=DEV))[0].permute(1,2,0).cpu().numpy()
    i=int(np.linalg.norm(feat-cen,axis=1).argmin())
    ax[0,j].imshow(np.clip(dec,0,1)); ax[0,j].axis('off'); ax[0,j].set_title(('decoded centroid  ' if j==0 else '')+f'p={pv:.2f}',fontsize=9)
    ax[1,j].imshow(imgs[i]); ax[1,j].axis('off'); ax[1,j].set_title('retrieved nearest real frame' if j==0 else '',fontsize=9)
fig.suptitle('coffee milestone centroids: top=decoder synthesis / bottom=retrieval (nearest real frame), by progress p',fontsize=12)
fig.tight_layout(); fig.savefig(REPO/'temp/centroid_decode_coffee.png',dpi=105,bbox_inches='tight'); print('SAVED coffee',flush=True)
# self-recon coffee (EN)
ev=np.arange(N)[int(N*.9):][:10]; from crave.encoders import load_encoder; enc=load_encoder('dinov3-base',device='cuda')
fig,ax=plt.subplots(2,10,figsize=(22,4.6)); coss=[]
for j,i in enumerate(ev):
    with torch.no_grad(): rec=net(torch.tensor(feat[i][None],device=DEV))[0].permute(1,2,0).cpu().numpy()
    ax[0,j].imshow(imgs[i].astype(np.float32)/255.); ax[0,j].axis('off'); ax[0,j].set_title('original' if j==0 else '',fontsize=9)
    ax[1,j].imshow(np.clip(rec,0,1)); ax[1,j].axis('off'); ax[1,j].set_title('decoded' if j==0 else '',fontsize=9)
    re=np.asarray(enc.encode_pooled(cv2.resize((np.clip(rec,0,1)*255).astype(np.uint8),(224,224))[None]))[0]; coss.append(float(l2(re[None])[0]@feat[i]))
fig.suptitle(f'DINOv3-base pooled decoder self-reconstruction (coffee) · re-encode cos={np.mean(coss):.3f} (synthesis benchmark cap ~0.47)',fontsize=12)
fig.tight_layout(); fig.savefig(REPO/'temp/decoder_selfrecon_base.png',dpi=105,bbox_inches='tight'); print('SAVED selfrecon',flush=True)
del imgs
# ===== XVLA =====
XROOT=REPO/'xvla/data/xvla_soft_fold'; WORK=REPO/'temp/xvla_extract_base'; gidmap={}
for g in range(8):
    for bn,lep,gid in json.load(open(WORK/f'chunk_{g}.json')): gidmap[int(gid)]=(bn,int(lep))
z=np.load(REPO/'lmvla/crave/data/xvla_dinov3base_full/index.npz'); s=np.load(REPO/'lmvla/crave/data/xvla_dinov3base_full/shard_0.npz')
E=z['E'];FR=z['FR']; feat=l2(s['feat'].astype(np.float32)); N=len(feat); eps=sorted(np.unique(E).tolist()); NC=len(eps)
T=np.zeros(N,np.float32)
for e in eps: m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
net=Dec().to(DEV); net.load_state_dict(torch.load(REPO/'lmvla/crave/data/base_decoder_xvla.pt')); net.eval()
ms=milestones(feat,E,T,NC); fig,ax=plt.subplots(2,len(ms),figsize=(2.3*len(ms),5))
for j,(pv,cen) in enumerate(ms):
    with torch.no_grad(): dec=net(torch.tensor(cen[None],device=DEV))[0].permute(1,2,0).cpu().numpy()
    i=int(np.linalg.norm(feat-cen,axis=1).argmin()); gid=int(E[i]); fr=int(FR[i]); bn,lep=gidmap[gid]
    f=h5py.File(XROOT/bn/f'episode_{lep}.hdf5','r'); raw=f['observations/images/cam_high'][fr]; f.close()
    rimg=cv2.resize(np.ascontiguousarray(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]),(RES,RES))
    ax[0,j].imshow(np.clip(dec,0,1)); ax[0,j].axis('off'); ax[0,j].set_title(('decoded centroid  ' if j==0 else '')+f'p={pv:.2f}',fontsize=9)
    ax[1,j].imshow(rimg); ax[1,j].axis('off'); ax[1,j].set_title('retrieved nearest real frame' if j==0 else '',fontsize=9)
fig.suptitle('xvla milestone centroids: top=decoder synthesis / bottom=retrieval (nearest real frame), by progress p',fontsize=12)
fig.tight_layout(); fig.savefig(REPO/'temp/centroid_decode_xvla.png',dpi=105,bbox_inches='tight'); print('SAVED xvla',flush=True)
