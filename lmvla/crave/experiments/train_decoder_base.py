import sys, numpy as np, av, cv2, time, torch, torch.nn as nn
from pathlib import Path
from crave.render import setup_mpl
plt=setup_mpl()
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); DEV='cuda:0'; rng=np.random.RandomState(0); torch.manual_seed(0)
RES=128; DS='coffee'; BANK='coffee_dinov3base'; VID=REPO/'temp/aloha_static_coffee/videos/observation.images.cam_high/chunk-000/file-000.mp4'
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
# 特征(pooled 768) + 顺序解码视频帧配对
z=np.load(REPO/f'lmvla/crave/data/{BANK}/index.npz'); s=np.load(REPO/f'lmvla/crave/data/{BANK}/shard_0.npz')
feat=s['feat'].astype(np.float32); N=len(feat); print(f'{DS} feat {feat.shape}',flush=True)
t0=time.time(); imgs=np.zeros((N,RES,RES,3),np.uint8); cont=av.open(str(VID)); i=0
for fr in cont.decode(video=0):
    if i>=N: break
    imgs[i]=cv2.resize(fr.to_ndarray(format='rgb24'),(RES,RES)); i+=1
cont.close(); print(f'decoded {i} frames ({time.time()-t0:.0f}s)',flush=True)
feat=l2(feat); idx=np.arange(N); rng.shuffle(idx); tr=idx[:int(N*.9)]; ev=idx[int(N*.9):]
class Dec(nn.Module):
    def __init__(s,d=768):
        super().__init__(); s.fc=nn.Linear(d,512*4*4)
        def blk(i,o): return nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.SiLU())
        s.up=nn.Sequential(blk(512,256),blk(256,128),blk(128,64),blk(64,32),blk(32,32)); s.out=nn.Conv2d(32,3,3,1,1)
    def forward(s,x): h=s.fc(x).view(-1,512,4,4); return torch.sigmoid(s.out(s.up(h)))
net=Dec().to(DEV); opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=1e-5)
print(f'decoder 参数 {sum(p.numel() for p in net.parameters())/1e6:.1f}M',flush=True)
Ft=torch.tensor(feat[tr],device=DEV); It=torch.tensor(imgs[tr].astype(np.float32)/255.,device=DEV).permute(0,3,1,2)
for ep in range(40):
    perm=torch.randperm(len(tr),device=DEV); tl=0
    for k in range(0,len(tr),128):
        b=perm[k:k+128]; pr=net(Ft[b]); loss=(pr-It[b]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    if (ep+1)%10==0: print(f'ep{ep+1} L1={tl/(len(tr)//128):.4f}',flush=True)
# 自重建 + 再编码cos(需encoder)
net.eval()
from crave.encoders import load_encoder; enc=load_encoder('dinov3-base',device='cuda')
sel=ev[:10]; fig,ax=plt.subplots(2,10,figsize=(22,4.6))
coss=[]; l1s=[]
for j,i in enumerate(sel):
    with torch.no_grad(): rec=net(torch.tensor(feat[i][None],device=DEV))[0].permute(1,2,0).cpu().numpy()
    orig=imgs[i].astype(np.float32)/255.
    ax[0,j].imshow(orig); ax[0,j].axis('off'); ax[0,j].set_title('原图' if j==0 else '',fontsize=9)
    ax[1,j].imshow(np.clip(rec,0,1)); ax[1,j].axis('off'); ax[1,j].set_title('解码' if j==0 else '',fontsize=9)
    l1s.append(np.abs(orig-rec).mean())
    re=np.asarray(enc.encode_pooled(cv2.resize((np.clip(rec,0,1)*255).astype(np.uint8),(224,224))[None]))[0]
    coss.append(float(l2(re[None])[0]@feat[i]))
fig.suptitle(f'DINOv3-base pooled decoder 自重建 · coffee · 再编码cos={np.mean(coss):.3f} L1={np.mean(l1s):.3f}(基准: 合成封顶cos~0.47)',fontsize=12)
fig.tight_layout(); fig.savefig(REPO/'temp/decoder_selfrecon_base.png',dpi=105,bbox_inches='tight')
torch.save(net.state_dict(),REPO/'lmvla/crave/data/base_decoder_coffee.pt')
print(f'SAVED selfrecon · cos={np.mean(coss):.3f} L1={np.mean(l1s):.3f}',flush=True)
