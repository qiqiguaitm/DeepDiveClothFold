"""端到端TCC readout: 置信门控 — 区分 fold匹配失败凹口(低置信,保持) vs 松手真回退(高置信,允许掉)。
验证: fold凹口处匹配置信是否确实低(证明该门控而非填平)。复用已存模型 temp/_e2e_kai0base_model.pt。
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, av, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from transformers import AutoModel
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
REPO = Path("/vePFS/tim/workspace/deepdive_kai0"); BASE = REPO / "kai0/data/Task_A/kai0_base"
FR = REPO / "temp/tcc_e2e_frames/kai0base"; csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
TEST = 2047; dev = "cuda"
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
eps = sorted(int(p.stem[2:]) for p in FR.glob("ep*.npz")); TRAIN = [e for e in eps if e != TEST]
def prop3(e, n, stride=10):
    st = np.stack(pd.read_parquet(BASE/"data"/f"chunk-{e//csB:03d}"/f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n)*stride, len(st)-1)] if stride>1 else st[:n]
    return np.concatenate([st, np.vstack([np.zeros((1,14)), np.diff(st,axis=0)])],1).astype(np.float32)
IMG, PRr = {}, {}
for e in TRAIN: IMG[e] = np.load(FR/f"ep{e}.npz")["frames"]; PRr[e] = prop3(e, len(IMG[e]))
allp = np.concatenate([PRr[e] for e in TRAIN]); MU, SD = allp.mean(0), allp.std(0)+1e-8
def pn(p): q=(p-MU)/SD; return (q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-9)).astype(np.float32)
for e in PRr: PRr[e] = pn(PRr[e])
bb = AutoModel.from_pretrained("facebook/dinov2-small").to(dev)
head = nn.Sequential(nn.Linear(412,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,128)).to(dev)
ck = torch.load(REPO/"temp/_e2e_kai0base_model.pt"); bb.load_state_dict(ck["bb"]); head.load_state_dict(ck["head"]); bb.eval(); head.eval()
@torch.no_grad()
def embed(fr_u8, pr):
    out=[]
    for b in range(0,len(fr_u8),128):
        x=torch.from_numpy(fr_u8[b:b+128]).to(dev).permute(0,3,1,2).float()/255.; x=(x-IMEAN)/ISTD
        with torch.autocast("cuda",dtype=torch.bfloat16): vis=bb(x).last_hidden_state[:,1:].mean(1).float()
        vis=vis/(vis.norm(dim=-1,keepdim=True)+1e-9)
        out.append(head(torch.cat([vis, torch.from_numpy(pr[b:b+128]).to(dev)],-1)).cpu().numpy())
    z=np.concatenate(out); return z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-9)
REFS=TRAIN[:30]; REs=[embed(IMG[e],PRr[e]) for e in REFS]; RTs=[np.arange(len(z))/max(1,len(z)-1) for z in REs]
# ep2047 全 2629 帧
import cv2
def crop(im): s=224/min(im.shape[:2]); g=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s))); h,w=g.shape[:2]; return g[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224]
c=av.open(str(BASE/"videos"/f"chunk-{TEST//csB:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4")); frs=np.stack([crop(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]).astype(np.uint8); c.close()
zq=embed(frs, pn(prop3(TEST,len(frs),stride=1)[:len(frs)])); n=len(zq)
# 每帧: 各参考最佳余弦 → 时间估计 + 置信(最佳相似度)
times=[]; sims=[]
for k in range(len(REFS)):
    s=zq@REs[k].T; j=s.argmax(1); times.append(RTs[k][j]); sims.append(s.max(1))
raw=np.median(np.stack(times),0); conf=np.median(np.stack(sims),0)   # 置信=各参考最佳相似度中位
def med(a,w=27): h=w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
raw_s=med(raw); conf_s=med(conf,15)
# 置信门控: 向上自由; 向下仅当高置信(>阈值=置信中位)才接受, 否则保持
thr=np.median(conf_s)
V=np.zeros(n); V[0]=raw_s[0]
for i in range(1,n):
    if raw_s[i]>=V[i-1]: V[i]=raw_s[i]                      # 进步: 跟随
    elif conf_s[i]>=thr: V[i]=raw_s[i]                      # 高置信下降=真回退: 允许掉
    else: V[i]=V[i-1]                                       # 低置信下降=匹配失败: 保持
def mono(v): return np.mean(np.diff(v)>=-1e-6)
cur=med(raw)
print(f"末段(后25%)最低: 原中值{cur[int(.75*n):].min():.2f} 门控{V[int(.75*n):].min():.2f}; 单调 原{mono(cur):.0%} 门控{mono(V):.0%}")
print(f"置信: 全程中位{np.median(conf_s):.3f}, 末段fold区(后15%)中位{np.median(conf_s[int(.85*n):]):.3f}", flush=True)
x=np.arange(n)
fig,ax=plt.subplots(2,1,figsize=(13,7),height_ratios=[1.5,1],sharex=True)
ax[0].plot(x,cur,color="#bbb",lw=1.5,label=f"原中值readout (单调{mono(cur):.0%}, 末段凹口→{cur[int(.75*n):].min():.2f})")
ax[0].plot(x,V,color="#2ca02c",lw=2.0,label=f"置信门控 (单调{mono(V):.0%}; 低置信下降→保持, 高置信下降→允许掉)")
ax[0].axhline(1,color="#ddd",ls=":",lw=1); ax[0].set_ylim(-0.05,1.12); ax[0].set_ylabel("value"); ax[0].grid(alpha=.25); ax[0].legend(fontsize=9,loc="upper left")
ax[0].set_title("端到端TCC: 置信门控(保留真回退/只压匹配失败凹口) — 非填平",fontsize=12)
ax[1].plot(x,conf_s,color="#9467bd",lw=1.5,label="匹配置信(各参考最佳相似度中位)")
ax[1].axhline(thr,color="r",ls="--",lw=1,label=f"门控阈值(中位{thr:.3f})")
ax[1].set_ylabel("置信"); ax[1].set_xlabel("frame(30Hz)"); ax[1].grid(alpha=.25); ax[1].legend(fontsize=8,loc="lower left")
ax[1].set_title("验证: fold凹口处匹配置信应偏低(=匹配失败, 该保持而非真回退)",fontsize=10.5)
out=REPO/"docs/visualization/cross_episode_recurrence_value/e2e_confgate_ep2047.png"
fig.tight_layout(); fig.savefig(out,dpi=125); print("SAVED",out)
np.savez(REPO/"temp/_e2e_confgate_ep2047.npz", raw=raw, conf=conf_s, gated=V); print("DONE")
