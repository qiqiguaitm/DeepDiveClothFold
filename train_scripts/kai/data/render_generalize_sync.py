"""三数据集连续value × 画面 同步mp4(3Hz分辨率, 8fps)。读 temp/_gen_{tag}.npz 的 cont + 按格式取该ep的3Hz帧。
用法: python render_generalize_sync.py <tag: xvla|visbase|coffee>
"""
import sys, json, glob
from pathlib import Path
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt, os
_sh=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"]=["SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
TAG=sys.argv[1]; R=Path("/vePFS/tim/workspace/deepdive_kai0")
z=np.load(R/f"temp/_gen_{TAG}.npz"); cont=z["cont"]; disc=z["disc"]; TEST=int(z["test"]); print(f"[{TAG}] ep{TEST} cont {len(cont)}",flush=True)

def crop224(im):  # im: HxWx3 RGB
    s=224/min(im.shape[:2]); g=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s))); h,w=g.shape[:2]; return g[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224]

def frames_xvla():
    import h5py
    fp=R/"xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow"/f"episode_{TEST}.hdf5"
    with h5py.File(fp,"r") as h:
        T=h["observations/qpos"].shape[0]; sel=np.arange(0,T,10)
        return [crop224(cv2.cvtColor(cv2.imdecode(np.frombuffer(h["observations/images/cam_high"][i],np.uint8),1),cv2.COLOR_BGR2RGB)) for i in sel]

def frames_visbase():
    import av
    mp4=R/"kai0/data/Task_A/vis_base/v3/2026-04-24-v3/videos/chunk-000/observation.images.top_head"/f"episode_{TEST:06d}.mp4"
    out=[]; c=av.open(str(mp4))
    for i,f in enumerate(c.decode(video=0)):
        if i%10==0: out.append(crop224(f.to_ndarray(format="rgb24")))
    c.close(); return out

def frames_coffee():
    import av, pandas as pd
    repo=Path("/vePFS/tim/workspce/hf_cache/hub_default/datasets--lerobot--aloha_static_coffee")
    snap=sorted(repo.glob("snapshots/*"))[0]
    epm=pd.read_parquet(glob.glob(str(snap/"meta/episodes/**/*.parquet"),recursive=True)[0])
    epm=epm.sort_values("episode_index").reset_index(drop=True)
    r=epm[epm.episode_index==TEST].iloc[0]; f0,t0=int(r.dataset_from_index),int(r.dataset_to_index)
    cam="observation.images.cam_high"
    mp4=glob.glob(str(snap/"videos"/cam/"**/*.mp4"),recursive=True)[0]
    sel=set(range(f0,t0,16)); out=[]; c=av.open(mp4)
    for gi,f in enumerate(c.decode(video=0)):
        if gi in sel: out.append(crop224(f.to_ndarray(format="rgb24")))
        if gi>=t0: break
    c.close(); return out

frs={"xvla":frames_xvla,"visbase":frames_visbase,"coffee":frames_coffee}[TAG]()
n=min(len(frs),len(cont),len(disc)); frs=frs[:n]; cont=cont[:n]; disc=disc[:n]; print(f"frames {len(frs)} vs cont {n}",flush=True)
x=np.arange(n)
fig=plt.figure(figsize=(10,8)); gs=fig.add_gridspec(2,1,height_ratios=[1.5,1.0],hspace=0.2)
axc=fig.add_subplot(gs[0]); axc.axis("off"); im=axc.imshow(frs[0]); ttl=axc.set_title("",fontsize=11)
axv=fig.add_subplot(gs[1])
axv.step(x,disc,where="post",color="#1f77b4",lw=1.5,alpha=.7,label="离散 CRAVE(阶梯)")
axv.plot(x,cont,color="#2ca02c",lw=2.0,label="连续 TCC+DP")
axv.axhline(1,color="#ddd",ls=":",lw=1)
cur=axv.axvline(0,color="k",lw=1.3)
dotd,=axv.plot([0],[disc[0]],"s",color="#1f77b4",ms=7,mec="k")
dot,=axv.plot([0],[cont[0]],"o",color="#2ca02c",ms=8,mec="k")
axv.set_xlim(0,n); axv.set_ylim(-0.02,1.05); axv.set_xlabel("3Hz step"); axv.set_ylabel("value"); axv.grid(alpha=.25); axv.legend(fontsize=9,loc="upper left")
fig.suptitle(f"跨数据泛化 [{TAG}] ep{TEST}: 离散CRAVE vs 连续TCC+DP × 画面同步",fontsize=12,y=0.97)
import av
OUT=R/f"temp/generalize_sync_{TAG}.mp4"; oc=av.open(str(OUT),mode="w"); st=oc.add_stream("libx264",rate=8); done=None
for t in range(n):
    im.set_data(frs[t]); ttl.set_text(f"step {t}/{n}  离散={disc[t]:.2f}  连续={cont[t]:.3f}")
    cur.set_xdata([t,t]); dot.set_data([t],[cont[t]]); dotd.set_data([t],[disc[t]]); fig.canvas.draw()
    arr=np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[...,:3])
    if done is None: H,W=arr.shape[:2]; H-=H%2; W-=W%2; st.width,st.height,st.pix_fmt=W,H,"yuv420p"; st.options={"crf":"20"}; done=True
    for pkt in st.encode(av.VideoFrame.from_ndarray(arr[:H,:W],format="rgb24")): oc.mux(pkt)
for pkt in st.encode(): oc.mux(pkt)
oc.close(); print(f"SAVED {OUT} {W}x{H} {n}f @8fps",flush=True); print("DONE")
