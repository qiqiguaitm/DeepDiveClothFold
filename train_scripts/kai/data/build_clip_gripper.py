import glob, os, re, shutil, json
import numpy as np, pandas as pd
SB="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built"
S=f"{SB}/A_smooth800_dagger_all"; OUT=f"{SB}/A_smooth800_dagger_clip_all"
CAMS=["observation.images.top_head","observation.images.hand_left","observation.images.hand_right"]
GRIP=[6,13]
# source meta keyed by OLD episode_index
seps={json.loads(l)['episode_index']:json.loads(l) for l in open(f"{S}/meta/episodes.jsonl")}
sstats={json.loads(l)['episode_index']:json.loads(l) for l in open(f"{S}/meta/episodes_stats.jsonl")}
pqs=sorted(glob.glob(f"{S}/data/chunk-000/episode_*.parquet"), key=lambda p:int(re.search(r'episode_(\d+)',p).group(1)))
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(f"{OUT}/data/chunk-000"); os.makedirs(f"{OUT}/meta")
for c in CAMS: os.makedirs(f"{OUT}/videos/chunk-000/{c}")
eps_out=[]; stats_out=[]; gidx=0; n_clip=0; n_tot=0
for new,pq in enumerate(pqs):
    old=int(re.search(r'episode_(\d+)',pq).group(1))
    df=pd.read_parquet(pq); T=len(df)
    a=np.stack(df['action'].to_numpy()).astype(np.float32); g=a[:,GRIP]
    n_tot+=g.size; g[g>0.1]=0.1; m=g<=0.005; n_clip+=int(m.sum()); g[m]=0.0; a[:,GRIP]=g
    df=df.copy(); df['action']=list(a)
    df['episode_index']=np.int64(new); df['frame_index']=np.arange(T,dtype=np.int64); df['index']=np.arange(gidx,gidx+T,dtype=np.int64)
    df.to_parquet(f"{OUT}/data/chunk-000/episode_{new:06d}.parquet", index=False)
    for c in CAMS:
        src=f"{S}/videos/chunk-000/{c}/episode_{old:06d}.mp4"
        os.symlink(os.path.realpath(src), f"{OUT}/videos/chunk-000/{c}/episode_{new:06d}.mp4")
    er=dict(seps[old]); er['episode_index']=new; eps_out.append(er)
    sr=dict(sstats[old]); sr['episode_index']=new; stats_out.append(sr)
    gidx+=T
with open(f"{OUT}/meta/episodes.jsonl","w") as f:
    for e in eps_out: f.write(json.dumps(e)+"\n")
with open(f"{OUT}/meta/episodes_stats.jsonl","w") as f:
    for s in stats_out: f.write(json.dumps(s)+"\n")
shutil.copy(f"{S}/meta/tasks.jsonl", f"{OUT}/meta/tasks.jsonl")
info=json.load(open(f"{S}/meta/info.json")); N=len(pqs)
info.update(total_episodes=N, total_frames=gidx, total_videos=N*3, total_chunks=1, chunks_size=N, splits={"train":f"0:{N}"})
json.dump(info, open(f"{OUT}/meta/info.json","w"), indent=2)
print(f"REBUILT renumbered: {N} ep, {gidx} frames, chunks_size={N}; gripper ≤5mm→0: {n_clip}/{n_tot} ({100*n_clip/n_tot:.1f}%)")
