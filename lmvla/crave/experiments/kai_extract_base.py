import sys, os, json, numpy as np, cv2, time
from pathlib import Path
sys.path.insert(0, "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/crave/src")
D = Path("/vePFS-North-E/vis_robot/workspace/deepdive_kai0")
VID = D / "kai0/data/Task_A/kai0_base/videos"; WORK = D / "temp/kai_extract_base"; WORK.mkdir(parents=True, exist_ok=True)
def vpath(e): return VID / f"chunk-{e//1000:03d}/observation.images.top_head/episode_{e:06d}.mp4"

def enum(nsplit=8):
    mp4 = sorted(VID.glob("chunk-*/observation.images.top_head/episode_*.mp4"))
    eps = sorted(int(p.stem.split("_")[1]) for p in mp4)
    for g in range(nsplit): json.dump(eps[g::nsplit], open(WORK/f"chunk_{g}.json","w"))
    print(f"enum {len(eps)} kai eps -> {nsplit} chunks")

def worker(g):
    from crave.encoders import load_encoder
    eps = json.load(open(WORK/f"chunk_{g}.json")); enc = load_encoder("dinov3-base", device="cuda"); t0=time.time()
    E=[]; FR=[]; FE=[]
    for ci, e in enumerate(eps):
        fp = vpath(e)
        if not fp.exists(): print(f"[g{g}] miss ep{e}",flush=True); continue
        cap = cv2.VideoCapture(str(fp)); frames=[]
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (224,224)))
        cap.release()
        if len(frames) < 5: continue
        feats=[]
        for k in range(0,len(frames),256):
            feats.append(np.asarray(enc.encode_pooled(np.stack(frames[k:k+256]))).astype(np.float16))
        fe=np.concatenate(feats); n=len(fe); E+=[e]*n; FR+=list(range(n)); FE.append(fe)
        if ci%50==0: print(f"[g{g}] {ci+1}/{len(eps)} ep{e} n{n} · {(time.time()-t0)/60:.1f}min",flush=True)
    np.savez(WORK/f"part_{g}.npz", E=np.array(E,np.int64), FR=np.array(FR,np.int64), feat=np.concatenate(FE))
    print(f"[g{g}] DONE {len(E)} frames ({(time.time()-t0)/60:.1f}min)",flush=True)

def merge():
    parts=sorted(WORK.glob("part_*.npz"))
    E=np.concatenate([np.load(p)["E"] for p in parts]); FR=np.concatenate([np.load(p)["FR"] for p in parts]); feat=np.concatenate([np.load(p)["feat"] for p in parts])
    o=np.lexsort((FR,E)); E,FR,feat=E[o],FR[o],feat[o]; n=len(E)
    out=D/"temp/kai_dinov3base"; out.mkdir(exist_ok=True)
    np.savez(out/"index.npz",E=E.astype(np.int64),FR=FR.astype(np.int64),T=(FR/30.0).astype(np.float32),n=np.int64(n))
    np.savez(out/"shard_0.npz",gidx=np.arange(n,dtype=np.int64),feat=feat,valid=np.ones(n,bool))
    print(f"merged {len(np.unique(E))} eps / {n} frames -> temp/kai_dinov3base/")

if __name__=="__main__":
    {"enum":lambda:enum(8),"worker":lambda:worker(int(sys.argv[2])),"merge":merge}[sys.argv[1]]()
