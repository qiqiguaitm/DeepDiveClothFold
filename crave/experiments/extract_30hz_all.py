#!/usr/bin/env python
"""全量 3055 ep 的 30Hz native DINOv3-H 特征抽取(crop224 = shard 同空间, cos=1.0)。
每 ep 存 temp/crave_30hz_feat_v2/ep{e}.npy (n30,1280) fp16。可断点续跑(跳过已存在)。
线程池并行解码(crop224)+ 主线程 GPU encode_pooled。
Run(双卡): CUDA_VISIBLE_DEVICES=0 ... --rank 0 --world 2  &  CUDA_VISIBLE_DEVICES=1 ... --rank 1 --world 2
"""
import argparse, time, av, cv2, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from crave.encoders import load_encoder
from crave.config import resolve_dataset
from crave.data import kai0

REPO=Path("/home/tim/workspace/deepdive_kai0"); OUT=REPO/"temp/crave_30hz_feat_v2"; OUT.mkdir(exist_ok=True,parents=True)
def crop224(rgb):
    h,w=rgb.shape[:2]; s=224/min(h,w); r=cv2.resize(rgb,(int(round(w*s)),int(round(h*s))))
    hh,ww=r.shape[:2]; return np.ascontiguousarray(r[(hh-224)//2:(hh+224)//2,(ww-224)//2:(ww+224)//2])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--rank",type=int,default=0); ap.add_argument("--world",type=int,default=1); a=ap.parse_args()
    cfg=resolve_dataset("kai0_base"); cs=kai0.chunks_size(cfg.root); DS=Path(cfg.root)
    all_eps=sorted(int(p.stem.split("_")[1]) for p in (DS/"data").glob("chunk-*/episode_*.parquet"))
    mine=[e for i,e in enumerate(all_eps) if i%a.world==a.rank and not (OUT/f"ep{e}.npy").exists()]
    print(f"[rank{a.rank}/{a.world}] {len(mine)}/{len(all_eps)} eps to do",flush=True)
    enc=load_encoder("dinov3-h")
    def decode(e):
        vid=DS/f"videos/chunk-{e//cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
        cap=av.open(str(vid)); frs=[crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]; cap.close(); return e,frs
    t0=time.time(); done=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for e,frs in ex.map(decode,mine):
            f=enc.encode_pooled(frs).astype(np.float16); np.save(OUT/f"ep{e}.npy",f); done+=1
            if done%50==0:
                el=time.time()-t0; print(f"[rank{a.rank}] {done}/{len(mine)} ({el:.0f}s, {el/done:.1f}s/ep, ~{el/done*(len(mine)-done)/60:.0f}min left)",flush=True)
    print(f"[rank{a.rank}] DONE {done} eps in {(time.time()-t0)/60:.1f}min",flush=True)
if __name__=="__main__": main()
