"""缓存 kai0_base 251 个挖矿 ep(crave_kai0bd 里 e<100000)的 3Hz 224 帧, 供端到端 TCC。幂等。"""
import json
from pathlib import Path
import numpy as np, av
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
BASE = REPO / "kai0/data/Task_A/kai0_base"
FC = REPO / "temp/crave_kai0bd/feat_cache"
OUT = REPO / "temp/tcc_e2e_frames/kai0base"; OUT.mkdir(parents=True, exist_ok=True)
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
eps = sorted(e for e in (int(p.stem[2:]) for p in FC.glob("ep*.npz")) if e < 100000)
print(f"caching {len(eps)} base eps -> {OUT}", flush=True)
def cam(e): return BASE / "videos" / f"chunk-{e//csB:03d}" / "observation.images.top_head" / f"episode_{e:06d}.mp4"
done = 0
for e in eps:
    fp = OUT / f"ep{e}.npz"
    if fp.exists(): done += 1; continue
    p = cam(e)
    if not p.is_file(): print(f"  [miss] ep{e}"); continue
    fr = []
    c = av.open(str(p))
    for i, f in enumerate(c.decode(video=0)):
        if i % 10: continue
        s = 224 / min(f.height, f.width)
        g = f.reformat(width=round(f.width*s), height=round(f.height*s), format="rgb24")
        im = g.to_ndarray(format="rgb24"); hh, ww = im.shape[:2]; y, xx = (hh-224)//2, (ww-224)//2
        fr.append(im[y:y+224, xx:xx+224])
    c.close(); np.savez_compressed(fp, frames=np.stack(fr).astype(np.uint8)); done += 1
    if done % 25 == 0: print(f"  {done}/{len(eps)}", flush=True)
print(f"DONE {done}/{len(eps)}", flush=True)
