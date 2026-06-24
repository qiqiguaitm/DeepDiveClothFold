"""为 TCC 端到端微调缓存 3Hz 224x224 uint8 帧 (kai0). 幂等(跳过已存在)。
用法: python cache_frames_224.py [n_train]
"""
import json, sys
from pathlib import Path
import numpy as np, av

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_advantage"
OUT = REPO / "temp/tcc_e2e_frames/kai0"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = REPO / "temp/tcc_kai0_armmask/feat_cache"
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
STRIDE = 10

zp = np.load(REPO / "temp/recurrence_v0_kai0/embeddings.npz")
EVAL = sorted(set(zp["ep_ids"].tolist()))
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
pool = np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()
eps = sorted(set(pool[:N] + EVAL))
print(f"caching {len(eps)} eps (train {N} + eval {len(EVAL)}) -> {OUT}")

def cam(e):
    return DS / "videos" / f"chunk-{e // chunks_size:03d}" / "observation.images.top_head" / f"episode_{e:06d}.mp4"

done = 0
for e in eps:
    fp = OUT / f"ep{e}.npz"
    if fp.exists():
        done += 1; continue
    p = cam(e)
    if not p.is_file():
        print(f"  [miss] ep{e}"); continue
    frames = []
    c = av.open(str(p))
    for i, f in enumerate(c.decode(video=0)):
        if i % STRIDE: continue
        s = 224 / min(f.height, f.width)
        g = f.reformat(width=round(f.width * s), height=round(f.height * s), format="rgb24")
        im = g.to_ndarray(format="rgb24")
        hh, ww = im.shape[:2]; y, x = (hh - 224) // 2, (ww - 224) // 2
        frames.append(im[y:y + 224, x:x + 224])
    c.close()
    np.savez_compressed(fp, frames=np.stack(frames).astype(np.uint8))
    done += 1
    if done % 25 == 0:
        print(f"  {done}/{len(eps)}")
print(f"DONE {done}/{len(eps)} cached")
