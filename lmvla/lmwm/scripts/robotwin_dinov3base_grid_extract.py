#!/usr/bin/env python
"""[B step1] 抽 robotwin2.0 cam_high DINOv3-base **grid** 特征 [N,256,768] fp16 —— 与 libero_dinov3base 同格式。
vs robotwin_dinov3base_extract.py(pooled): 保留空间网格(不 .mean(spatial)), 供 LMWM generator/InverseEnc 训练 + 策略 provider 目标源。
encode_grid[b,D,P,P] → reshape[b,D,256] → transpose → [b,256,D]=[b,256,768](256=P*P token, 768=D), 与 LIBERO grid 消费一致。
只抽 --eps-file 指定的 episode(pairs 覆盖的 1315 ep, 省空间)。断点续抽。
用法: CUDA_VISIBLE_DEVICES=0 python robotwin_dinov3base_grid_extract.py --eps-file <txt> --shard 0 --nshard 1
Out: lmwm/data/robotwin_dinov3base_grid/ep{e}.npz  key=grid [N,256,768] fp16
"""
import os, sys, glob, argparse, time
import numpy as np, cv2

REPO = "/vePFS/tim/workspace/deepdive_kai0"
DS = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
CAM = "observation.images.cam_high"
CRAVE_SRC = f"{REPO}/lmvla/crave/src"
OUT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base_grid"

def build_ep_map():
    m = {}
    for p in glob.glob(f"{DS}/frame_cache_jpeg256/chunk-*/{CAM}/episode_*.npz"):
        e = int(os.path.basename(p).split("_")[1].split(".")[0]); m[e] = p
    return m

def decode_ep(path):
    d = np.load(path)
    n = len(d.files)
    out = []
    for i in range(n):
        img = cv2.imdecode(d[str(i)], cv2.IMREAD_COLOR)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--eps-file", default=None, help="逗号分隔 ep 列表文件(pairs 覆盖)")
    ap.add_argument("--eps", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    epmap = build_ep_map()
    if a.eps_file:
        want = [int(x) for x in open(a.eps_file).read().strip().split(",") if x]
        eps = [e for e in want if e in epmap]
    elif a.eps:
        eps = [int(x) for x in a.eps.split(",")]
    else:
        eps = sorted(epmap)
    eps = eps[a.shard::a.nshard]
    print(f"[shard {a.shard}/{a.nshard}] {len(eps)} eps (grid)", flush=True)

    sys.path.insert(0, CRAVE_SRC)
    from crave.encoders import load_encoder
    import torch
    enc = load_encoder("dinov3-base", dtype="bf16")
    print(f"[enc] dinov3-base dim={enc.dim} loaded", flush=True)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time(); done = 0; nfr = 0; skipped = 0
    for e in eps:
        op = os.path.join(a.out, f"ep{e}.npz")
        if os.path.exists(op) and not a.smoke:
            skipped += 1; done += 1; continue
        frames = decode_ep(epmap[e]); n = len(frames)
        grids = []
        with torch.no_grad():
            for i in range(0, n, a.bs):
                g = enc.encode_grid(frames[i:i+a.bs])             # [b,D,P,P]
                if hasattr(g, "detach"):
                    b, D, P, _ = g.shape
                    g = g.detach().float().reshape(b, D, P*P).transpose(1, 2).cpu().numpy()  # [b,256,D]
                else:
                    g = np.asarray(g); b, D, P, _ = g.shape
                    g = g.reshape(b, D, P*P).transpose(0, 2, 1)
                grids.append(g.astype(np.float16))
        grid = np.concatenate(grids, 0)                            # [N,256,768]
        if a.smoke:
            print(f"  ep{e}: N={n} grid={grid.shape} ({grid.nbytes/1e6:.1f}MB) mean={grid.astype(np.float32).mean():.3f}", flush=True)
        else:
            np.savez_compressed(op, grid=grid)
        done += 1; nfr += n
        if done % 100 == 0:
            el = time.time()-t0; print(f"  [shard {a.shard}] {done}/{len(eps)} eps {nfr}f | {nfr/max(el,1):.0f} fr/s | ETA {(len(eps)-done)*el/max(done-skipped,1)/60:.0f}min", flush=True)
    print(f"[shard {a.shard}] DONE {done} eps ({skipped} skipped) {nfr}f ({time.time()-t0:.0f}s)", flush=True)
    print(f"SHARD_{a.shard}_GRID_DONE", flush=True)

if __name__ == "__main__":
    main()
