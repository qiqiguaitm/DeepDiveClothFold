#!/usr/bin/env python
"""P1: 抽 LIBERO(v3.0) primary 相机的 DINOv3-vitb16(=crave dinov3-base) grid 特征。
不改 crave 共享 loader —— 自己按 v3.0 的 file_index+timestamp 解码。
输出: 每 episode 的 grid 特征 [N, 256, 768] 存 npz。
用法: srpo python p1_libero_dinov3base_extract.py --eps 0,1 --smoke   (冒烟)
       srpo python ... --eps all --out <dir>                          (全量)
"""
import os, sys, argparse, glob
import numpy as np, pandas as pd

ROOT = "/home/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
CAM = "observation.images.image"
CRAVE_SRC = "/home/tim/workspace/deepdive_kai0/lmvla/crave/src"

def load_episode_frames(root, cam, ep_row, stride=1):
    import av
    fi = int(ep_row[f"videos/{cam}/file_index"])
    ci = int(ep_row[f"videos/{cam}/chunk_index"])
    t0 = float(ep_row[f"videos/{cam}/from_timestamp"])
    t1 = float(ep_row[f"videos/{cam}/to_timestamp"])
    mp4 = os.path.join(root, "videos", cam, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
    cont = av.open(mp4)
    stream = cont.streams.video[0]
    # seek 到 episode 起点(避免从文件头重解码整个 mp4)
    try:
        cont.seek(max(0, int((t0 - 0.5) / stream.time_base)), stream=stream, any_frame=False, backward=True)
    except Exception:
        pass
    frames = []; k = 0
    for fr in cont.decode(video=0):
        ts = float(fr.pts * stream.time_base)
        if ts < t0 - 1e-4: continue
        if ts > t1 - 1e-4: break
        if k % stride == 0:
            frames.append(fr.to_ndarray(format="rgb24"))
        k += 1
    cont.close()
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0,1", help="all | a,b,c | start:end")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base")
    ap.add_argument("--encoder", default="dinov3-base")
    args = ap.parse_args()

    epm = pd.read_parquet(sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))[0])
    print(f"[meta] {len(epm)} episodes; length range {epm['length'].min()}-{epm['length'].max()}", flush=True)

    allep = list(epm["episode_index"])
    if args.eps == "all":
        eps = allep
    elif ":" in args.eps:
        a, b = args.eps.split(":"); eps = allep[int(a):int(b)]
    else:
        eps = [int(x) for x in args.eps.split(",")]

    sys.path.insert(0, CRAVE_SRC)
    from crave.encoders import load_encoder
    enc = load_encoder(args.encoder, dtype="bf16")
    print(f"[enc] {args.encoder} dim={enc.dim} loaded", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for e in eps:
        row = epm[epm["episode_index"] == e].iloc[0]
        frames = load_episode_frames(ROOT, CAM, row, stride=args.stride)
        n = len(frames)
        grids = enc.encode_grid(frames)          # [N, dim, P, P]
        import numpy as _np, torch
        g = grids.detach().cpu().float().numpy() if hasattr(grids, "detach") else _np.asarray(grids)
        N, D, P, _ = g.shape
        g = g.transpose(0, 2, 3, 1).reshape(N, P*P, D)   # [N, 256, 768]
        if args.smoke:
            print(f"  ep{e}: frames={n} grid={g.shape} (期望 [~{row['length']}, 256, {enc.dim}]) "
                  f"mean={g.astype(np.float32).mean():.3f} std={g.astype(np.float32).std():.3f} maxabs={np.abs(g.astype(np.float32)).max():.3f}", flush=True)
        else:
            np.savez_compressed(os.path.join(args.out, f"ep{e}.npz"), grid=g.astype(np.float16))
            print(f"  ep{e}: saved {g.shape}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
