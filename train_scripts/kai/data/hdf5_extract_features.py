#!/usr/bin/env python
"""HDF5 → (raw-DINOv2 patch-mean ⊕ armmask-DINOv2 ⊕ qpos) 特征提取 (3Hz).
xvla_soft_fold 等原始 HDF5 数据集(cam_high JPEG + qpos 14维)的 V2.4 特征前端,
一次 DINOv2 前向同时产 raw + armmask 两路(与 smooth800 lerobot pipeline 同构)。

用法: python hdf5_extract_features.py --dir <hdf5_dir> --out <cache_dir> [--stride 10] [--limit N]
产物: <out>/ep{idx}.npz  (raw=(T,384) armmask=(T,384) state=(T,14))  idx=episode_<idx>.hdf5
"""
import argparse, colorsys, glob, os, re
from pathlib import Path
import numpy as np, cv2, torch

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROTO = np.load(REPO / "temp/armmask/arm_prototypes.npz")["proto"]   # (K,384) L2-normed
THR = 0.6
P = 16  # 224/14 patch grid


def crop224(img_bgr):
    h, w = img_bgr.shape[:2]
    s = 224 / min(h, w)
    r = cv2.resize(img_bgr, (round(w * s), round(h * s)))
    hh, ww = r.shape[:2]
    rgb = r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2][:, :, ::-1]  # BGR→RGB
    return np.ascontiguousarray(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="max episodes (0=all)")
    a = ap.parse_args()
    import h5py
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = sorted(glob.glob(os.path.join(a.dir, "episode_*.hdf5")),
                 key=lambda p: int(re.search(r"episode_(\d+)", p).group(1)))
    if a.limit:
        eps = eps[:a.limit]
    print(f"[hdf5-extract] {len(eps)} eps  dir={a.dir}  thr={THR} stride={a.stride}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    proto_t = torch.from_numpy(PROTO).float().to(dev)

    def feats(imgs):
        raw, arm = [], []
        with torch.no_grad():
            for b in range(0, len(imgs), 32):
                batch = imgs[b:b + 32]
                px = proc(images=batch, return_tensors="pt").to(dev)
                toks = enc(**px).last_hidden_state[:, 1:]               # (B,256,384)
                raw.append(torch.nn.functional.normalize(toks.mean(1), dim=-1).cpu().numpy())
                tn = torch.nn.functional.normalize(toks, dim=-1)
                sim = (tn @ proto_t.T).max(-1).values                    # (B,256)
                om = []
                for im in batch:
                    rgb = im.reshape(P, 14, P, 14, 3).mean((1, 3)) / 255.0
                    hsv = np.array([[colorsys.rgb_to_hsv(*rgb[i, j]) for j in range(P)] for i in range(P)])
                    om.append(((hsv[..., 0] > 0.02) & (hsv[..., 0] < 0.12) &
                               (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.25)).reshape(-1))
                om = torch.from_numpy(np.stack(om)).to(dev)
                keep = (~((sim > THR) | om)).float().unsqueeze(-1)
                emb = (toks * keep).sum(1) / keep.sum(1).clamp(min=8)
                arm.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(raw), np.concatenate(arm)

    skipped = 0
    for n, fp in enumerate(eps):
        idx = int(re.search(r"episode_(\d+)", fp).group(1))
        p = out / f"ep{idx}.npz"
        if p.exists():
            continue
        try:
            with h5py.File(fp, "r") as h:
                T = h["observations/qpos"].shape[0]
                sel = np.arange(0, T, a.stride)
                imgs = [crop224(cv2.imdecode(np.frombuffer(h["observations/images/cam_high"][i], np.uint8),
                                             cv2.IMREAD_COLOR)) for i in sel]
                state = h["observations/qpos"][:][sel]
            r, m = feats(imgs)
            np.savez_compressed(p, raw=r.astype(np.float32), armmask=m.astype(np.float32),
                                state=state.astype(np.float32))
        except Exception as e:
            skipped += 1
            print(f"  [skip] {fp}: {type(e).__name__} {str(e)[:80]}", flush=True)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(eps)}", flush=True)
    print(f"[hdf5-extract] done, skipped={skipped} → {out}", flush=True)


if __name__ == "__main__":
    main()
