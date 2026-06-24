#!/usr/bin/env python
"""臂掩膜版特征提取(§5.7 缓解a): DINOv2 patch tokens → 剔除臂 patch
(与 arm_prototypes 余弦相似>THR 或橙色线缆)→ 余下 patch 均值 = 帧嵌入。
用法: python extract_masked_features.py --dataset <ds> --out <cache_root> [--shard I N]
产物: <out>/feat_cache/ep{N}.npz (与 tcc 缓存同构, 供 recurrence_full_mining 等复用)
"""
from __future__ import annotations
import argparse, colorsys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROTO = np.load(REPO / "temp/armmask/arm_prototypes.npz")["proto"]   # (K,384) L2-normed
THR = 0.6
P = 16  # 224/14 patch grid
CAM_CANDIDATES = ("observation.images.top_head", "top_head", "observation.images.cam_high")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--shard", type=int, nargs=2, default=None, metavar=("I", "N"))
    args = ap.parse_args()
    ds, out = Path(args.dataset), Path(args.out)
    (out / "feat_cache").mkdir(parents=True, exist_ok=True)
    cam = next(c for c in CAM_CANDIDATES if (ds / "videos/chunk-000" / c).is_dir())
    chunks_size = json.load(open(ds / "meta/info.json")).get("chunks_size", 1000)
    eps = [(lambda r: r.get("episode_index", r.get("episode_id")))(json.loads(l))
           for l in open(ds / "meta/episodes.jsonl")]
    eps = sorted(eps)
    if args.shard:
        i, n = args.shard
        eps = [e for j, e in enumerate(eps) if j % n == i]
    print(f"[mask-extract] {len(eps)} eps  cam={cam}  protos={len(PROTO)} thr={THR}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    proto_t = torch.from_numpy(PROTO).float().to(dev)

    import av
    def episode_feats(ep):
        mp4 = ds / "videos" / f"chunk-{ep // chunks_size:03d}" / cam / f"episode_{ep:06d}.mp4"
        imgs = []
        c = av.open(str(mp4))
        for i, f in enumerate(c.decode(video=0)):
            if i % args.stride:
                continue
            h, w = f.height, f.width
            s = 224 / min(h, w)
            a = f.reformat(width=round(w*s), height=round(h*s), format="rgb24").to_ndarray(format="rgb24")
            hh, ww = a.shape[:2]
            imgs.append(a[(hh-224)//2:(hh+224)//2, (ww-224)//2:(ww+224)//2])
        c.close()
        feats = []
        with torch.no_grad():
            for b in range(0, len(imgs), 32):
                batch = imgs[b:b+32]
                px = proc(images=batch, return_tensors="pt").to(dev)
                toks = enc(**px).last_hidden_state[:, 1:]                 # (B,256,384)
                tn = torch.nn.functional.normalize(toks, dim=-1)
                sim = (tn @ proto_t.T).max(-1).values                      # (B,256)
                # 橙色线缆 patch(逐帧 HSV)
                om = []
                for im in batch:
                    rgb = im.reshape(P, 14, P, 14, 3).mean((1, 3)) / 255.0
                    hsv = np.array([[colorsys.rgb_to_hsv(*rgb[i, j]) for j in range(P)] for i in range(P)])
                    om.append(((hsv[..., 0] > 0.02) & (hsv[..., 0] < 0.12) &
                               (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.25)).reshape(-1))
                om = torch.from_numpy(np.stack(om)).to(dev)
                keep = ~((sim > THR) | om)                                 # (B,256) cloth/table patches
                keep_f = keep.float().unsqueeze(-1)
                denom = keep_f.sum(1).clamp(min=8)                         # 全被 mask 时退化保护
                emb = (toks * keep_f).sum(1) / denom
                feats.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, 384), np.float32)

    skipped = 0
    for n, ep in enumerate(eps):
        p = out / "feat_cache" / f"ep{ep}.npz"
        if p.exists():
            continue
        try:
            f = episode_feats(ep)
            np.savez_compressed(p, f=f)
        except Exception as e:
            skipped += 1
            print(f"  [skip] ep{ep}: {type(e).__name__} {str(e)[:60]}")
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(eps)}")
    print(f"[mask-extract] done, skipped={skipped}")


if __name__ == "__main__":
    main()
