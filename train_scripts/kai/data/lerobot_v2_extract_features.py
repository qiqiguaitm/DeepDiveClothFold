#!/usr/bin/env python
"""LeRobot v2.0 (per-episode mp4 + per-episode parquet) → (raw ⊕ armmask ⊕ state) 特征, 3Hz.
用于 kai0_base / kai0_dagger 等 v2.0 数据集的 CRAVE 特征前端, 产物与 hdf5_extract_features 同构
(ep*.npz: raw/armmask/state), 供 hdf5_v24_eval.build_model 直接挖矿。
--offset 给输出文件名加偏移, 便于把 base/dagger 混进一个 cache 不撞号(dagger 用 100000+e)。

用法: python lerobot_v2_extract_features.py --dataset <ds> --out <cache> [--n 250] [--include 2302] [--offset 0]
"""
import argparse, colorsys, json, glob
from pathlib import Path
import numpy as np, cv2, torch, pandas as pd

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROTO = np.load(REPO / "temp/armmask/arm_prototypes.npz")["proto"]
THR = 0.6; P = 16
CAMS = ("observation.images.top_head", "top_head", "observation.images.cam_high")


def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (round(w * s), round(h * s))); hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n", type=int, default=0, help="random sample size (0=all)")
    ap.add_argument("--include", default="", help="csv eps to force-include")
    ap.add_argument("--offset", type=int, default=0, help="output filename id offset")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import av
    ds = Path(a.dataset); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = json.load(open(ds / "meta/info.json")); cs = info["chunks_size"]
    cam = next(c for c in CAMS if (ds / "videos/chunk-000" / c).is_dir())
    all_eps = sorted((lambda r: r.get("episode_index", r.get("episode_id")))(json.loads(l))
                     for l in open(ds / "meta/episodes.jsonl"))
    inc = [int(x) for x in a.include.split(",") if x.strip()]
    if a.n and a.n < len(all_eps):
        sample = np.random.RandomState(a.seed).permutation(all_eps)[:a.n].tolist()
        eps = sorted(set(sample) | set(inc))
    else:
        eps = all_eps
    print(f"[v2-extract] {len(eps)} eps  cam={cam}  offset={a.offset}  ds={ds.name}", flush=True)

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
                toks = enc(**px).last_hidden_state[:, 1:]
                raw.append(torch.nn.functional.normalize(toks.mean(1), dim=-1).cpu().numpy())
                tn = torch.nn.functional.normalize(toks, dim=-1); sim = (tn @ proto_t.T).max(-1).values
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
    for n, e in enumerate(eps):
        p = out / f"ep{a.offset + e}.npz"
        if p.exists():
            continue
        try:
            mp4 = ds / "videos" / f"chunk-{e//cs:03d}" / cam / f"episode_{e:06d}.mp4"
            c = av.open(str(mp4)); imgs = []
            for i, f in enumerate(c.decode(video=0)):
                if i % a.stride == 0:
                    imgs.append(crop224(f.to_ndarray(format="rgb24")))
            c.close()
            st = np.stack(pd.read_parquet(ds / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet",
                                          columns=["observation.state"])["observation.state"].to_numpy())
            sel = np.minimum(np.arange(len(imgs)) * a.stride, len(st) - 1)
            r, m = feats(imgs); k = min(len(r), len(sel))
            np.savez_compressed(p, raw=r[:k].astype(np.float32), armmask=m[:k].astype(np.float32),
                                state=st[sel][:k].astype(np.float32))
        except Exception as ex:
            skipped += 1; print(f"  [skip] ep{e}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(eps)}", flush=True)
    print(f"[v2-extract] done skipped={skipped} → {out}", flush=True)


if __name__ == "__main__":
    main()
