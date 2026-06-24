#!/usr/bin/env python
"""LeRobot v3.0 (concatenated mp4 + 单 parquet) → (raw ⊕ armmask ⊕ state) 特征, 3Hz.
用于互联网真实数据集(如 lerobot/aloha_static_coffee)的 V2.4 泛化验证前端。
与 hdf5_extract_features.py 产物同构 (ep*.npz: raw/armmask/state), 复用同一 V2.4 eval。

用法: python lerobot_v3_extract_features.py --repo-dir <dl> --cam observation.images.cam_high \
        --out <cache> [--stride 16] [--limit N]
"""
import argparse, colorsys, glob
from pathlib import Path
import numpy as np, cv2, torch, pandas as pd

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROTO = np.load(REPO / "temp/armmask/arm_prototypes.npz")["proto"]
THR = 0.6
P = 16


def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (round(w * s), round(h * s))); hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--cam", default="observation.images.cam_high")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import av
    rd = Path(a.repo_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    epm = pd.read_parquet(glob.glob(str(rd / "meta/episodes/**/*.parquet"), recursive=True)[0])
    data = pd.read_parquet(glob.glob(str(rd / "data/**/*.parquet"), recursive=True)[0],
                           columns=["observation.state"])
    ST = np.stack(data["observation.state"].to_numpy())   # (total,14)
    epm = epm.sort_values("episode_index").reset_index(drop=True)
    rng = {int(r.episode_index): (int(r.dataset_from_index), int(r.dataset_to_index)) for r in epm.itertuples()}
    eps = sorted(rng)[: a.limit or None]
    mp4 = glob.glob(str(rd / "videos" / a.cam / "**/*.mp4"), recursive=True)[0]
    print(f"[lerobot-extract] {len(eps)} eps  cam={a.cam}  mp4={Path(mp4).name}  stride={a.stride}", flush=True)

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

    # 单趟解码: 全局帧索引 == data 行号; 按 stride 采样, 路由到所属 episode
    ep_for = {}
    for e in eps:
        f, t = rng[e]
        for gi in range(f, t, a.stride):
            ep_for[gi] = e
    cur = {e: [] for e in eps}
    done = set(); container = av.open(mp4); gi = 0
    total_to = max(rng[e][1] for e in eps)
    for frame in container.decode(video=0):
        if gi in ep_for:
            cur[ep_for[gi]].append(crop224(frame.to_ndarray(format="rgb24")))
        gi += 1
        if gi >= total_to:
            break
    container.close()
    for n, e in enumerate(eps):
        p = out / f"ep{e}.npz"
        if p.exists() or not cur[e]:
            continue
        f0, t0 = rng[e]
        sel = np.arange(f0, t0, a.stride)
        st = ST[sel]
        r, m = feats(cur[e])
        k = min(len(r), len(st))
        np.savez_compressed(p, raw=r[:k].astype(np.float32), armmask=m[:k].astype(np.float32),
                            state=st[:k].astype(np.float32))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(eps)}", flush=True)
    print(f"[lerobot-extract] done → {out}", flush=True)


if __name__ == "__main__":
    main()
