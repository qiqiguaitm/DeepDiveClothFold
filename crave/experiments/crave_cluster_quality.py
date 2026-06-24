"""聚类质量对比:DINOv2-large vs 全-Wan2.2-latent,谁的 milestone 更"按任务相位"(决定要不要上 8 卡全 Wan)。
同一批 ep:train 聚类 → held-out test 上 value=最近簇 tpos,测 corr(value,时间)+ 单调性 + 时间纯度。
高 corr / 高单调 / 低簇内时间std = milestone 更贴进度 = "提升"。
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_cluster_quality.py [--n 140]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
import pandas as pd

from crave.config import REPO
from crave.encoders import load_encoder

dev = "cuda"

# TODO(crave-lib): kai0_base raw-video frame grabber + parallel decode (DS/cs/camp/crop224/grab_ep/decode_images/n30)
#                  should move into crave.data (a "kai0_base" DatasetConfig + frame-grab loader).
#                  Re-inlined here verbatim from crave_decoder_scale_ablation.
DS = REPO / "kai0/data/Task_A/kai0_base"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]


def n30(e):
    return len(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["timestamp"]))


def camp(e):
    return DS / "videos" / f"chunk-{e//cs:03d}" / "observation.images.top_head" / f"episode_{e:06d}.mp4"


def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])


def grab_ep(e, frames30):
    want = set(int(f) for f in frames30); out = {}
    try:
        c = av.open(str(camp(e)))
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = crop224(f.to_ndarray(format="rgb24"))
                if len(out) == len(want): break
        c.close()
    except Exception:
        pass
    return out


def decode_images(pool_idx, E, FR, t0, workers=32):
    """并行(56核)按 ep 解码 224 crop —— 解决单进程 pyav 瓶颈。返回 imgs224(N,224,224,3) + valid mask。"""
    by_ep = {}
    for k, i in enumerate(pool_idx): by_ep.setdefault(int(E[i]), []).append((k, int(FR[i])))
    imgs224 = np.zeros((len(pool_idx), 224, 224, 3), np.uint8); valid = np.zeros(len(pool_idx), bool)

    def work(item):
        e, kfs = item; fm = grab_ep(e, [f for _, f in kfs])
        return [(k, fm[f]) for k, f in kfs if f in fm]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(work, list(by_ep.items())):
            for k, im in res: imgs224[k] = im; valid[k] = True
            done += 1
            if done % 80 == 0: print(f"    decoded {done}/{len(by_ep)} eps  ({time.time()-t0:.0f}s)", flush=True)
    return imgs224, valid


def quality(feat, lab_tr, tpos, E, T, tr_mask, te_mask, cen):
    # held-out: 每 test 帧 value = 最近簇中心的 tpos; 比 corr/单调/纯度
    te = np.where(te_mask)[0]
    d = np.linalg.norm(feat[te][:, None] - cen[None], axis=2); val = tpos[d.argmin(1)]
    cors, mons = [], []
    for e in np.unique(E[te]):
        m = E[te] == e; v = val[m]; t = T[te][m]
        if v.std() > 1e-6 and len(v) > 3:
            cors.append(np.corrcoef(v, t)[0, 1]); mons.append(np.mean(np.diff(v) >= -1e-6))
    # 时间纯度(train 簇内时间 std,越低越"单相位")
    pur = np.mean([T[tr_mask][lab_tr == c].std() for c in np.unique(lab_tr) if (lab_tr == c).sum() > 2])
    return dict(corr=round(float(np.mean(cors)), 3), mono=round(float(np.mean(mons)), 3), time_std=round(float(pur), 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=140); ap.add_argument("--k", type=int, default=96); a = ap.parse_args(); t0 = time.time()
    all_eps = sorted(int(p.stem.split("_")[1]) for p in (DS / "data").glob("chunk-*/episode_*.parquet"))
    eps = sorted(np.random.RandomState(0).permutation(all_eps)[:a.n].tolist())
    te_eps = set(eps[-25:])                       # 25 ep held-out
    E_, FR_, T_ = [], [], []
    for e in eps:
        n = max(1, n30(e) // 10)
        for i in range(n): E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_, np.float32); N = len(E_)
    te_mask = np.isin(E_, list(te_eps)); tr_mask = ~te_mask
    print(f"{len(eps)} ep ({N} 帧), test {len(te_eps)} ep; 解码 ...", flush=True)
    imgs224, valid = decode_images(np.arange(N), E_, FR_, t0)
    vi = np.where(valid)[0]

    from sklearn.cluster import KMeans
    res = {}

    # --- DINOv2-large pooled ---
    print("DINOv2-large pooled ...", flush=True)
    enc = load_encoder("dinov2-large")
    Dn = np.zeros((N, enc.dim), np.float32)
    Dn[vi] = enc.encode_pooled([imgs224[i] for i in vi], bs=64)
    enc.unload(); Dn /= (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-9)
    trv = np.where(tr_mask & valid)[0]
    km = KMeans(a.k, n_init=3, random_state=0).fit(Dn[trv]); cen = km.cluster_centers_
    labtr = np.full(N, -1); labtr[trv] = km.labels_
    tpos = np.array([T_[trv][km.labels_ == c].mean() for c in range(a.k)])
    res["DINOv2-large"] = quality(Dn, km.labels_, tpos, E_, T_, trv, te_mask & valid, cen)
    print("  DINOv2:", res["DINOv2-large"], flush=True)

    # --- Wan2.2 latent ---
    print("Wan2.2 latent ...", flush=True)
    wenc = load_encoder("wan-vae")
    Wn = np.zeros((N, wenc.dim), np.float32)
    Wn[vi] = wenc.encode_pooled([imgs224[i] for i in vi], bs=16)
    wenc.unload()
    mu = Wn[vi].mean(0); sd = Wn[vi].std(0) + 1e-6; Wz = (Wn - mu) / sd
    km2 = KMeans(a.k, n_init=3, random_state=0).fit(Wz[trv]); cen2 = km2.cluster_centers_
    tpos2 = np.array([T_[trv][km2.labels_ == c].mean() for c in range(a.k)])
    res["Wan2.2-latent"] = quality(Wz, km2.labels_, tpos2, E_, T_, trv, te_mask & valid, cen2)
    print("  Wan:", res["Wan2.2-latent"], flush=True)

    res["_note"] = "corr/mono 越高越好(value 贴进度), time_std 越低越好(簇=单相位). 决定全 Wan 是否提升 milestone 质量."
    json.dump(res, open(REPO / "temp/crave_a1a2/cluster_quality.json", "w"), indent=2, ensure_ascii=False)
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
