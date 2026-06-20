"""聚类质量对比:DINOv2-large vs 全-Wan2.2-latent,谁的 milestone 更"按任务相位"(决定要不要上 8 卡全 Wan)。
同一批 ep:train 聚类 → held-out test 上 value=最近簇 tpos,测 corr(value,时间)+ 单调性 + 时间纯度。
高 corr / 高单调 / 低簇内时间std = milestone 更贴进度 = "提升"。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_cluster_quality.py [--n 140]
"""
import sys, argparse, time, json
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from crave_decoder_scale_ablation import REPO, DS, cs, decode_images
from diffusers import AutoencoderKLWan
import pandas as pd
WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; LARGE = "/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large"; dev = "cuda"


def n30(e):
    return len(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["timestamp"]))


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
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(LARGE); enc = AutoModel.from_pretrained(LARGE).to(dev).eval()
    Dn = np.zeros((N, 1024), np.float32)
    for b in range(0, len(vi), 64):
        bb = vi[b:b + 64]
        with torch.no_grad():
            px = proc(images=[imgs224[i] for i in bb], return_tensors="pt").to(dev)
            tok = enc(**px).last_hidden_state[:, 1:].mean(1)
        for k_, i in enumerate(bb): Dn[i] = tok[k_].float().cpu().numpy()
    del enc; torch.cuda.empty_cache(); Dn /= (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-9)
    trv = np.where(tr_mask & valid)[0]
    km = KMeans(a.k, n_init=3, random_state=0).fit(Dn[trv]); cen = km.cluster_centers_
    labtr = np.full(N, -1); labtr[trv] = km.labels_
    tpos = np.array([T_[trv][km.labels_ == c].mean() for c in range(a.k)])
    res["DINOv2-large"] = quality(Dn, km.labels_, tpos, E_, T_, trv, te_mask & valid, cen)
    print("  DINOv2:", res["DINOv2-large"], flush=True)

    # --- Wan2.2 latent ---
    print("Wan2.2 latent ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    Wn = np.zeros((N, 48 * 16 * 16), np.float32)
    for b in range(0, len(vi), 16):
        bb = vi[b:b + 16]
        x = torch.from_numpy(np.stack([cv2.resize(imgs224[i], (256, 256), interpolation=cv2.INTER_AREA) for i in bb]).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)[:, :, None].to(dev)
        with torch.no_grad():
            e = vae.encode(x); z = (e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent)
        for k_, i in enumerate(bb): Wn[i] = z[k_, :, 0].reshape(-1).cpu().numpy()
        if b % 1600 == 0: print(f"    wan {b}/{len(vi)} ({time.time()-t0:.0f}s)", flush=True)
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
