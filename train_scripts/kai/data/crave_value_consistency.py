"""逐簇 value(进度)分布一致性对比:DINOv2-large vs Wan2.2-VAE-latent 聚类。
回答:为什么 Wan 图最好但 value 顺序最乱 —— 量化每个簇内"成员真实进度"的离散度。
指标(每编码器):
  - 簇内 time(=进度)std/IQR 的分布(median/p90)、"混相位簇"占比(std>0.2)
  - milestone 排序歧义:每帧真实进度最近的 milestone-tpos 是否就是它被分到的簇(mismatch 率)
  - held-out 读出:corr / 单调违反率 / 回退跳变幅度
图:每编码器 全簇按 tpos 排序的成员-进度 boxplot(瘦且不重叠=一致;胖且重叠=乱)。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_value_consistency.py [--n 160]
"""
import sys, argparse, time, json
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, torch
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_decoder_scale_ablation import REPO, DS, cs, decode_images
from diffusers import AutoencoderKLWan
import pandas as pd
WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; LARGE = "/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large"; dev = "cuda"
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"


def n30(e):
    return len(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["timestamp"]))


def analyze(feat, E, T, vi, tr_mask, te_mask, K=96):
    from sklearn.cluster import KMeans
    F = feat[vi].astype(np.float32)
    if F.shape[1] > 4096:  # wan: 标准化
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    trv_local = np.where(tr_mask[vi])[0]
    km = KMeans(K, n_init=3, random_state=0).fit(F[trv_local]); cen = km.cluster_centers_
    lab = km.predict(F); Tv = T[vi]
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else np.nan for c in range(K)])
    # 逐簇进度离散度
    stds = np.array([Tv[lab == c].std() for c in range(K) if (lab == c).sum() > 2])
    iqrs = np.array([np.subtract(*np.percentile(Tv[lab == c], [75, 25])) for c in range(K) if (lab == c).sum() > 2])
    # 排序歧义:每帧真实进度最近的 milestone-tpos 是否=它的簇
    order = np.argsort(np.nan_to_num(tpos, nan=9)); tp_sorted = tpos[order]
    nearest_ms = order[np.argmin(np.abs(Tv[:, None] - np.nan_to_num(tpos, nan=9)[None]), axis=1)]
    mism = float(np.mean(nearest_ms != lab))
    # held-out 读出一致性
    te_local = np.where(te_mask[vi])[0]
    d = np.linalg.norm(F[te_local][:, None] - cen[None], axis=2); val = tpos[d.argmin(1)]
    Ete = E[vi][te_local]; Tte = Tv[te_local]
    cors, monv, backjump = [], [], []
    for e in np.unique(Ete):
        m = Ete == e; v = val[m]; t = Tte[m]
        if np.nanstd(v) > 1e-6 and len(v) > 3:
            cors.append(np.corrcoef(v, t)[0, 1]); dv = np.diff(v)
            monv.append(np.mean(dv < -1e-6)); backjump.append(-dv[dv < 0].sum() if (dv < 0).any() else 0.0)
    return dict(lab=lab, Tv=Tv, tpos=tpos,
                cluster_time_std_med=round(float(np.median(stds)), 3), cluster_time_std_p90=round(float(np.percentile(stds, 90)), 3),
                frac_mixed_std_gt0p2=round(float(np.mean(stds > 0.2)), 3),
                cluster_time_iqr_med=round(float(np.median(iqrs)), 3),
                order_mismatch=round(mism, 3),
                heldout_corr=round(float(np.mean(cors)), 3), heldout_mono_violation=round(float(np.mean(monv)), 3),
                heldout_backjump=round(float(np.mean(backjump)), 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=160); a = ap.parse_args(); t0 = time.time()
    all_eps = sorted(int(p.stem.split("_")[1]) for p in (DS / "data").glob("chunk-*/episode_*.parquet"))
    eps = sorted(np.random.RandomState(0).permutation(all_eps)[:a.n].tolist()); te = set(eps[-30:])
    E_, FR_, T_ = [], [], []
    for e in eps:
        n = max(1, n30(e) // 10)
        for i in range(n): E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_, np.float32); N = len(E_)
    te_mask = np.isin(E_, list(te)); tr_mask = ~te_mask
    print(f"{len(eps)} ep ({N}帧), test30; 解码 ...", flush=True)
    imgs224, valid = decode_images(np.arange(N), E_, FR_, t0); vi = np.where(valid)[0]

    # DINOv2
    print("DINOv2-large ...", flush=True)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(LARGE); enc = AutoModel.from_pretrained(LARGE).to(dev).eval()
    Dn = np.zeros((N, 1024), np.float32)
    for b in range(0, len(vi), 64):
        bb = vi[b:b + 64]
        with torch.no_grad():
            px = proc(images=[imgs224[i] for i in bb], return_tensors="pt").to(dev)
            Dn[bb] = enc(**px).last_hidden_state[:, 1:].mean(1).float().cpu().numpy()
    del enc; torch.cuda.empty_cache()
    rd = analyze(Dn, E_, T_, vi, tr_mask, te_mask)

    # Wan latent
    print("Wan2.2 latent ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    Wn = np.zeros((N, 48 * 16 * 16), np.float32)
    for b in range(0, len(vi), 16):
        bb = vi[b:b + 16]
        x = torch.from_numpy(np.stack([cv2.resize(imgs224[i], (256, 256), interpolation=cv2.INTER_AREA) for i in bb]).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)[:, :, None].to(dev)
        with torch.no_grad():
            e = vae.encode(x); z = e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent
        Wn[bb] = z[:, :, 0].reshape(len(bb), -1).cpu().numpy()
        if b % 3200 == 0: print(f"  wan {b}/{len(vi)} ({time.time()-t0:.0f}s)", flush=True)
    rw = analyze(Wn, E_, T_, vi, tr_mask, te_mask)

    summ = {k: {kk: v for kk, v in d.items() if kk not in ("lab", "Tv", "tpos")} for k, d in [("DINOv2-large", rd), ("Wan2.2-latent", rw)]}
    summ["_note"] = "簇内 time std/iqr 低 + mismatch 低 + corr 高 + 单调违反低 = value 一致. 看 Wan 哪项差."
    json.dump(summ, open(REPO / "temp/crave_a1a2/value_consistency.json", "w"), indent=2, ensure_ascii=False)
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)

    # 图:每编码器 全簇按 tpos 排序的成员-进度 boxplot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for ax, (name, d) in zip(axes, [("DINOv2-large (semantic)", rd), ("Wan2.2-VAE latent (appearance)", rw)]):
        lab, Tv, tpos = d["lab"], d["Tv"], d["tpos"]
        order = np.argsort(np.nan_to_num(tpos, nan=9))
        data = [Tv[lab == c] for c in order if (lab == c).sum() > 2]
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True)
        for p in bp["boxes"]: p.set_facecolor("#4c78a8"); p.set_alpha(0.6)
        ax.plot(range(1, len(data) + 1), [np.nan_to_num(tpos, nan=9)[c] for c in order if (lab == c).sum() > 2], "r.-", ms=3, lw=0.8, label="cluster tpos (=value)")
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel("member true progress (per-ep time)")
        ax.set_title(f"{name}  |  cluster time-std med={d['cluster_time_std_med']} p90={d['cluster_time_std_p90']}  mixed(std>0.2)={d['frac_mixed_std_gt0p2']}  order-mismatch={d['order_mismatch']}  held-out corr={d['heldout_corr']} mono-viol={d['heldout_mono_violation']}", fontsize=9)
        ax.set_xlabel("cluster (sorted by tpos)"); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Per-cluster VALUE (progress) consistency — tight non-overlapping boxes = consistent value; fat/overlapping = inconsistent", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTV / "crave_value_consistency.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED crave_value_consistency.png  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
