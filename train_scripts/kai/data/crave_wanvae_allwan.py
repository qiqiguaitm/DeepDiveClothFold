"""全 Wan2.2 链路验证(编码器+解码器都用 Wan VAE):Wan 编码 → 在 Wan latent 聚类 → Wan 解码簇中心。
回答用户:"编码器解码器全用 WAN2.2 这一套" 是否给出合理/更好的 milestone + 簇中心。
本地小样(默认 120 ep)先验证语义是否成立, 再决定是否上 8 卡全量(3055 ep)。
三行:① Wan latent 平均→解码(合成质心) ② Wan medoid 解码(锐利) ③ 最近真实帧。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_wanvae_allwan.py [--mine-n 120]
"""
import sys, argparse, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from crave_decoder_scale_ablation import REPO, DS, cs, decode_images
from diffusers import AutoencoderKLWan
import json, av
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"
WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; dev = "cuda"


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def n30(e):  # 30fps 帧数 → 3Hz 取样
    import pandas as pd
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    return len(pd.read_parquet(pq, columns=["timestamp"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mine-n", type=int, default=120); a = ap.parse_args(); t0 = time.time()
    all_eps = sorted(int(p.stem.split("_")[1]) for p in (DS / "data").glob("chunk-*/episode_*.parquet"))
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    E_, FR_, T_ = [], [], []
    for e in mined:
        n = max(1, n30(e) // 10)
        for i in range(n): E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_, np.float32); N = len(E_)
    print(f"全 Wan 验证: {len(mined)} ep, {N} 帧 @3Hz; 并行解码 ...", flush=True)
    imgs224, valid = decode_images(np.arange(N), E_, FR_, t0)
    print(f"  解码完成 ({time.time()-t0:.0f}s); 加载 Wan VAE 编码全部帧 ...", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    Z = np.zeros((N, 48, 16, 16), np.float32)
    vi = np.where(valid)[0]
    for b in range(0, len(vi), 16):
        bb = vi[b:b + 16]
        x = torch.from_numpy(np.stack([cv2.resize(imgs224[i], (256, 256), interpolation=cv2.INTER_AREA) for i in bb]).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)[:, :, None].to(dev)
        with torch.no_grad():
            e = vae.encode(x); z = e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent
        for k_, i in enumerate(bb): Z[i] = z[k_, :, 0].cpu().numpy()
        if b % 1600 == 0: print(f"    Wan-enc {b}/{len(vi)} ({time.time()-t0:.0f}s)", flush=True)

    # 在 Wan latent 聚类(flatten + 标准化)
    F = Z.reshape(N, -1)[vi]; mu = F.mean(0); sd = F.std(0) + 1e-6; Fz = (F - mu) / sd
    from sklearn.cluster import KMeans
    print("KMeans-96 on Wan latent ...", flush=True)
    km = KMeans(96, n_init=3, random_state=0).fit(Fz); labv = km.labels_
    lab = np.full(N, -1); lab[vi] = labv
    Tv = T_[vi]; Ev = E_[vi]
    tpos = np.array([Tv[labv == c].mean() if (labv == c).any() else 0 for c in range(96)])
    cov = np.array([len(set(Ev[labv == c].tolist())) / len(mined) if (labv == c).any() else 0 for c in range(96)])
    tau = otsu(cov); selall = [c for c in range(96) if cov[c] >= tau]
    # 单调性诊断:milestone 时间序是否与覆盖一致
    sel = sorted(selall, key=lambda c: tpos[c]); NS = min(12, len(sel))
    selshow = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]
    print(f"自适应 milestone {len(selall)} (tau={tau:.3f}); tpos 范围 {tpos[selall].min():.2f}-{tpos[selall].max():.2f}; 渲染 {NS}", flush=True)

    def wan_dec(zc):
        with torch.no_grad(): o = vae.decode(torch.from_numpy(zc[None, :, None]).to(dev)).sample
        return np.clip((o[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)

    rows = {"avg": [], "med": [], "near": []}
    cglobal = km.cluster_centers_
    for c in selshow:
        idx_local = np.where(labv == c)[0]; idx = vi[idx_local]      # idx = 全局帧索引
        d = np.linalg.norm(Fz[idx_local] - cglobal[c], axis=1)
        md = vi[idx_local[int(np.argmin(d))]]                        # medoid 全局索引
        zc = Z[idx].mean(0)                       # latent 平均(原始空间)
        rows["avg"].append(wan_dec(zc)); rows["med"].append(wan_dec(Z[md]))
        rows["near"].append(cv2.resize(imgs224[md], (256, 256), interpolation=cv2.INTER_AREA))
    print(f"  渲染完成 ({time.time()-t0:.0f}s); 出图 ...", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = ["(1) Wan latent-AVG\n(synthetic centroid)", "(2) Wan medoid\n(sharp)", "(3) nearest real frame"]
    keys = ["avg", "med", "near"]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 4.9))
    for r, k in enumerate(keys):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[k][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[selshow[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle(f"ALL-Wan2.2 (encode+cluster+decode) — {len(mined)}ep/{N}fr @3Hz, milestones={len(selall)}  [validation before 8-GPU full]", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTV / "crave_wanvae_allwan.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    json.dump({"mine_n": len(mined), "frames": int(N), "milestones": len(selall), "tau": float(tau),
               "tpos_min": float(tpos[selall].min()), "tpos_max": float(tpos[selall].max())},
              open(REPO / "temp/crave_a1a2/allwan_validation.json", "w"), indent=2)
    print(f"SAVED crave_wanvae_allwan.png  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
