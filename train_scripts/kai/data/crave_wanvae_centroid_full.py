"""全量 kai0_base @3Hz 聚类 → Wan2.2 VAE 渲染簇中心(看更大数据是否让 milestone/质心更好)。
内存友好:分块 decode→DINOv2-large pooled(只留 pooled, 不留 32GB grids);只对入选 milestone 的成员跑 Wan。
三行:① Wan latent 平均(合成质心) ② Wan medoid 解码(锐利) ③ 最近真实帧。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_wanvae_centroid_full.py [--mine-n 550]
"""
import sys, argparse, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from crave_decoder_scale_ablation import REPO, ARM, RAW, loadep, decode_images, grab_ep, RES
from diffusers import AutoencoderKLWan
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"
LARGE = "/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large"; DIM = 1024; dev = "cuda"; P = 16
WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mine-n", type=int, default=550); ap.add_argument("--chunk", type=int, default=6000)
    a = ap.parse_args(); t0 = time.time()
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    E_, FR_, T_ = [], [], []
    for e in mined:
        _, _, _, n = loadep(e)
        for i in range(n): E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_, np.float32); N = len(E_)
    print(f"全量 {len(mined)} ep, {N} 帧 @3Hz; 分块 DINOv2-large pooled ...", flush=True)

    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(LARGE); enc = AutoModel.from_pretrained(LARGE).to(dev).eval()
    pooled = np.zeros((N, DIM), np.float32)
    for s in range(0, N, a.chunk):
        idx = np.arange(s, min(s + a.chunk, N))
        imgs224, valid = decode_images(idx, E_, FR_, t0)
        vi = np.where(valid)[0]
        for b in range(0, len(vi), 64):
            bb = vi[b:b + 64]; batch = [imgs224[i] for i in bb]
            with torch.no_grad():
                px = proc(images=batch, return_tensors="pt").to(dev)
                tok = enc(**px).last_hidden_state[:, 1:].mean(1)            # 池化 = patch tokens 均值
            for k_, i in enumerate(bb): pooled[s + i] = tok[k_].float().cpu().numpy()
        del imgs224; print(f"  pooled {min(s+a.chunk,N)}/{N} ({time.time()-t0:.0f}s)", flush=True)
    del enc; torch.cuda.empty_cache()
    pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)

    from sklearn.cluster import KMeans
    print("KMeans-96 (全量) ...", flush=True)
    km = KMeans(96, n_init=3, random_state=0).fit(pooled); lab = km.labels_; cen = km.cluster_centers_
    tpos = np.array([T_[lab == c].mean() for c in range(96)])
    cov = np.array([len(set(E_[lab == c].tolist())) / len(mined) for c in range(96)])
    tau = otsu(cov); selall = [c for c in range(96) if cov[c] >= tau]
    sel = sorted(selall, key=lambda c: tpos[c]); NS = min(12, len(sel))
    sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]
    print(f"自适应 milestone {len(selall)} (tau={tau:.3f}), 展示 {NS}; 加载 Wan VAE ...", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()

    def wan_enc(imgs256):
        zs = []
        for b in range(0, len(imgs256), 8):
            x = torch.from_numpy(np.stack(imgs256[b:b + 8]).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)[:, :, None].to(dev)
            with torch.no_grad():
                e = vae.encode(x); z = e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent
            zs.append(z.cpu())
        return torch.cat(zs)

    def wan_dec(z):
        with torch.no_grad(): o = vae.decode(z.to(dev)).sample
        return np.clip((o[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)

    rows = {"avg": [], "med": [], "near": []}
    for c in sel:
        mem = np.where(lab == c)[0]; d = np.linalg.norm(pooled[mem] - cen[c], axis=1); order = mem[np.argsort(d)][:40]
        need = {}
        for i in order: need.setdefault(int(E_[i]), []).append((int(FR_[i]), int(i)))
        imgs256, ids = [], []
        for e, lst in need.items():
            fm = grab_ep(e, [f for f, _ in lst])
            for f, gi in lst:
                if f in fm: imgs256.append(cv2.resize(fm[f], (256, 256), interpolation=cv2.INTER_AREA)); ids.append(gi)
        if not imgs256:
            for k in rows: rows[k].append(np.zeros((256, 256, 3), np.uint8));
            continue
        zs = wan_enc(imgs256)
        rows["avg"].append(wan_dec(zs.mean(0, keepdim=True)))
        mpos = int(np.argmin([np.linalg.norm(pooled[g] - cen[c]) for g in ids]))   # medoid = 离 center 最近且成功解码的
        rows["med"].append(wan_dec(zs[mpos:mpos + 1])); rows["near"].append(imgs256[mpos])
    print(f"  渲染完成 ({time.time()-t0:.0f}s); 出图 ...", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = ["(1) Wan-VAE latent-AVG\n(synthetic centroid)", "(2) Wan-VAE medoid\n(sharp)", "(3) nearest real frame"]
    keys = ["avg", "med", "near"]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 4.9))
    for r, k in enumerate(keys):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[k][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle(f"FULL kai0_base @3Hz ({len(mined)}ep/{N}fr) cluster → Wan2.2-VAE centroid: latent-avg / medoid / nearest-real  (milestones={len(selall)})", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTV / "crave_wanvae_centroid_full.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED crave_wanvae_centroid_full.png  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
