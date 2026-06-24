"""CRAVE 簇质心 · 重方案(最划算的生成式):pix2pix 式条件 GAN —— patch-grid 解码器 + PatchGAN 判别器。

deep research 结论:可形变布料的"清晰平均"是 ill-posed;真要清晰合成质心,得用生成式从簇中心
"生成"一张连贯清晰图(采样一个 mode),而非平均。全扩散(RCDM)太重 → 用轻量条件 GAN:
  生成器 G = GridDecoder(DINOv2 patch grid 16×16×384 → 128² 图)  (保空间)
  判别器 D = PatchGAN(LSGAN)                                       (把输出逼清晰)
  G_loss = adv + 10·L1   (pix2pix, 稳定且锐)
推理:喂 ① 簇内 grid 平均 ② medoid grid → 生成清晰质心,对比 ③ 最近真实帧。

数据 kai0_base(kai-only),top_head。本地 2×A100 短任务(~12min)。
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_patch_gan.py [--mine-n 200] [--pool 9000] [--epochs 120]
输出: docs/visualization/cross_episode_recurrence_value/crave_patch_gan_{compare,recon}.png + temp/crave_a1a2/patch_gan_metrics.json
"""
import argparse, json, os, time
import numpy as np, pandas as pd, av, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_base"
ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"; RAW = REPO / "temp/tcc_kai0_raw/feat_cache"
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"; OUTJ = REPO / "temp/crave_a1a2"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]; RES = 128; P = 16; DGRID = 384; dev = "cuda"


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


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


class GridDecoder(nn.Module):  # G: grid(384,16,16) → img(3,128,128)
    def __init__(self):
        super().__init__()
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.head = nn.Sequential(nn.Conv2d(DGRID, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.net = nn.Sequential(up(256, 128), up(128, 64), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, g): return self.net(self.head(g))


class PatchD(nn.Module):  # PatchGAN on image
    def __init__(self):
        super().__init__()
        def c(i, o, s=2): return nn.Sequential(spectral_norm(nn.Conv2d(i, o, 4, s, 1)), nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(c(3, 64), c(64, 128), c(128, 256), c(256, 512, 1),
                                 spectral_norm(nn.Conv2d(512, 1, 4, 1, 1)))

    def forward(self, x): return self.net(x)


def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200); ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--pool", type=int, default=9000); ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args(); OUTJ.mkdir(parents=True, exist_ok=True); t0 = time.time()

    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    T, E, FR, A, Rr, Sx = [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = loadep(e); A.append(aa); Rr.append(rr); Sx.append(st)
        T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); FR.append(np.arange(n) * 10)
    A = np.concatenate(A); Rr = np.concatenate(Rr); Sx = np.concatenate(Sx); T = np.concatenate(T); E = np.concatenate(E); FR = np.concatenate(FR)
    G = emb(A, Rr, Sx).astype(np.float32); K = a.k
    km = KMeans(K, n_init=3, random_state=0).fit(G); lab = km.labels_; cen = km.cluster_centers_.astype(np.float32)
    tpos = np.array([T[lab == c].mean() for c in range(K)]); cov = np.array([len(set(E[lab == c].tolist())) / len(mined) for c in range(K)])
    print(f"frames {len(G)} → KMeans {K}  ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(1); pool = rng.choice(len(G), min(a.pool, len(G)), replace=False)
    by_ep = {}
    for i in pool: by_ep.setdefault(int(E[i]), []).append(i)
    from transformers import AutoImageProcessor, AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    print(f"解码+抽 grid: {len(pool)} 帧 ...", flush=True)
    grids, imgs, owners, buf_img, buf_idx = [], [], [], [], []

    def flush():
        if not buf_img: return
        with torch.no_grad():
            px = proc(images=buf_img, return_tensors="pt").to(dev)
            toks = enc(**px).last_hidden_state[:, 1:]
            g = toks.reshape(len(buf_img), P, P, DGRID).permute(0, 3, 1, 2).contiguous().cpu().numpy().astype(np.float16)
        for k_, i in enumerate(buf_idx):
            grids.append(g[k_]); imgs.append(cv2.resize(buf_img[k_], (RES, RES), interpolation=cv2.INTER_AREA)); owners.append(i)
        buf_img.clear(); buf_idx.clear()

    for e, ii in by_ep.items():
        fr_map = grab_ep(e, [FR[i] for i in ii])
        for i in ii:
            im = fr_map.get(int(FR[i]))
            if im is None: continue
            buf_img.append(im); buf_idx.append(i)
            if len(buf_img) == 64: flush()
    flush()
    grids = np.array(grids, np.float16); imgs = np.array(imgs, np.float32); owners = np.array(owners); olab = lab[owners]
    print(f"pool {len(grids)}  ({time.time()-t0:.0f}s)", flush=True)
    mu = grids.mean(axis=(0, 2, 3), dtype=np.float32); sd = grids.astype(np.float32).std(axis=(0, 2, 3)) + 1e-4
    muT = torch.from_numpy(mu).view(1, DGRID, 1, 1).to(dev); sdT = torch.from_numpy(sd).view(1, DGRID, 1, 1).to(dev)
    Y = torch.from_numpy(imgs / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(dev)
    Gg = torch.from_numpy(grids.astype(np.float32)).to(dev)

    # ---- pix2pix 训练 ----
    Gn = GridDecoder().to(dev); Dn = PatchD().to(dev)
    oG = torch.optim.Adam(Gn.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(Dn.parameters(), lr=2e-4, betas=(0.5, 0.999))
    n = len(grids); bs = 64
    # 色彩一致性:低频(=色彩/大结构)用强 L1 锚住, 让对抗只负责高频细节 → 修复 GAN 掉色
    def lowpass(z): return F.avg_pool2d(z, 16, 16)        # 128→8 色块
    for ep in range(a.epochs):
        perm = torch.randperm(n, device=dev); gl = dl = 0.0
        for b in range(0, n, bs):
            bi = perm[b:b + bs]; x = (Gg[bi] - muT) / sdT; real = Y[bi]
            fake = Gn(x)
            # D
            oD.zero_grad(); ld = 0.5 * ((Dn(real) - 1) ** 2).mean() + 0.5 * (Dn(fake.detach()) ** 2).mean()
            ld.backward(); oD.step()
            # G: adv(高频锐) + L1(像素) + 强低频 L1(色彩锚)
            oG.zero_grad()
            lg = ((Dn(fake) - 1) ** 2).mean() + 20.0 * (fake - real).abs().mean() + 50.0 * (lowpass(fake) - lowpass(real)).abs().mean()
            lg.backward(); oG.step(); gl += float(lg) * len(bi); dl += float(ld) * len(bi)
        if (ep + 1) % 20 == 0: print(f"  epoch {ep+1}/{a.epochs}  G {gl/n:.3f}  D {dl/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    Gn.eval()

    def gen(gnp):
        with torch.no_grad():
            x = (torch.from_numpy(np.atleast_3d(gnp).astype(np.float32)).to(dev).view(-1, DGRID, P, P) - muT) / sdT
            o = Gn(x).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    sel = [c for c in range(K) if cov[c] >= np.quantile(cov, 0.6)]
    sel = sorted(sel, key=lambda c: tpos[c]); NS = min(12, len(sel)); sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]
    rows = {"gridavg": [], "medoid": [], "nearest": []}
    for c in sel:
        mem = np.where(olab == c)[0]
        if len(mem):
            rows["gridavg"].append(gen(grids[mem].astype(np.float32).mean(0)[None])[0])
            md = mem[np.argmin(np.linalg.norm(G[owners[mem]] - cen[c], axis=1))]
            rows["medoid"].append(gen(grids[md][None].astype(np.float32))[0])
        else:
            rows["gridavg"].append(np.zeros((RES, RES, 3), np.uint8)); rows["medoid"].append(np.zeros((RES, RES, 3), np.uint8))
        gi = np.where(lab == c)[0]; nn_i = gi[np.argmin(np.linalg.norm(G[gi] - cen[c], axis=1))]
        fm = grab_ep(int(E[nn_i]), [FR[nn_i]]); im = fm.get(int(FR[nn_i]))
        rows["nearest"].append(cv2.resize(im, (RES, RES)) if im is not None else np.zeros((RES, RES, 3), np.uint8))

    rec = gen(grids[:16].astype(np.float32)); rec_l1 = float(np.mean(np.abs(rec.astype(float) - imgs[:16]))) / 255
    # 色彩保真: 16×16 下采样(=色彩/大结构)后的 L1, 越低色越准
    def lp(x): return cv2.resize(x.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
    color_l1 = float(np.mean([np.mean(np.abs(lp(rec[j]) - lp(imgs[j]))) for j in range(16)])) / 255
    sh = {k: round(float(np.mean([sharp(x) for x in rows[k]])), 1) for k in rows}
    metrics = {"pool": len(grids), "epochs": a.epochs, "gan_recon_L1": round(rec_l1, 4),
               "color_L1_lowpass": round(color_l1, 4), "sharpness": sh, "n_selected": NS,
               "note": "色彩锚: lowpass-L1*50 + L1*20; color_L1_lowpass 越低色越准"}
    json.dump(metrics, open(OUTJ / "patch_gan_metrics.json", "w"), indent=2, ensure_ascii=False)
    print("METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)

    labels = [f"(1) GAN gen · grid-AVERAGE\n(synthetic centroid)  sharp={sh['gridavg']}",
              f"(2) GAN gen · medoid grid\n  sharp={sh['medoid']}",
              f"(3) nearest real frame\n(current)  sharp={sh['nearest']}"]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 5.2))
    for r, key in enumerate(["gridavg", "medoid", "nearest"]):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[key][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle(f"pix2pix-GAN patch decoder — sharp synthetic centroid (grid-avg / medoid) vs nearest  (recon L1={rec_l1:.3f})", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTV / "crave_patch_gan_compare.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_patch_gan_compare.png", flush=True)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.2))
    for j in range(8):
        axes[0, j].imshow(imgs[j].astype(np.uint8)); axes[0, j].axis("off")
        axes[1, j].imshow(rec[j]); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=10, rotation=0, ha="right", va="center"); axes[0, 0].axis("on"); axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    axes[1, 0].set_ylabel("GAN", fontsize=10, rotation=0, ha="right", va="center"); axes[1, 0].axis("on"); axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    fig.suptitle(f"pix2pix-GAN reconstruction (real vs G(its grid)), L1={rec_l1:.3f}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTV / "crave_patch_gan_recon.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_patch_gan_recon.png  total", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
