"""CRAVE 簇中心解码器:训练 latent(796-d frozen DINOv2 三路嵌入) → 图像 的解码器,
把 KMeans 簇中心 latent 解码成"中心图像",对比现有"取最近中心帧"代表法。

思路:CRAVE 的 latent 是 frozen DINOv2 三路均值嵌入(raw384⊕arm384⊕pro28)。簇中心 = 该空间均值。
  训练一个小解码器 D: 796 → 128×128 RGB(用 缓存特征↔真实帧 配对),
  然后 D(簇中心) = 合成的"中心图像"。三方对比回答"能否替代最近帧法":
    ① D(center)  解码中心图(模型从 latent 还原)
    ② pixel-mean 簇内真实帧像素平均(平凡基线)
    ③ nearest    离簇心最近的真实帧(现行法)

数据:kai0_base(kai-only),相机 top_head。短任务,本地 2×A100。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_center_decoder.py [--mine-n 200] [--k 96] [--train-imgs 8000] [--epochs 45]
输出: crave/docs/visualization/centroid_decoder/crave_center_decoder_compare.png
      temp/crave_a1a2/center_decoder_metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from crave.config import REPO, resolve_dataset, viz_dir
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import mkp

OUTV = viz_dir("centroid_decoder")
OUTJ = REPO / "temp/crave_a1a2"
RES = 128
dev = "cuda"


def grab_ep_frames(cfg, e, frames30):
    """单次打开视频, 抓取该 ep 的多个 30fps 帧 → {fr: img(RES,RES,3)}。"""
    # TODO(crave-lib): kai0.grab_ep crops to 224; this script resizes raw frame → RES(128)
    #                  directly (no center-crop), so the per-frame decode stays inlined.
    import av
    want = set(int(f) for f in frames30); out = {}
    try:
        c = av.open(str(kai0.video_path(cfg, e)))
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = cv2.resize(f.to_ndarray(format="rgb24"), (RES, RES), interpolation=cv2.INTER_AREA)
                if len(out) == len(want): break
        c.close()
    except Exception:
        pass
    return out


# ---------------- 解码器: 796 → 3×128×128 ----------------
# TODO(crave-lib): this fc-based latent→image Decoder (796→512·4·4→128²) is NOT the
#                  grid decoder in crave.decoding (16→128); kept inline as it has no
#                  library equivalent.
class Decoder(nn.Module):
    def __init__(self, din=796):
        super().__init__()
        self.fc = nn.Linear(din, 512 * 4 * 4)
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.net = nn.Sequential(up(512, 256), up(256, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())  # 4→8→16→32→64→128

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 512, 4, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--train-imgs", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=45)
    a = ap.parse_args()
    OUTJ.mkdir(parents=True, exist_ok=True); t0 = time.time()
    cfg = resolve_dataset("kai0_base")

    rawset = set(int(p.stem[2:]) for p in Path(cfg.raw_cache).glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in Path(cfg.arm_cache).glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    Sall = [kai0.loadep_tcc(cfg, e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    A, R, S, T, E, FR = [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = kai0.loadep_tcc(cfg, e)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); FR.append(np.arange(n) * 10)
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); FR = np.concatenate(FR)
    G = emb(A, R, S).astype(np.float32); K = a.k
    print(f"frames {len(G)} → KMeans {K}", flush=True)
    km = KMeans(K, n_init=3, random_state=0).fit(G); lab = km.labels_; cen = km.cluster_centers_.astype(np.float32)
    tpos = np.array([T[lab == c].mean() for c in range(K)]); cov = np.array([len(set(E[lab == c].tolist())) / len(mined) for c in range(K)])

    # ---- 采样训练帧, 按 ep 单次解码 ----
    rng = np.random.RandomState(1); idx = rng.choice(len(G), min(a.train_imgs, len(G)), replace=False)
    by_ep = {}
    for i in idx: by_ep.setdefault(int(E[i]), []).append(i)
    print(f"解码 {len(idx)} 训练帧, {len(by_ep)} eps ...", flush=True)
    feats, imgs, owners = [], [], []
    for k_, (e, ii) in enumerate(by_ep.items()):
        fr_map = grab_ep_frames(cfg, e, [FR[i] for i in ii])
        for i in ii:
            img = fr_map.get(int(FR[i]))
            if img is None: continue
            feats.append(G[i]); imgs.append(img); owners.append(i)
        if (k_ + 1) % 40 == 0: print(f"  decoded {k_+1}/{len(by_ep)} eps", flush=True)
    feats = np.array(feats, np.float32); imgs = np.array(imgs, np.float32)
    owners = np.array(owners); olab = lab[owners]
    Y = torch.from_numpy(imgs / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(dev)  # [-1,1], NCHW
    print(f"训练对 {len(imgs)}  ({time.time()-t0:.0f}s)", flush=True)

    # 特征模式: raw=只用头部视角全图描述子(384); full=raw+armmask+proprio(796)
    MODES = [("raw_headview_384", slice(0, 384)), ("full_796", slice(0, 796))]

    def train_dec(Fsub):
        Xg = torch.from_numpy(np.ascontiguousarray(Fsub)).to(dev)
        D = Decoder(Fsub.shape[1]).to(dev)
        opt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
        n = len(Xg); bs = 128
        for ep in range(a.epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0
            for b in range(0, n, bs):
                bi = perm[b:b + bs]; pred = D(Xg[bi]); loss = (pred - Y[bi]).abs().mean() + 0.5 * ((pred - Y[bi]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(bi)
            if (ep + 1) % 15 == 0: print(f"    epoch {ep+1}/{a.epochs} L1+MSE {tot/n:.4f}", flush=True)
        D.eval()
        def dec(z):
            with torch.no_grad():
                o = D(torch.from_numpy(np.atleast_2d(z).astype(np.float32)).to(dev)).cpu().numpy()
            return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
        return dec

    decs = {}
    for name, sl in MODES:
        print(f"  训练 decoder [{name}] dim={sl.stop-sl.start} ...", flush=True)
        decs[name] = (train_dec(feats[:, sl]), sl)

    # ---- 选 ~12 个高覆盖簇按进度排 ----
    sel = [c for c in range(K) if cov[c] >= np.quantile(cov, 0.6)]
    sel = sorted(sel, key=lambda c: tpos[c]); NS = min(12, len(sel))
    sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]

    # 每模式: 解码中心 + 重建保真
    metrics = {"train_pairs": len(imgs), "epochs": a.epochs, "K": K, "n_selected": NS, "modes": {}}
    dec_rows = {}
    for name, (dec, sl) in decs.items():
        dec_rows[name] = dec(np.array([cen[c][sl] for c in sel]))
        rp = dec(feats[:8, sl]); metrics["modes"][name] = {"recon_L1_8frames": round(float(np.mean(np.abs(rp.astype(float) - imgs[:8]))) / 255, 4)}
    # nearest real frame per cluster (现行法)
    nearest = []
    for c in sel:
        gi = np.where(lab == c)[0]; nn_i = gi[np.argmin(np.linalg.norm(G[gi] - cen[c], axis=1))]
        fm = grab_ep_frames(cfg, int(E[nn_i]), [FR[nn_i]]); img = fm.get(int(FR[nn_i]))
        nearest.append(img if img is not None else np.zeros((RES, RES, 3), np.uint8))
    json.dump(metrics, open(OUTJ / "center_decoder_metrics.json", "w"), indent=2)
    print("METRICS", metrics, flush=True)

    # ---- 对比图: decoded(raw-headview) vs decoded(full) vs nearest ----
    plt = setup_mpl()
    row_specs = [(f"(1) decoded · raw head-view 384\nrecon L1={metrics['modes']['raw_headview_384']['recon_L1_8frames']:.3f}", dec_rows["raw_headview_384"]),
                 (f"(2) decoded · full latent 796\nrecon L1={metrics['modes']['full_796']['recon_L1_8frames']:.3f}", dec_rows["full_796"]),
                 ("(3) nearest real frame\n(current method)", nearest)]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 5.2))
    for r, (label, imgrow) in enumerate(row_specs):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(imgrow[j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle("Cluster-center representative — does head-view-only (384) decode cleaner than full latent (796)? vs nearest real frame", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTV / "crave_center_decoder_compare.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_center_decoder_compare.png  total", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
