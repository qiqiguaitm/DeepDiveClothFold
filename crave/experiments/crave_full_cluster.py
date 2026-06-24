"""全量 kai0_base @3Hz 聚类 + Wan2.2 渲染(8 卡 sharded)。两编码器可选:
  --encoder dino : DINOv2-large 池化聚类(推荐, milestone 质量高)
  --encoder wan  : 全 Wan2.2 latent 聚类(对照, 已知 corr 更低)
两者渲染都用 Wan2.2 VAE(medoid + latent 平均)以公平对比。
阶段:
  --stage encode --rank R --world W : 第 R 片帧 decode→编码 → temp/crave_full/<enc>/shard_R.npz
  --stage aggregate                 : 汇总所有 shard → 聚类 → Wan 渲染簇中心 → 出图+bundle
本地小验: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_full_cluster.py --encoder dino --stage encode --rank 0 --world 1 --mine-n 8
          然后 --stage aggregate --encoder dino --mine-n 8
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
import pandas as pd
import torch

from crave.config import REPO
from crave.encoders import load_encoder
from crave.utils import otsu

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; dev = "cuda"; DIMS = {"dino": 1024, "wan": 48 * 16 * 16}
WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"
ENCNAME = {"dino": "dinov2-large", "wan": "wan-vae"}

# TODO(crave-lib): kai0_base raw-video frame grabber + parallel decode + frame index
#                  (DS/cs/camp/crop224/grab_ep/decode_images/n30/build_index) should move into crave.data
#                  (a "kai0_base" DatasetConfig + frame-grab loader). Re-inlined verbatim from
#                  crave_decoder_scale_ablation; these paths/helpers have no library equivalent yet.
DS = REPO / "kai0/data/Task_A/kai0_base"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]


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


def n30(e):
    return len(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["timestamp"]))


def build_index(mine_n):
    all_eps = sorted(int(p.stem.split("_")[1]) for p in (DS / "data").glob("chunk-*/episode_*.parquet"))
    if mine_n and mine_n < len(all_eps):
        all_eps = sorted(np.random.RandomState(0).permutation(all_eps)[:mine_n].tolist())
    E, FR, T = [], [], []
    for e in all_eps:
        n = max(1, n30(e) // 10)
        for i in range(n): E.append(e); FR.append(i * 10); T.append(i / max(1, n - 1))
    return np.array(E), np.array(FR), np.array(T, np.float32)


def stage_encode(a):
    t0 = time.time(); E, FR, T = build_index(a.mine_n); N = len(E)
    od = OUTD / a.encoder; od.mkdir(parents=True, exist_ok=True)
    if a.rank == 0: np.savez(OUTD / f"index_{a.encoder}.npz", E=E, FR=FR, T=T, n=N)
    shard = np.arange(N)[a.rank::a.world]
    print(f"[rank {a.rank}/{a.world}] {a.encoder} shard {len(shard)}/{N} 帧", flush=True)
    bs = 64 if a.encoder == "dino" else 16
    enc = load_encoder(ENCNAME[a.encoder])
    feat = np.zeros((len(shard), DIMS[a.encoder]), np.float16); valid = np.zeros(len(shard), bool)
    for c in range(0, len(shard), a.chunk):
        sub = shard[c:c + a.chunk]
        imgs, val = decode_images(sub, E, FR, t0)
        vi = np.where(val)[0]
        for b in range(0, len(vi), bs):
            bb = vi[b:b + bs]; batch = [imgs[i] for i in bb]
            f = enc.encode_pooled(batch, bs=bs)
            feat[c + bb] = f.astype(np.float16); valid[c + bb] = True
        print(f"[rank {a.rank}] {min(c+a.chunk,len(shard))}/{len(shard)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(od / f"shard_{a.rank}.npz", gidx=shard, feat=feat, valid=valid)
    print(f"[rank {a.rank}] SAVED shard_{a.rank}.npz ({time.time()-t0:.0f}s)", flush=True)


def stage_aggregate(a):
    t0 = time.time(); zf = np.load(OUTD / f"index_{a.encoder}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, DIMS[a.encoder]), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / a.encoder / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; print(f"汇总 {len(vi)}/{N} 有效帧; 聚类 ...", flush=True)
    F = feat[vi].astype(np.float32)
    if a.encoder == "wan":
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))   # 自适应过聚类数: 随帧数 sqrt 缩放 (120ep→96, 全量335k→~318)
    print(f"自适应 K0={K0} (frames={len(vi)}); KMeans ...", flush=True)
    fitn = min(len(vi), 120000); fit_idx = np.random.RandomState(0).choice(len(vi), fitn, replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); cen = km.cluster_centers_
    lab = km.predict(F)                                          # 全量分配
    Tv = T[vi]; Ev = E[vi]; ne = len(set(E.tolist()))
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    # 自适应选"好"milestone: ① 高复现(覆盖率 Otsu) ② 时间纯=单相位(tstd≤P60) ③ 非冗余(进度最小间隔, 同段保覆盖高的)
    tau_cov = otsu(cov); valid_t = tstd[tstd < 9]; tau_pur = float(np.percentile(valid_t, 60)) if len(valid_t) else 9.0
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    gap = max(0.006, 0.5 / max(len(cand), 1))
    sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= gap: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c                  # 同进度段保覆盖更高的代表
    selall = sel; NS = min(14, len(sel))
    selshow = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]
    print(f"K0={K0} cov-Otsu τ={tau_cov:.3f} purity τ={tau_pur:.3f} cand={len(cand)} → milestone {len(sel)}; 加载 Wan VAE ...", flush=True)

    # TODO(crave-lib): Wan2.2-VAE decode (vae.decode latent → RGB) + raw-frame re-encode is not exposed by
    #                  the Encoder interface (only encode_pooled/encode_grid); a crave.render.wan_decode or
    #                  WanVAEEncoder.decode/encode_latents would remove this inlined VAE load + wan_dec.
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()

    def wan_dec(z):
        with torch.no_grad(): o = vae.decode(torch.from_numpy(z[None, :, None]).to(dev)).sample
        return np.clip((o[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)

    rows = {"avg": [], "med": [], "near": []}
    for c in selshow:
        loc = np.where(lab == c)[0]; d = np.linalg.norm(F[loc] - cen[c], axis=1); ord_ = loc[np.argsort(d)][:40]
        g = vi[ord_]; need = {}
        for k, gi in enumerate(g): need.setdefault(int(E[gi]), []).append((int(FR[gi]), k))
        imgs, zs = [], []
        for e, lst in need.items():
            fm = grab_ep(e, [fr for fr, _ in lst])
            for fr, k in lst:
                if fr in fm: imgs.append(cv2.resize(fm[fr], (256, 256), interpolation=cv2.INTER_AREA))
        if not imgs:
            for kk in rows: rows[kk].append(np.zeros((256, 256, 3), np.uint8))
            continue
        zl = []
        for i in range(0, len(imgs), 8):
            x = torch.from_numpy(np.stack(imgs[i:i + 8]).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)[:, :, None].to(dev)
            with torch.no_grad():
                e = vae.encode(x); zz = e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent
            zl.append(zz[:, :, 0].cpu().numpy())
        zl = np.concatenate(zl)
        rows["avg"].append(wan_dec(zl.mean(0))); rows["med"].append(wan_dec(zl[0])); rows["near"].append(imgs[0])
    print(f"渲染完成 ({time.time()-t0:.0f}s); 出图 ...", flush=True)

    from crave.render import setup_mpl
    plt = setup_mpl()
    labels = ["(1) Wan latent-AVG\n(synthetic)", "(2) Wan medoid\n(sharp)", "(3) nearest real"]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 4.9))
    for r, k in enumerate(["avg", "med", "near"]):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[k][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[selshow[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    enc_name = "DINOv2-large cluster" if a.encoder == "dino" else "ALL-Wan2.2 cluster"
    fig.suptitle(f"FULL kai0_base @3Hz ({len(set(E.tolist()))}ep/{len(vi)}fr) — {enc_name} + Wan2.2 render — milestones={len(selall)}", fontsize=11)
    fig.tight_layout(); out = OUTV / f"crave_full_{a.encoder}.png"; fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)

    # ===== LOCKED-method full gallery: ALL milestones as Wan-medoid representatives (DINOv2 cluster + Wan render) =====
    galN = len(sel); meds = []
    for c in sel:
        loc = np.where(lab == c)[0]; d = np.linalg.norm(F[loc] - cen[c], axis=1); gi = vi[loc[int(np.argmin(d))]]
        fm = grab_ep(int(E[gi]), [int(FR[gi])])
        if int(FR[gi]) not in fm:
            meds.append(np.zeros((256, 256, 3), np.uint8)); continue
        img = cv2.resize(fm[int(FR[gi])], (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None].to(dev)
        with torch.no_grad():
            e = vae.encode(x); zz = (e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent)[0, :, 0].cpu().numpy()
        meds.append(wan_dec(zz))
    ncol = min(14, galN); nrow = int(np.ceil(galN / ncol))
    fg, ax2 = plt.subplots(nrow, ncol, figsize=(1.3 * ncol, 1.5 * nrow)); ax2 = np.atleast_2d(ax2)
    for idx in range(nrow * ncol):
        r2, c2 = divmod(idx, ncol); ax = ax2[r2, c2]; ax.axis("off")
        if idx < galN:
            ax.imshow(meds[idx]); ax.set_title(f"m{idx} P={tpos[sel[idx]]:.2f}", fontsize=6)
    fg.suptitle(f"LOCKED: DINOv2-large cluster + Wan2.2-VAE medoid render  |  FULL {len(set(E.tolist()))}ep / {len(vi)}fr @3Hz  |  {len(selall)} milestones  (medoid = nearest-real-to-center, Wan-decoded)", fontsize=11)
    fg.tight_layout(); gout = OUTV / f"crave_full_{a.encoder}_gallery.png"; fg.savefig(gout, dpi=125, bbox_inches="tight"); plt.close(fg)
    print(f"SAVED {gout.name}", flush=True)

    json.dump({"encoder": a.encoder, "ep": len(set(E.tolist())), "frames": int(len(vi)), "milestones": len(selall), "K0": int(K0), "tau_cov": float(tau_cov), "tau_purity": float(tau_pur)},
              open(OUTD / f"summary_{a.encoder}.json", "w"), indent=2)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dino", "wan"], required=True)
    ap.add_argument("--stage", choices=["encode", "aggregate"], required=True)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--mine-n", type=int, default=0); ap.add_argument("--chunk", type=int, default=6000)
    a = ap.parse_args()
    (stage_encode if a.stage == "encode" else stage_aggregate)(a)


if __name__ == "__main__":
    main()
