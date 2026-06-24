"""CRAVE 簇中心 decoder · 规模消融:更多数据 / 更大解码器 / 更大编码器 是否改善"簇中心(grid 平均)"。

诚实假设:
  - 单帧重建保真:更大 enc/dec + 更多数据 → 会更清晰(常规)。
  - 簇中心(grid 平均)发软:根因是**解码器输入本身**(未对齐+语义多样的布料帧的 grid 平均 = 糊/离流形),
    更多数据治不了(输入问题);更大解码器可能因**更强先验**把离流形平均 grid 拉向清晰,边际可能有;
    更大编码器给更细 per-frame grid(重建/medoid 更好),但平均仍 smear。
  本实验同时量 重建保真 + 簇中心锐度,实测各杠杆是否真的改善"簇中心"。

配置(L1 解码器, 隔离 GAN 混杂):
  A baseline   small-enc(384) + small-dec + 9k
  B +data      small-enc(384) + small-dec + 24k
  C +decoder   small-enc(384) + BIG-dec   + 9k
  D +encoder   base-enc (768) + small-dec + 9k

数据 kai0_base(kai-only),top_head。一进程内跑完 4 配置。
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_decoder_scale_ablation.py [--mine-n 400] [--big-pool 24000] [--epochs 55]
输出: docs/visualization/cross_episode_recurrence_value/crave_scale_ablation.png + temp/crave_a1a2/scale_ablation_metrics.json
"""
import argparse, json, os, time
import numpy as np, pandas as pd, av, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import torch, torch.nn as nn

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_base"
ARM = REPO / "temp/tcc_kai0_armmask/feat_cache"; RAW = REPO / "temp/tcc_kai0_raw/feat_cache"
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"; OUTJ = REPO / "temp/crave_a1a2"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]; RES = 128; P = 16; dev = "cuda"


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


class Dec(nn.Module):
    """三档解码器 16×16→128×128: small(~0.8M) / big(~5M) / xl(~13-15M, 更宽+更深 refine)。"""
    def __init__(self, din, dec="small"):
        super().__init__()
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        if dec == "tiny":
            self.head = nn.Sequential(nn.Conv2d(din, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))
            self.net = nn.Sequential(up(128, 64), up(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())
        elif dec == "small":
            self.head = nn.Sequential(nn.Conv2d(din, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
            self.net = nn.Sequential(up(256, 128), up(128, 64), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
        elif dec == "medium":
            self.head = nn.Sequential(nn.Conv2d(din, 384, 1), nn.BatchNorm2d(384), nn.ReLU(True))
            self.net = nn.Sequential(up(384, 192), up(192, 96), nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh())
        elif dec == "big":
            self.head = nn.Sequential(nn.Conv2d(din, 512, 1), nn.BatchNorm2d(512), nn.ReLU(True))
            self.net = nn.Sequential(up(512, 384), up(384, 192), up(192, 96), cb(96, 96),
                                     nn.Conv2d(96, 3, 3, 1, 1), nn.Tanh())
        elif dec == "xl":
            self.head = nn.Sequential(nn.Conv2d(din, 768, 1), nn.BatchNorm2d(768), nn.ReLU(True))
            self.net = nn.Sequential(up(768, 512), up(512, 384), up(384, 256),
                                     cb(256, 256), cb(256, 128), cb(128, 128),
                                     nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())   # 16→32→64→128 + 深 refine
        else:
            raise ValueError(dec)

    def forward(self, g): return self.net(self.head(g))


def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def decode_images(pool_idx, E, FR, t0, workers=32):
    """并行(56核)按 ep 解码 224 crop —— 解决单进程 pyav 瓶颈。返回 imgs224(N,224,224,3) + valid mask。"""
    from concurrent.futures import ThreadPoolExecutor
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


def encode_grids(imgs224, valid, model_name, dim, nprefix=1, res=224, dtype="fp32"):
    """GPU 批量抽 patch grid (dim,16,16)。仅 valid 帧。nprefix=1(dinov2)/5(dinov3 1CLS+4reg); res=224(dinov2)/256(dinov3)→均 16x16。"""
    from transformers import AutoImageProcessor, AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proc = AutoImageProcessor.from_pretrained(model_name)
    if dtype == "int8":
        from transformers import BitsAndBytesConfig
        enc = AutoModel.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval(); tdt = torch.bfloat16
    elif dtype == "bf16":
        enc = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(dev).eval(); tdt = torch.bfloat16
    else:
        enc = AutoModel.from_pretrained(model_name).to(dev).eval(); tdt = None
    grids = np.zeros((len(imgs224), dim, P, P), np.float16)
    idxs = np.where(valid)[0]; sz = {"height": res, "width": res}
    for b in range(0, len(idxs), 64):
        bi = idxs[b:b + 64]; batch = [imgs224[i] for i in bi]
        with torch.no_grad():
            px = proc(images=batch, return_tensors="pt", size=sz).to(dev)
            if tdt is not None: px = {k: (v.to(tdt) if torch.is_floating_point(v) else v) for k, v in px.items()}
            toks = enc(**px).last_hidden_state[:, nprefix:]
            side = int(round(toks.shape[1] ** 0.5)); assert side == P, f"grid {side}!=P({P}); res={res} patch mismatch"
            g = toks.reshape(len(batch), P, P, dim).permute(0, 3, 1, 2).float().contiguous().cpu().numpy().astype(np.float16)
        for k_, i in enumerate(bi): grids[i] = g[k_]
    del enc; torch.cuda.empty_cache()
    return grids


def train_dec(grids, imgs, din, dec, epochs):
    mu = grids.mean(axis=(0, 2, 3), dtype=np.float32); sd = grids.astype(np.float32).std(axis=(0, 2, 3)) + 1e-4
    muT = torch.from_numpy(mu).view(1, din, 1, 1).to(dev); sdT = torch.from_numpy(sd).view(1, din, 1, 1).to(dev)
    Y = torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(dev)
    Gg = torch.from_numpy(grids.astype(np.float32)).to(dev)
    D = Dec(din, dec).to(dev); opt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n = len(grids); bs = 64
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            bi = perm[b:b + bs]; x = (Gg[bi] - muT) / sdT
            pred = D(x); loss = (pred - Y[bi]).abs().mean() + 0.5 * ((pred - Y[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    D.eval()
    def dec(gnp):
        with torch.no_grad():
            x = (torch.from_numpy(np.atleast_3d(gnp).astype(np.float32)).to(dev).view(-1, din, P, P) - muT) / sdT
            o = D(x).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    del Gg, Y; torch.cuda.empty_cache()
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=400); ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--big-pool", type=int, default=24000); ap.add_argument("--small-pool", type=int, default=9000)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--configs", default="A,B,C,D", help="子集, 逗号分隔 A..E, 支持多 GPU 并行分摊")
    ap.add_argument("--tag", default="main", help="输出文件后缀, 并行时区分")
    a = ap.parse_args(); OUTJ.mkdir(parents=True, exist_ok=True); t0 = time.time()
    # 配置注册表: key → (encoder, dim, big_decoder, pool)
    REG = {
        "A": dict(enc="facebook/dinov2-small", dim=384, dec="small", pool="small", label="A baseline (s384·s-dec·9k)"),
        "B": dict(enc="facebook/dinov2-small", dim=384, dec="small", pool="big", label="B +data (s384·s-dec·24k)"),
        "C": dict(enc="facebook/dinov2-small", dim=384, dec="big", pool="small", label="C +decoder (s384·BIG·9k)"),
        "D": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-base", dim=768, dec="small", pool="small", label="D +encoder (base768·s-dec·9k)"),
        "E": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="small", pool="small", label="E +encoder (large1024·s-dec·9k)"),
        "F": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="big", pool="big", label="F ALL (large1024·BIG·24k)"),
        "G": dict(enc="facebook/dinov2-small", dim=384, dec="big", pool="big", label="G +decoder+data (s384·BIG·24k)"),
        "H": dict(enc="facebook/dinov2-small", dim=384, dec="xl", pool="small", label="H +decoderXL (s384·XL·9k)"),
        "I": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="xl", pool="big", label="I MAX (large1024·XL·24k)"),
        "J": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="small", pool="big", label="J sweet (large1024·s-dec·24k)"),
        "K": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-base", dim=768, dec="small", pool="big", label="K sweet (base768·s-dec·24k)"),
        # 固定最佳编码器 large + 9k, 扫解码器阶梯 tiny→small(=E)→medium→big→xl
        "L": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="tiny", pool="small", label="L (large·TINY-dec·9k)"),
        "M": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="medium", pool="small", label="M (large·MED-dec·9k)"),
        "N": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="big", pool="small", label="N (large·BIG-dec·9k)"),
        "O": dict(enc="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, dec="xl", pool="small", label="O (large·XL-dec·9k)"),
    }
    keys = [k.strip() for k in a.configs.split(",") if k.strip() in REG]

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
    print(f"mining frames {len(G)} → KMeans {K}  ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(1); big_pool = rng.choice(len(G), min(a.big_pool, len(G)), replace=False)
    small_pool = big_pool[:a.small_pool]
    need_big = any(REG[k]["pool"] == "big" for k in keys)
    dec_pool = big_pool if need_big else small_pool
    print(f"配置 {keys} | 需解码 {'big' if need_big else 'small'} pool = {len(dec_pool)} 帧", flush=True)

    print(f"并行解码 {len(dec_pool)} 帧(56核)...", flush=True)
    imgs224, valid = decode_images(dec_pool, E, FR, t0)
    imgs128 = np.stack([cv2.resize(imgs224[i], (RES, RES), interpolation=cv2.INTER_AREA) for i in range(len(imgs224))]).astype(np.uint8)
    print(f"  解码完成 valid={int(valid.sum())}/{len(dec_pool)}  ({time.time()-t0:.0f}s)", flush=True)

    # 按需 encode(缓存 per encoder)
    enc_cache = {}
    def get_grids(enc, dim, pool):
        npool = a.small_pool if pool == "small" else len(dec_pool)
        key = (enc, npool)
        if key not in enc_cache:
            print(f"GPU encode {enc} on {npool} ...", flush=True)
            enc_cache[key] = encode_grids(imgs224[:npool], valid[:npool], enc, dim)
        return enc_cache[key], imgs128[:npool], dec_pool[:npool]

    sel = [c for c in range(K) if cov[c] >= np.quantile(cov, 0.6)]
    sel = sorted(sel, key=lambda c: tpos[c]); NS = min(10, len(sel)); sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]

    def centroid_medoid(dec, grids, owners):
        cents, meds = [], []; olab = lab[owners]
        for c in sel:
            mem = np.where(olab == c)[0]
            if len(mem):
                cents.append(dec(grids[mem].astype(np.float32).mean(0)[None])[0])
                md = mem[np.argmin(np.linalg.norm(G[owners[mem]] - cen[c], axis=1))]
                meds.append(dec(grids[md][None].astype(np.float32))[0])
            else:
                cents.append(np.zeros((RES, RES, 3), np.uint8)); meds.append(np.zeros((RES, RES, 3), np.uint8))
        return cents, meds

    metrics = {}; cent_rows = {}; med_rows = {}
    for k in keys:
        cfg = REG[k]; name = cfg["label"]
        grids, imgs, owners = get_grids(cfg["enc"], cfg["dim"], cfg["pool"])
        print(f"训练 [{name}] ...", flush=True)
        decf = train_dec(grids, imgs, cfg["dim"], cfg["dec"], a.epochs)
        rec = decf(grids[:16].astype(np.float32)); rl1 = float(np.mean(np.abs(rec.astype(float) - imgs[:16]))) / 255
        rsh = float(np.mean([sharp(x) for x in rec]))
        cents, meds = centroid_medoid(decf, grids, owners)
        cent_rows[name] = cents; med_rows[name] = meds
        metrics[name] = {"recon_L1": round(rl1, 4), "recon_sharp": round(rsh, 1),
                         "centroid_sharp": round(float(np.mean([sharp(x) for x in cents])), 1),
                         "medoid_sharp": round(float(np.mean([sharp(x) for x in meds])), 1)}
        # 存 per-config 工件供并行聚合
        np.savez(OUTJ / f"scale_cfg_{k}.npz", cents=np.array(cents), meds=np.array(meds),
                 label=name, metrics=json.dumps(metrics[name]), tpos_sel=tpos[sel])
        print(f"  {name}: {metrics[name]}  ({time.time()-t0:.0f}s)", flush=True)

    # nearest 真实帧(参考)
    nearest = []
    for c in sel:
        gi = np.where(lab == c)[0]; nn_i = gi[np.argmin(np.linalg.norm(G[gi] - cen[c], axis=1))]
        fm = grab_ep(int(E[nn_i]), [FR[nn_i]]); im = fm.get(int(FR[nn_i]))
        nearest.append(cv2.resize(im, (RES, RES)) if im is not None else np.zeros((RES, RES, 3), np.uint8))
    metrics["nearest_real"] = {"sharp": round(float(np.mean([sharp(x) for x in nearest])), 1)}
    np.savez(OUTJ / "scale_nearest.npz", nearest=np.array(nearest), tpos_sel=tpos[sel])
    json.dump(metrics, open(OUTJ / f"scale_ablation_metrics_{a.tag}.json", "w"), indent=2, ensure_ascii=False)
    print("METRICS", json.dumps(metrics, ensure_ascii=False, indent=1), flush=True)

    # 图: 簇中心(grid 平均) across configs + nearest
    names = [REG[k]["label"] for k in keys]
    rows = names + ["nearest real (ref)"]
    fig, axes = plt.subplots(len(rows), NS, figsize=(1.5 * NS, 1.6 * len(rows)))
    for r, name in enumerate(rows):
        imgs_r = cent_rows[name] if name in cent_rows else nearest
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(imgs_r[j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        lab_txt = name + (f"\ncentroid sharp={metrics[name]['centroid_sharp']}" if name in cent_rows else f"\nsharp={metrics['nearest_real']['sharp']}")
        axes[r, 0].set_ylabel(lab_txt, fontsize=7.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle("Cluster CENTROID (grid-average) across scale levers — does more data / bigger decoder / bigger encoder sharpen the centroid?", fontsize=12)
    fig.tight_layout(); fig.savefig(OUTV / f"crave_scale_ablation_{a.tag}.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED crave_scale_ablation_{a.tag}.png  total", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
