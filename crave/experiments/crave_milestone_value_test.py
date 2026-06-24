"""在几条 episode 上测试 milestone value 改造效果 + 输出 milestone 顺序信息。
对比逐帧读出: 旧 value(Pk=首达时间中位序) vs 新 value(precedence 定序 + isotonic 度量)。
milestone centers 相同, 只改其 value 位置(bin), Viterbi-DP 读出 → smooth_monotone。
复用全量 dino shard 特征(测试帧直接取, 无需重编码)。

Thin entrypoint over `crave`: otsu/med/viterbi/smooth_monotone from crave.utils, REPO
from crave.config, Agg+SimHei via crave.render.setup_mpl. The dino shard cache
(temp/crave_full), the kai0_base video grabber (grab_ep/camp/crop224) and the Wan VAE
render path stay inlined — see TODOs below.

跑法: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_milestone_value_test.py
"""
import glob
import json
import time

import av
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan

from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import med, otsu, smooth_monotone, viterbi

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; dev = "cuda"; ENC = "dino"

# TODO(crave-lib): the kai0_base dataset (DS/cs) + grab_ep video grabber (camp/crop224)
# duplicate crave_decoder_scale_ablation; a kai0 raw-frame grabber should live in
# crave.data, and the temp/crave_full dino shard cache + temp/_triple style caches in
# crave.config / crave.data.
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


def readout(Fq, C, Pval, startK=None, endK=None, NB=41):
    bins = np.linspace(0, 1, NB); n = len(Fq)
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    cb = [int(np.argmin(abs(bins - Pval[m]))) for m in range(len(Pval))]
    em = np.full((n, NB), 1e3)
    for m in range(len(Pval)): em[:, cb[m]] = np.minimum(em[:, cb[m]], d[:, m])
    if startK is not None:                                  # crave.value 的 start/end 锚点(到 0/1)
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1)
        de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(n) / n
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    v = med(viterbi(em, bins, lam=8.0, end_bonus=2.0)[0], 5)  # crave.utils.viterbi → (values, path)
    return smooth_monotone(v, fps=3.0)


def main():
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; Fn = (feat.astype(np.float32)); Fn /= (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
    F = Fn[vi]; Ev, Tv = E[vi], T[vi]
    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); cen = km.cluster_centers_; lab = km.predict(F)
    ne = len(set(E.tolist()))
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); cl = np.array(sel)

    ep_list = sorted(set(Ev.tolist())); fe = np.full((len(ep_list), M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == cl[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            both = ~np.isnan(fe[:, i]) & ~np.isnan(fe[:, j])
            if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.array([np.nansum(Pbef[i, :]) for i in range(M)]); prec_order = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec_order])
    iso_val = np.empty(M); iso_val[np.array(prec_order)] = iso
    C = cen[cl]
    # start/end 锚点(复刻 crave.value): 各 ep 首2帧/末2帧聚 8 簇
    SP, EP = [], []
    for e in ep_list:
        fi = np.where(Ev == e)[0]; oo = np.argsort(Tv[fi]); fi = fi[oo]
        if len(fi) >= 2: SP.append(F[fi[:2]]); EP.append(F[fi[-2:]])
    from sklearn.cluster import KMeans
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_

    # ---- milestone 顺序信息(precedence 序) ----
    print("=== MILESTONE ORDER (precedence) ===", flush=True)
    order_info = []
    for r, m in enumerate(prec_order):
        order_info.append({"rank": r, "cluster": int(cl[m]), "old_Pk": round(float(Pk[m]), 3),
                           "new_iso_value": round(float(iso_val[m]), 3), "coverage": round(float(cov[cl[m]]), 3)})
    for o in order_info: print(f"  r{o['rank']:2d} | old_Pk={o['old_Pk']:.3f} new={o['new_iso_value']:.3f} cov={o['coverage']:.2f} clu{o['cluster']}", flush=True)
    json.dump(order_info, open(OUTD / "milestone_order_info.json", "w"), indent=2)

    # ---- 选 6 条测试 ep(覆盖不同长度) ----
    lens = {e: int((Ev == e).sum()) for e in ep_list}
    cand_e = [e for e in ep_list if lens[e] >= 30]
    pick = [2302] + [e for e in [cand_e[0], cand_e[len(cand_e) // 4], cand_e[len(cand_e) // 2], cand_e[3 * len(cand_e) // 4], cand_e[-1]] if e != 2302]
    pick = pick[:6]
    print(f"测试 ep: {pick}", flush=True)
    curves = []
    for e in pick:
        fi = np.where(Ev == e)[0]; ordr = np.argsort(Tv[fi]); fi = fi[ordr]
        Fq = F[fi]; tn = Tv[fi]
        v_old = readout(Fq, C, Pk, startK, endK); v_new = readout(Fq, C, iso_val, startK, endK)
        curves.append((e, tn, v_old, v_new))

    # ---- Wan 渲 precedence 序 medoid 条 ----
    print("加载 Wan VAE 渲 milestone 顺序条 ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    def wan_dec(z):
        with torch.no_grad(): o = vae.decode(torch.from_numpy(z[None, :, None]).to(dev)).sample
        return np.clip((o[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    meds = []
    for m in prec_order:
        loc = np.where(lab == cl[m])[0]; d = np.linalg.norm(F[loc] - cen[cl[m]], axis=1); gi = vi[loc[int(np.argmin(d))]]
        fm = grab_ep(int(E[gi]), [int(FR[gi])])
        if int(FR[gi]) not in fm: meds.append(np.zeros((256, 256, 3), np.uint8)); continue
        img = cv2.resize(fm[int(FR[gi])], (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None].to(dev)
        with torch.no_grad():
            ee = vae.encode(x); zz = (ee.latent_dist.mode() if hasattr(ee, "latent_dist") else ee.latent)[0, :, 0].cpu().numpy()
        meds.append(wan_dec(zz))

    plt = setup_mpl()
    # 图1: 6 条 ep value 曲线(old vs new)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for k, (e, tn, vo, vn) in enumerate(curves):
        ax = axes[k // 3, k % 3]
        ax.plot(tn, vo, color="#999", lw=2, label="old (Pk time-order)")
        ax.plot(tn, vn, color="#1a7f37", lw=2, label="new (precedence+isotonic)")
        moo = float(np.mean(np.diff(vo) >= -1e-6)); mon = float(np.mean(np.diff(vn) >= -1e-6))
        ax.set_title(f"ep{e}  ({len(tn)}fr)  mono old={moo:.2f} new={mon:.2f}", fontsize=9)
        ax.set_xlabel("progress (norm time)"); ax.set_ylabel("value"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
        if k == 0: ax.legend(fontsize=8)
    fig.suptitle("Per-episode value readout: OLD (Pk time-order) vs NEW (precedence+isotonic) — same milestone centers, different value positions", fontsize=12)
    fig.tight_layout(); o1 = OUTV / "crave_milestone_value_test.png"; fig.savefig(o1, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {o1.name}", flush=True)

    # 图2: milestone 顺序条(precedence 序, 标 new value)
    nc = M; fg, ax2 = plt.subplots(1, nc, figsize=(0.85 * nc, 1.7))
    for j in range(nc):
        a = ax2[j]; a.imshow(meds[j]); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"r{j}\nv={iso_val[prec_order[j]]:.2f}", fontsize=6)
    fg.suptitle(f"MILESTONE ORDER (precedence) — {M} milestones, value = isotonic metric (early→late)", fontsize=11)
    fg.tight_layout(); o2 = OUTV / "crave_milestone_order_strip.png"; fg.savefig(o2, dpi=125, bbox_inches="tight"); plt.close(fg)
    print(f"SAVED {o2.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
