"""milestone value 前后顺序诊断 + 修复:当前按"时间分位(tpos/首达中位)"排序 → 用跨 ep "成对先后(precedence)"重排。
复用全量 dino shard 特征(无需重编码)。产出:
  ① 当前 tpos 序的逆序对数 + Kendall-τ(tpos序 vs precedence序)
  ② 簇中心 medoid 画廊两行: 上=当前 tpos 序, 下=precedence 序; 逆序 milestone 红框标出
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_milestone_order.py
"""
from __future__ import annotations

import glob
import json
import time

import av
import cv2
import numpy as np
import torch

from crave.config import REPO
from crave.utils import otsu

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; dev = "cuda"; ENC = "dino"

# TODO(crave-lib): kai0_base raw-video frame grabber (DS/cs/camp/crop224/grab_ep) should move into crave.data
#                  (a "kai0_base" DatasetConfig + a frame-grab loader). Re-inlined here verbatim.
DS = REPO / "kai0/data/Task_A/kai0_base"
cs = __import__("json").load(open(DS / "meta/info.json"))["chunks_size"]


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


def main():
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; F = feat[vi].astype(np.float32); F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    Ev, Tv = E[vi], T[vi]
    print(f"{len(vi)} 有效帧; 聚类(自适应 K0)...", flush=True)
    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); cen = km.cluster_centers_
    lab = km.predict(F)
    ne = len(set(E.tolist()))
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); vt = tstd[tstd < 9]; tau_pur = float(np.percentile(vt, 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    gap = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= gap: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); print(f"K0={K0} → {M} milestones", flush=True)

    # ---- 每 ep 对每 milestone 的首达归一化时间 ----
    ep_list = sorted(set(Ev.tolist())); fe = np.full((len(ep_list), M), np.nan)
    cl = np.array(sel)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == cl[m]]
            if len(hit): fe[ei, m] = hit.min()
    # ---- 成对先后 precedence ----
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            both = ~np.isnan(fe[:, i]) & ~np.isnan(fe[:, j])
            if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.array([np.nansum(Pbef[i, :]) for i in range(M)])          # 软胜场: 越大=越早
    prec_order = list(np.argsort(-soft))                                # precedence 序(早→晚)
    tpos_order = list(np.argsort([tpos[sel[m]] for m in range(M)]))     # 当前 tpos 序(早→晚)

    # ---- 逆序对 + Kendall-τ ----
    def inversions(order):
        inv = 0; pairs = []
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i, j = order[a], order[b]                                # 序里 i 在 j 前
                if not np.isnan(Pbef[i, j]) and Pbef[i, j] < 0.5:        # 但多数 ep 里 j 在 i 前
                    inv += 1; pairs.append((i, j, float(Pbef[i, j])))
        return inv, pairs
    inv_tpos, bad = inversions(tpos_order); inv_prec, _ = inversions(prec_order)
    # Kendall-τ between the two orders
    rk_t = {m: r for r, m in enumerate(tpos_order)}; rk_p = {m: r for r, m in enumerate(prec_order)}
    conc = disc = 0
    for a in range(M):
        for b in range(a + 1, M):
            s1 = np.sign(rk_t[a] - rk_t[b]); s2 = np.sign(rk_p[a] - rk_p[b])
            if s1 == s2: conc += 1
            else: disc += 1
    ktau = (conc - disc) / (conc + disc)
    moved = [m for m in range(M) if abs(rk_t[m] - rk_p[m]) >= 3]
    summ = {"M": M, "inversions_tpos_order": inv_tpos, "inversions_precedence_order": inv_prec,
            "kendall_tau_tpos_vs_prec": round(ktau, 3), "milestones_moved_ge3": len(moved)}
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump(summ, open(OUTD / "milestone_order_summary.json", "w"), indent=2)

    # ---- medoid 渲染(Wan)----
    # TODO(crave-lib): Wan2.2-VAE decode (vae.decode latent → RGB) is not exposed by the Encoder
    #                  interface (only encode_pooled/encode_grid); a crave.render.wan_decode or a
    #                  WanVAEEncoder.decode would remove this inlined VAE load + wan_dec.
    print("加载 Wan VAE 渲 medoid ...", flush=True)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    def wan_dec(z):
        with torch.no_grad(): o = vae.decode(torch.from_numpy(z[None, :, None]).to(dev)).sample
        return np.clip((o[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    med = {}
    for m in range(M):
        loc = np.where(lab == cl[m])[0]; d = np.linalg.norm(F[loc] - cen[cl[m]], axis=1); gi = vi[loc[int(np.argmin(d))]]
        fm = grab_ep(int(E[gi]), [int(FR[gi])])
        if int(FR[gi]) not in fm: med[m] = np.zeros((256, 256, 3), np.uint8); continue
        img = cv2.resize(fm[int(FR[gi])], (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None].to(dev)
        with torch.no_grad():
            e = vae.encode(x); zz = (e.latent_dist.mode() if hasattr(e, "latent_dist") else e.latent)[0, :, 0].cpu().numpy()
        med[m] = wan_dec(zz)
    print(f"渲染完成 ({time.time()-t0:.0f}s); 出图 ...", flush=True)

    from crave.render import setup_mpl
    plt = setup_mpl()
    badset = set([p[0] for p in bad] + [p[1] for p in bad])
    rows = [("CURRENT order (by tpos / mean-time)  — red = precedence inversion", tpos_order, True),
            ("FIXED order (by cross-ep precedence, value = ordinal rank)", prec_order, False)]
    ncol = M; fig, axes = plt.subplots(2, ncol, figsize=(0.95 * ncol, 3.2))
    for r, (title, order, markbad) in enumerate(rows):
        for j in range(ncol):
            ax = axes[r, j]; m = order[j]; ax.imshow(med[m]); ax.set_xticks([]); ax.set_yticks([])
            lab_v = (f"v={tpos[sel[m]]:.2f}" if r == 0 else f"r={j}/{M-1}")
            ax.set_title(f"m{m}\n{lab_v}", fontsize=5)
            for sp in ax.spines.values():
                sp.set_color("red" if (markbad and m in badset) else "0.8"); sp.set_linewidth(2 if (markbad and m in badset) else 0.5)
        axes[r, 0].set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
    fig.suptitle(f"Milestone VALUE ordering: CURRENT(tpos) inversions={inv_tpos} vs PRECEDENCE inversions={inv_prec}  |  Kendall-τ={ktau:.2f}  |  {len(moved)} milestones moved ≥3", fontsize=11)
    fig.tight_layout(); out = OUTV / "crave_milestone_order_fix.png"; fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
