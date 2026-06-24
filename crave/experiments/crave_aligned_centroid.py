"""CRAVE 簇质心代表图 · 方案 B:align-then-average(无 landmark 配准后再平均, 零训练)。

deep research 核心原理:像素平均只有"先对齐"才清晰。patch-grid 平均仍鬼影是因为布料帧没空间配准。
这里做 congealing-lite:每簇取若干成员真实帧 → 用布料掩码的质心+主轴(+尺度)把每帧配准到
canonical 位姿 → 再像素平均。对比:① 对齐后平均(本方案)② 朴素像素平均(不对齐)③ 最近真实帧。
全程零训练 / 纯 cv2,直接回答"轻量配准能否得到清晰合成质心"。

数据 kai0_base(kai-only),相机 top_head。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_aligned_centroid.py [--mine-n 200] [--members 40]
输出: crave/docs/visualization/centroid_decoder/crave_aligned_centroid.png
      temp/crave_a1a2/aligned_centroid_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from crave.config import REPO, resolve_dataset, viz_dir
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import mkp

OUTV = viz_dir("centroid_decoder")
OUTJ = REPO / "temp/crave_a1a2"
RES = 128


def grab_ep(cfg, e, frames30):
    # TODO(crave-lib): kai0.grab_ep crops to 224; this script resizes raw → RES(128) directly.
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


def cloth_mask(img):
    g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    th, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g < th).astype(np.uint8)  # 布料比桌面暗
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def align_canonical(img, ref_area):
    """用布料掩码的质心+主轴(+尺度)把图配准到 canonical:质心居中、主轴竖直、面积归一。"""
    m = cloth_mask(img); ys, xs = np.where(m > 0)
    if len(xs) < 30: return None
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    w, v = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))  # 主轴角
    area = len(xs); scale = np.sqrt(ref_area / max(area, 1)); scale = float(np.clip(scale, 0.6, 1.6))
    M = cv2.getRotationMatrix2D((cx, cy), ang - 90, scale)   # 主轴转到竖直
    M[0, 2] += RES / 2 - cx; M[1, 2] += RES / 2 - cy         # 质心移到中心
    out = cv2.warpAffine(img.astype(np.uint8), M, (RES, RES), borderValue=(235, 235, 235))
    # 180° 翻转歧义:让布料质心相对画面中心的竖直分量一致(重心偏下)
    m2 = cloth_mask(out); ys2, _ = np.where(m2 > 0)
    if len(ys2) and ys2.mean() < RES / 2:
        out = cv2.warpAffine(out, cv2.getRotationMatrix2D((RES / 2, RES / 2), 180, 1), (RES, RES), borderValue=(235, 235, 235))
    return out


def sharp(im):
    return float(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-n", type=int, default=200); ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--members", type=int, default=40)
    a = ap.parse_args(); OUTJ.mkdir(parents=True, exist_ok=True)
    cfg = resolve_dataset("kai0_base")

    rawset = set(int(p.stem[2:]) for p in Path(cfg.raw_cache).glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in Path(cfg.arm_cache).glob("ep*.npz")) if e in rawset)
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    Sall = [kai0.loadep_tcc(cfg, e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    T, E, FR, A, Rr, Sx = [], [], [], [], [], []
    for e in mined:
        aa, rr, st, n = kai0.loadep_tcc(cfg, e); A.append(aa); Rr.append(rr); Sx.append(st)
        T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); FR.append(np.arange(n) * 10)
    A = np.concatenate(A); Rr = np.concatenate(Rr); Sx = np.concatenate(Sx); T = np.concatenate(T); E = np.concatenate(E); FR = np.concatenate(FR)
    G = emb(A, Rr, Sx).astype(np.float32); K = a.k
    km = KMeans(K, n_init=3, random_state=0).fit(G); lab = km.labels_; cen = km.cluster_centers_.astype(np.float32)
    tpos = np.array([T[lab == c].mean() for c in range(K)]); cov = np.array([len(set(E[lab == c].tolist())) / len(mined) for c in range(K)])

    sel = [c for c in range(K) if cov[c] >= np.quantile(cov, 0.6)]
    sel = sorted(sel, key=lambda c: tpos[c]); NS = min(12, len(sel)); sel = [sel[i] for i in np.linspace(0, len(sel) - 1, NS).round().astype(int)]

    rows = {"aligned": [], "naive": [], "nearest": []}
    for c in sel:
        gi = np.where(lab == c)[0]; d = np.linalg.norm(G[gi] - cen[c], axis=1); order = gi[np.argsort(d)]
        mem = order[:a.members]
        # 解码这些成员帧
        need = {}
        for i in mem: need.setdefault(int(E[i]), []).append(int(FR[i]))
        imgs = []
        for e, frs in need.items():
            fm = grab_ep(cfg, e, frs)
            imgs += [fm[f] for f in frs if f in fm]
        if not imgs:
            for k in rows: rows[k].append(np.zeros((RES, RES, 3), np.uint8))
            continue
        imgs = np.array(imgs, np.float32)
        ref_area = np.median([int((cloth_mask(im) > 0).sum()) for im in imgs])
        aligned = [align_canonical(im, ref_area) for im in imgs]; aligned = [x for x in aligned if x is not None]
        rows["aligned"].append(np.mean(aligned, 0).astype(np.uint8) if aligned else np.zeros((RES, RES, 3), np.uint8))
        rows["naive"].append(imgs.mean(0).astype(np.uint8))
        rows["nearest"].append(imgs[0].astype(np.uint8))  # 离中心最近

    sh = {k: round(float(np.mean([sharp(x) for x in rows[k]])), 1) for k in rows}
    metrics = {"members_per_cluster": a.members, "n_selected": NS, "sharpness_laplacianVar": sh,
               "note": "对齐后平均 vs 朴素平均 vs 最近帧; 看 align-then-average 是否把合成质心做清晰"}
    json.dump(metrics, open(OUTJ / "aligned_centroid_metrics.json", "w"), indent=2, ensure_ascii=False)
    print("METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)

    plt = setup_mpl()
    labels = [f"(1) aligned-then-average\n(congealing-lite)  sharp={sh['aligned']}",
              f"(2) naive pixel-mean\n(unaligned)  sharp={sh['naive']}",
              f"(3) nearest real frame\n(current)  sharp={sh['nearest']}"]
    fig, axes = plt.subplots(3, NS, figsize=(1.5 * NS, 5.2))
    for r, key in enumerate(["aligned", "naive", "nearest"]):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rows[key][j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos[sel[j]]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle(f"Align-then-average centroid (mask-based registration, zero-training) vs naive mean vs nearest  "
                 f"(sharpness: aligned {sh['aligned']} / naive {sh['naive']} / nearest {sh['nearest']})", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTV / "crave_aligned_centroid.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_aligned_centroid.png", flush=True)


if __name__ == "__main__":
    main()
