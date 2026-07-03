#!/usr/bin/env python
"""统一解码器基准:同一批 held-out 帧、同一套指标,一次排定 CRAVE 解码方案。

候选:① 检索(最近真实帧) ② patch-grid 空间解码 ③ pooled-L1 ④ pooled-GAN。
指标(全在同一批帧、同一 DINOv3-H 上):
  - L1(解码图 vs 真实帧,/255)—— 像素保真;
  - 锐度(Laplacian variance)—— 越高越清晰;
  - 再编码 cos(解码图 → DINOv3-H pooled → 与目标 latent 的 cos)—— 语义保真(揭穿"锐但幻觉")。
自重建设定:每个解码器都吃"该帧自己的 latent",隔离预测误差,纯比解码器。

Run: REPO=/home/tim/workspace/deepdive_kai0 PYTHONPATH=crave/src:lmwm/src:lmwm/scripts \
  /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/decoder_benchmark.py
输出: crave/docs/visualization/decoder_benchmark/{decoder_benchmark.png,summary.json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from crave.decoding.decoder import make_decoder
from crave.encoders import load_encoder
from crave.render import setup_mpl
from lmwm.retrieval_decoder import LatentRetrievalDecoder
from train_dinov3h_decoder import PooledDecoder

plt = setup_mpl()
REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FEAT = REPO / "temp/crave_full_dinov3h"
DATA = REPO / "kai0/data/Task_A/kai0_base"
CKPT = REPO / "lmwm/checkpoints"
OUT = REPO / "crave/docs/visualization/decoder_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
N_Q = 120
RES = 128
rng = np.random.RandomState(0)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def sharp(img):  # Laplacian variance on gray
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def to_uint8(t):  # (B,3,H,W) in [-1,1] -> (B,H,W,3) uint8
    o = t.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip((o + 1) * 127.5, 0, 255).astype(np.uint8)


def main():
    enc = load_encoder("dinov3-h", device=DEV)
    rdec = LatentRetrievalDecoder(feature_dir=FEAT, dataset_root=DATA, device=DEV, res=RES)
    N = len(rdec.feat)
    q = rng.choice(N, N_Q, replace=False)
    q_pool = rdec.feat[q].astype(np.float32)            # (Q,1280) L2, cache pooled = target latent
    q_ep = rdec.E[q]
    print(f"queries={N_Q} from {N} cached frames", flush=True)

    real = np.stack([cv2.resize(rdec.frame(int(g)), (RES, RES)) for g in q])  # (Q,128,128,3) RGB uint8

    # ---- ① retrieval: nearest cached frame from a DIFFERENT episode ----
    ret_g = np.array([int(rdec.retrieve(q_pool[i:i + 1], topk=1, exclude_episode=int(q_ep[i]))[0][0, 0])
                      for i in range(N_Q)])
    ret_img = np.stack([cv2.resize(rdec.frame(int(g)), (RES, RES)) for g in ret_g])

    # ---- ③④ pooled decoders ----
    def run_pooled(ckpt):
        d = torch.load(ckpt, map_location=DEV, weights_only=False)
        G = PooledDecoder(din=d["din"], res=d["res"]).to(DEV).eval()
        G.load_state_dict(d["model"])
        with torch.no_grad():
            x = torch.from_numpy(l2(q_pool)).to(DEV)
            return to_uint8(G(x))
    pl1 = run_pooled(CKPT / "dinov3h_decoder/dec.pt")
    pgan = run_pooled(CKPT / "dinov3h_decoder/dec_gan.pt")

    # ---- ② patch-grid decoder (needs per-frame grid) ----
    grids = enc.encode_grid(list(real)).astype(np.float32)  # (Q,1280,16,16)
    dpk = torch.load(CKPT / "patch_decoder/patch_dec.pt", map_location=DEV, weights_only=False)
    Dp = make_decoder(dpk["din"], dpk["dec"]).to(DEV).eval()
    Dp.load_state_dict(dpk["model"])
    muT = torch.from_numpy(dpk["mu"]).view(1, -1, 1, 1).float().to(DEV)
    sdT = torch.from_numpy(dpk["sd"]).view(1, -1, 1, 1).float().to(DEV)
    with torch.no_grad():
        xg = (torch.from_numpy(grids).to(DEV) - muT) / sdT
        patch = to_uint8(Dp(xg))

    cands = {"检索(最近真实帧)": ret_img, "patch-grid 空间解码": patch,
             "pooled-L1": pl1, "pooled-GAN": pgan}

    # ---- metrics: L1 / sharpness / re-encode cos ----
    rows = {}
    for name, imgs in cands.items():
        l1 = np.mean(np.abs(imgs.astype(np.float32) - real.astype(np.float32))) / 255
        shp = float(np.mean([sharp(x) for x in imgs]))
        re_pool = l2(enc.encode_pooled(list(imgs)).astype(np.float32))
        rc = float(np.mean(np.sum(re_pool * q_pool, axis=1)))  # cos with target latent
        rows[name] = {"L1": round(float(l1), 4), "sharpness": round(shp, 1),
                      "reencode_cos": round(rc, 4)}
        print(f"{name:22s} L1={l1:.3f} sharp={shp:6.1f} reencode_cos={rc:.3f}", flush=True)
    real_sharp = float(np.mean([sharp(x) for x in real]))
    rows["_real_frames"] = {"sharpness": round(real_sharp, 1)}

    (OUT / "summary.json").write_text(json.dumps({"n_queries": N_Q, "results": rows}, ensure_ascii=False, indent=2))

    # ---- figure: examples (left) + metric bars (right) ----
    ex = [0, 1, 2, 3]
    order = ["real"] + list(cands.keys())
    names = list(cands.keys())
    colors = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(len(order), len(ex) + 4, wspace=0.1, hspace=0.12)
    labels = {"real": "真实帧"}
    for r, key in enumerate(order):
        imgs = real if key == "real" else cands[key]
        for c, e in enumerate(ex):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(imgs[e]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(labels.get(key, key), fontsize=12, rotation=0, ha="right", va="center")
    metrics = [("reencode_cos", "↑ 语义保真(核心)"), ("L1", "↓ 像素误差"), ("sharpness", "↑ 锐度")]
    span = len(order) // len(metrics) + 1
    for mi, (metric, better) in enumerate(metrics):
        axm = fig.add_subplot(gs[mi * span:min((mi + 1) * span, len(order)), len(ex) + 1:])
        vals = [rows[n][metric] for n in names]
        axm.barh(range(len(names)), vals, color=colors)
        axm.set_yticks(range(len(names))); axm.set_yticklabels(names, fontsize=10)
        axm.invert_yaxis()
        axm.margins(x=0.16)
        for i, v in enumerate(vals):
            axm.text(v, i, f" {v}", va="center", fontsize=10, fontweight="bold")
        if metric == "sharpness":
            axm.axvline(real_sharp, color="#333", ls="--", lw=1.2)
            axm.text(real_sharp, len(names) - 0.4, f" 真实={real_sharp:.0f}", fontsize=8.5, ha="right", color="#333")
        axm.set_title(f"{metric}  ({better})", fontsize=11)
        axm.grid(alpha=.25, axis="x")
    fig.suptitle(f"CRAVE 解码器统一基准 · {N_Q} held-out 帧自重建 · DINOv3-H | kai0_base  —— 检索在语义保真+锐度双赢,合成解码器语义保真封顶 ~0.47",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(OUT / "decoder_benchmark.png", dpi=120, bbox_inches="tight")
    print("SAVED", OUT / "decoder_benchmark.png", flush=True)


if __name__ == "__main__":
    main()
