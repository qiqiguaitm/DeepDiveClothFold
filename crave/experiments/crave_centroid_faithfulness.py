"""为"簇中心代表图"选最合适 编码器/解码器:用**感知保真度**(而非 recon 锐度)给 9 配置打分。

动机:recon 锐度=Laplacian 方差,奖励高频纹理/噪声;XL/BIG 解码器对"平均 grid"会编出高频噪声→
图很"锐"但可读性差、信息错乱。真正想要的是**语义保真 + 平滑可读**。
指标:把每个配置解码出的簇中心图 与 该簇最近真实帧,各过 frozen DINOv2-small → 余弦相似度(越高=越像真实里程碑=越可读);
      再给一个"高频噪声"参考(Laplacian 方差,越高越可能是噪声而非信息)。
读 temp/crave_a1a2/scale_cfg_{A..I}.npz + scale_nearest.npz。

Thin entrypoint over `crave` for paths only. The scoring encoder is kept inlined as
fp32 `facebook/dinov2-small` (NOT crave's `dinov2-small` spec, which is fp16 + the local
HF mirror) so the cosine scores stay byte-for-byte identical — see TODO below.

Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_centroid_faithfulness.py
"""
import json
import os

import cv2
import numpy as np
import torch

from crave.config import REPO

OUTJ = REPO / "temp/crave_a1a2"
SPEC = {"A": "small384·s-dec·9k", "B": "small384·s-dec·24k", "C": "small384·BIG·9k", "D": "base768·s-dec·9k",
        "E": "large1024·s-dec·9k", "F": "large1024·BIG·24k", "G": "small384·BIG·24k", "H": "small384·XL·9k", "I": "large1024·XL·24k"}
dev = "cuda"


def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # TODO(crave-lib): scoring uses fp32 `facebook/dinov2-small` (HF hub-id, default 224
    # processor). crave.encoders.load_encoder("dinov2-small") would load the fp16 local
    # mirror instead — a non-identical encoder that would shift the cosine scores, so it
    # stays inlined here to preserve behavior exactly. A fp32 "dinov2-small-score" spec
    # (or load_encoder dtype override) should live in crave.config.encoders.
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

    def feat(imgs):  # imgs list of HWC uint8 → pooled L2-norm DINOv2-small feature
        out = []
        for b in range(0, len(imgs), 64):
            with torch.no_grad():
                px = proc(images=[imgs[i] for i in range(b, min(b + 64, len(imgs)))], return_tensors="pt").to(dev)
                t = enc(**px).last_hidden_state[:, 1:].mean(1)
                out.append(torch.nn.functional.normalize(t, dim=-1).cpu().numpy())
        return np.concatenate(out)

    near = np.load(OUTJ / "scale_nearest.npz", allow_pickle=True)["nearest"]
    fn = feat([near[j] for j in range(len(near))])

    rows = []
    for k, name in SPEC.items():
        f = OUTJ / f"scale_cfg_{k}.npz"
        if not f.exists(): continue
        cents = np.load(f, allow_pickle=True)["cents"]
        fc = feat([cents[j] for j in range(len(cents))])
        cos = float(np.mean(np.sum(fc * fn, axis=1)))          # 感知保真:中心图 vs 真实里程碑(↑好)
        noise = float(np.mean([sharp(cents[j]) for j in range(len(cents))]))  # 高频量(过高=噪声)
        rows.append((k, name, round(cos, 3), round(noise, 1)))

    # 真实帧自身的高频量(参考基准)
    near_noise = float(np.mean([sharp(near[j]) for j in range(len(near))]))
    rows.sort(key=lambda r: -r[2])
    print(f"{'cfg':<4}{'spec':<22}{'faithfulness(cos↑)':<20}{'noise(LapVar)':<14}")
    for k, name, cos, noise in rows:
        print(f"{k:<4}{name:<22}{cos:<20}{noise:<14}")
    print(f"{'--':<4}{'NEAREST real (ref)':<22}{'1.000':<20}{round(near_noise,1):<14}")
    json.dump({"ranking_by_faithfulness": [{"cfg": k, "spec": n, "faithfulness_cos": c, "noise_lapvar": x} for k, n, c, x in rows],
               "nearest_noise_lapvar": round(near_noise, 1)},
              open(OUTJ / "centroid_faithfulness.json", "w"), indent=2, ensure_ascii=False)
    print("\nsaved centroid_faithfulness.json")


if __name__ == "__main__":
    main()
