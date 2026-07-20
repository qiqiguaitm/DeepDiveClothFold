#!/usr/bin/env python
"""抽严格对齐的 kai0 特征(修 dino_sub20 错对齐问题): 直接解 kai0_base top_head 视频,
每 20 帧取 1(视频帧数==parquet 行数已核对), 双编码器(DINOv3-base pooled / SigLIP-2 base)。
输出: <OUT>/{dino,siglip2}/ep<id>.npz  keys: pooled[n,768], fidx[n](对应 parquet 行号)
用法: srpo python kai0_aligned_extract.py dino|siglip2 [N_EP]
"""
import os, sys, glob
import numpy as np

ENC = sys.argv[1]; N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 110
VID = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base/videos/chunk-000/observation.images.top_head"
GT  = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage/data/chunk-000"
OUT = f"/vePFS/tim/tmp/claude-1000/-vePFS-tim-workspace-deepdive-kai0/52e86d52-cd8c-4dfd-9952-1594aae894a2/scratchpad/kai0_aligned/{ENC}"
STRIDE = 20
os.makedirs(OUT, exist_ok=True)

# 选 ep: 从 0 号起, 有视频+有 GT parquet 的前 N_EP 条
eps = []
for e in range(4000):
    if len(eps) >= N_EP: break
    if os.path.exists(f"{VID}/episode_{e:06d}.mp4") and os.path.exists(f"{GT}/episode_{e:06d}.parquet"):
        eps.append(e)
print(f"[sel] {len(eps)} eps: {eps[0]}..{eps[-1]}", flush=True)

def decode_stride(mp4, stride):
    import av
    fr_out, fidx = [], []
    with av.open(mp4) as cont:
        k = 0
        for fr in cont.decode(video=0):
            if k % stride == 0:
                fr_out.append(fr.to_ndarray(format="rgb24")); fidx.append(k)
            k += 1
    return fr_out, np.array(fidx, np.int32)

if ENC == "dino":
    sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/lmvla/crave/src")
    from crave.encoders import load_encoder
    enc = load_encoder("dinov3-base", dtype="bf16")
    embed = lambda imgs: enc.encode_pooled(imgs)
elif ENC == "siglip2":
    import torch
    from transformers import AutoProcessor, AutoModel
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    mdl = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.bfloat16).cuda().eval()
    def embed(imgs, bs=128):
        outs = []
        with torch.no_grad():
            for i in range(0, len(imgs), bs):
                px = proc(images=imgs[i:i+bs], return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
                outs.append(mdl.get_image_features(pixel_values=px).float().cpu().numpy())
        return np.concatenate(outs)
else:
    raise SystemExit(f"unknown encoder {ENC}")

for n, e in enumerate(eps):
    dst = f"{OUT}/ep{e}.npz"
    if os.path.exists(dst): continue
    frames, fidx = decode_stride(f"{VID}/episode_{e:06d}.mp4", STRIDE)
    np.savez_compressed(dst, pooled=embed(frames).astype(np.float32), fidx=fidx)
    if n % 10 == 0: print(f"[{ENC}] {n+1}/{len(eps)} ep{e} n={len(fidx)}", flush=True)
print(f"[{ENC}] ALL DONE {len(eps)} eps -> {OUT}", flush=True)
