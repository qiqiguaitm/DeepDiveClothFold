"""Benchmark Cosmos3OmniPipeline I2V generation time vs num_frames.
Usage: CUDA_VISIBLE_DEVICES=0 python bench_time.py <MODEL_DIR> [single|balanced]
Prints load time + per-clip wall time for num_frames in {24,72,189} (1s/3s/7s @24fps).
"""
import os, sys, time, json
import torch

# P0 shim: transformers 5.x references torch.float8_e8m0fnu (torch>=2.7) at import; torch 2.6 here.
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

from PIL import Image
import numpy as np
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

MODEL = sys.argv[1]
PLACEMENT = sys.argv[2] if len(sys.argv) > 2 else "single"
dev_map = "balanced" if PLACEMENT == "balanced" else "cuda"

CAP = {"temporal_caption": "A top-down overhead view of a dual-arm Agilex Piper robot flattening and folding a piece of cloth on a light mat; the grippers pinch opposite edges, pull to flatten, then fold the cloth in half and press the crease flat.",
       "fps": 24.0, "resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}
NEG = "low quality, static, no motion, blurry, distorted, flickering, artifacts."

print(f"[bench] MODEL={MODEL.split('/')[-1]} placement={PLACEMENT} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
t0 = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=dev_map,
                                           enable_safety_checker=False)  # skip gated guardrail download
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
load_s = time.time() - t0
print(f"[bench] load_time = {load_s:.1f}s", flush=True)

img = Image.fromarray((np.random.rand(480, 832, 3) * 255).astype("uint8"))

def gen(nf, dur, seed=0):
    cap = {**CAP, "duration": dur}
    t = time.time()
    _ = pipe(prompt=json.dumps(cap), negative_prompt=NEG, image=img,
             num_frames=nf, height=480, width=832, fps=24,
             num_inference_steps=50, guidance_scale=6.0, enable_safety_check=False,
             add_resolution_template=False, add_duration_template=False,
             generator=torch.Generator("cuda").manual_seed(seed))
    return time.time() - t

# warmup (one-time CUDA autotune/graph cost) on the smallest, then time each twice -> report 2nd
print("[bench] warmup 24f ...", flush=True); _ = gen(24, "1s")
for nf, dur in [(24, "1s"), (72, "3s"), (189, "7s")]:
    t1 = gen(nf, dur, seed=1)
    print(f"[bench] num_frames={nf:3d} ({dur}) -> {t1:6.1f}s", flush=True)
print("[bench] DONE", flush=True)
