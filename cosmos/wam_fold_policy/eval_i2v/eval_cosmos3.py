"""Cosmos3 I2V world-prediction eval worker (single model, a shard of work units).

Work unit = (episode, camera, horizon, anchor). Teacher-forced: condition on the GT
frame at `anchor`, generate `horizon` seconds, compare to GT[anchor:anchor+src_step].

Loader:
  - Nano (16B): mode=single  -> device_map="cuda" (whole pipe on 1 GPU)
  - Super/Super-I2V (64B): mode=shard -> Cosmos3OmniTransformer device_map="auto"
    (shards 120G transformer across all visible GPUs), other components on cuda:0.

Metrics: PSNR/SSIM/temporal_absdiff_ratio/LPIPS via video_metrics_gpu (ported verbatim
from tau0_wm/finetune/eval_gigaworld_dist.py). Results appended to --out (jsonl).
"""
import os, sys, json, time, argparse
import torch
# P0 shim: transformers 5.x refs torch.float8_e8m0fnu (torch>=2.7) at import; torch 2.6 here.
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import imageio.v3 as iio
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

DATA = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val"
CAM_KEY = {"cam_high": "observation.images.cam_high",
           "cam_left_wrist": "observation.images.cam_left_wrist",
           "cam_right_wrist": "observation.images.cam_right_wrist"}
HORIZON = {"1s": (24, 30), "3s": (72, 90), "7s": (189, 236)}   # name -> (gen_frames@24, src_frames@30)
NEG = ("low quality, static with no motion, blurry, distorted, flickering, jpeg artifacts, "
       "washed out colors, low resolution, choppy, jerky, unnatural transitions.")

# ---- metrics (ported verbatim) ----
def _gauss_win(ws, sigma, device, dtype):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return g[:, None] * g[None, :]

def video_metrics_gpu(pred_thwc, gt_thwc, lpips_fn, device):
    T = min(pred_thwc.shape[0], gt_thwc.shape[0])
    p = pred_thwc[:T].permute(0, 3, 1, 2).contiguous().float()
    g = gt_thwc[:T].permute(0, 3, 1, 2).contiguous().float()
    C = p.shape[1]
    mse = ((p - g) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    out = {"psnr": float((10.0 * torch.log10((255.0 ** 2) / mse)).mean().item())}
    ws = 11; w = _gauss_win(ws, 1.5, device, p.dtype).expand(C, 1, ws, ws); pad = ws // 2
    cv = lambda x: F.conv2d(x, w, padding=pad, groups=C)
    mu1, mu2 = cv(p), cv(g); mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1, s2, s12 = cv(p * p) - mu1s, cv(g * g) - mu2s, cv(p * g) - mu12
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out["ssim"] = float((((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))).mean().item())
    if T >= 2:
        out["temporal_absdiff_ratio"] = float(((p[1:] - p[:-1]).abs().mean() / ((g[1:] - g[:-1]).abs().mean() + 1e-6)).item())
    if lpips_fn is not None:
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(p / 127.5 - 1.0, g / 127.5 - 1.0).mean().item())
    return out

# ---- helpers ----
def center_crop_4x3(frame_hwc):  # np [H,W,C] -> 4:3 (crop width)
    H, W = frame_hwc.shape[:2]
    tw = int(round(H * 4 / 3))
    if W > tw:
        x0 = (W - tw) // 2; return frame_hwc[:, x0:x0 + tw]
    return frame_hwc

def to_eval_grid(frames_thwc_u8, device, size=256):  # np uint8 [T,H,W,C] -> tensor [T,size,size,C] float
    arr = np.stack([center_crop_4x3(f) for f in frames_thwc_u8])
    t = torch.from_numpy(arr).to(device).float().permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).contiguous()

def temporal_resample(frames_thwc, n):  # np [T,...] -> [n,...] by nearest index
    T = frames_thwc.shape[0]
    idx = np.linspace(0, T - 1, n).round().astype(int)
    return frames_thwc[idx]

def to_hw(frames_u8, H, W):  # np uint8 [T,h,w,c] -> [T,H,W,c], center-crop to 4:3 then resize
    arr = np.stack([center_crop_4x3(f) for f in frames_u8])
    t = torch.from_numpy(arr).float().permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()

def build_device_map(model_dir, n_gpu, reserve=4):
    """All non-layer top-level keys (glue modules + bare params/buffers like
    action_modality_embed) on cuda:0 so packing/unpacking stays single-device; only
    `layers.*` split across GPUs (block transitions handled by accelerate hooks).
    Introspects the empty model so it covers base-Super and I2V variants alike."""
    from accelerate import init_empty_weights
    cfg = Cosmos3OmniTransformer.load_config(model_dir, subfolder="transformer")
    with init_empty_weights():
        m = Cosmos3OmniTransformer.from_config(cfg)
    n_layers = len(m.layers)
    top = set()
    for n, _ in m.named_parameters(): top.add(n.split(".")[0])
    for n, _ in m.named_buffers(): top.add(n.split(".")[0])
    dm = {k: 0 for k in top if k != "layers"}
    if n_gpu <= 1:
        for i in range(n_layers): dm[f"layers.{i}"] = 0
        return dm
    on0 = max(1, n_layers // n_gpu - reserve)                 # fewer layers on cuda:0 (carries glue+comps)
    rest = n_layers - on0
    per = rest // (n_gpu - 1); extra = rest % (n_gpu - 1)
    counts = [on0] + [per + (1 if k < extra else 0) for k in range(n_gpu - 1)]
    li = 0
    for g, c in enumerate(counts):
        for _ in range(c):
            dm[f"layers.{li}"] = g; li += 1
    return dm

def load_pipe(model_dir, mode):
    if mode == "shard":
        n_gpu = torch.cuda.device_count()
        dm = build_device_map(model_dir, n_gpu)
        tf = Cosmos3OmniTransformer.from_pretrained(model_dir, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16, device_map=dm)
        pipe = Cosmos3OmniPipeline.from_pretrained(model_dir, transformer=tf, torch_dtype=torch.bfloat16,
                                                   enable_safety_checker=False)
        for name, mod in pipe.components.items():
            if name == "transformer":   # already sharded across GPUs
                continue
            if hasattr(mod, "to"):
                try: mod.to("cuda:0")
                except Exception: pass
    else:
        pipe = Cosmos3OmniPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                   device_map="cuda", enable_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    return pipe

def cap_for(cam, dur):
    base = ("A {view} of a dual-arm Agilex Piper robot — two black articulated arms with parallel-jaw "
            "grippers — flattening and folding a piece of cloth on a light-colored mat. The grippers pinch "
            "opposite edges and pull outward to flatten the fabric smooth, then the two arms coordinate to "
            "fold the cloth in half toward the center, aligning the edges and pressing the crease flat. "
            "Motion is smooth and physically plausible; the fabric deforms naturally and casts soft shadows; "
            "lighting is even, the mat and background stay static, only the arms and cloth move.")
    view = {"cam_high": "top-down overhead view looking straight down",
            "cam_left_wrist": "close first-person wrist-mounted view from the robot's left arm",
            "cam_right_wrist": "close first-person wrist-mounted view from the robot's right arm"}[cam]
    return {"temporal_caption": base.format(view=view), "duration": dur, "fps": 24.0,
            "resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["single", "shard"], default="single")
    ap.add_argument("--work", required=True, help="json list of [ep,cam,horizon,anchor,save_video]")
    ap.add_argument("--out", required=True, help="jsonl output (appended)")
    ap.add_argument("--report_dir", default="")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lpips", action="store_true", help="enable LPIPS (needs cached alexnet weights)")
    args = ap.parse_args()

    dev = "cuda:0"
    units = json.load(open(args.work))
    tag = os.path.basename(args.model)
    def log(*a): print(f"[{tag}]", *a, flush=True)
    log(f"loading ({args.mode}) ...")
    t0 = time.time(); pipe = load_pipe(args.model, args.mode); log(f"load {time.time()-t0:.1f}s; {len(units)} units")
    lpips_fn = None
    if args.lpips:
        try:
            import lpips as _l; lpips_fn = _l.LPIPS(net="alex").to(dev).eval()
        except Exception as e:
            log(f"lpips off ({e})")

    # group by (ep,cam) so each video decodes once
    from collections import defaultdict
    groups = defaultdict(list)
    for u in units: groups[(u[0], u[1])].append(u)
    outf = open(args.out, "a")
    done = 0
    for (ep, cam), us in groups.items():
        mp4 = f"{DATA}/videos/chunk-000/{CAM_KEY[cam]}/episode_{ep:06d}.mp4"
        vid = iio.imread(mp4, plugin="pyav")  # [N,H,W,C] uint8 @30fps
        N = vid.shape[0]
        for (_, _, hz, anchor, save_video) in us:
            gen_f, src_f = HORIZON[hz]
            if anchor >= N: continue
            cond = Image.fromarray(vid[anchor])
            t1 = time.time()
            res = pipe(prompt=json.dumps(cap_for(cam, hz)), negative_prompt=NEG, image=cond,
                       num_frames=gen_f, height=480, width=832, fps=24,
                       num_inference_steps=args.steps, guidance_scale=6.0, enable_safety_check=False,
                       add_resolution_template=False, add_duration_template=False, output_type="np",
                       generator=torch.Generator(dev).manual_seed(args.seed))
            gen_s = time.time() - t1
            pred = res.video[0] if isinstance(res.video, (list, tuple)) else res.video  # [T,H,W,C] float[0,1]
            pred = np.asarray(pred);
            if pred.ndim == 4 and pred.shape[0] == 1: pred = pred[0]
            pred_u8 = (np.clip(pred, 0, 1) * 255).astype("uint8")
            gt_win = vid[anchor:anchor + src_f]
            n = min(pred_u8.shape[0], gen_f)
            gt_rs = temporal_resample(gt_win, n)
            pe = to_eval_grid(pred_u8[:n], dev); ge = to_eval_grid(gt_rs, dev)
            m = video_metrics_gpu(pe, ge, lpips_fn, dev)
            rec = {"model": tag, "ep": ep, "cam": cam, "horizon": hz, "anchor": int(anchor),
                   "gen_s": round(gen_s, 1), **{k: round(v, 4) for k, v in m.items()}}
            outf.write(json.dumps(rec) + "\n"); outf.flush(); done += 1
            if args.report_dir:                                   # always save side-by-side (every window)
                os.makedirs(args.report_dir, exist_ok=True)
                stem = f"{tag}__{ep}__{cam}__{hz}__a{anchor:06d}"
                g640 = to_hw(gt_rs, 480, 640); p640 = to_hw(pred_u8[:n], 480, 640)
                sep = np.zeros((g640.shape[0], 480, 6, 3), dtype="uint8")   # thin black divider
                sbs = np.concatenate([g640, sep, p640], axis=2)            # GT | pred -> [T,480,1286,3]
                iio.imwrite(f"{args.report_dir}/{stem}__sbs.mp4", sbs, fps=24, codec="libx264")
            if done % 5 == 0: log(f"{done}/{len(units)} (last {hz} {gen_s:.0f}s)")
    outf.close(); log(f"DONE {done} units")

if __name__ == "__main__":
    main()
