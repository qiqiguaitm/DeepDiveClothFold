"""serving 全栈延迟剖析:拆 preprocess / VAE-encode / prepare(前缀30层) / per-step。
真实 gwp_ans 权重 + fp8 部署档 + T_a=3。"""
import sys, time, argparse
sys.path.insert(0, ".")
import torch
from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.pipeline.wa_pipeline import WAPipeline
from scripts.opt_ans import AnsPrefixRunner
from scripts.fp8_linear import swap_linears_to_fp8
from diffusers.models import AutoencoderKLWan

ap = argparse.ArgumentParser()
ap.add_argument("--transformer_dir", required=True)
ap.add_argument("--model_id", required=True)
ap.add_argument("--tier", default="fp8")
ap.add_argument("--ta", type=int, default=3); ap.add_argument("--tv", type=int, default=10)
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()
dev, dt = torch.device("cuda"), torch.bfloat16
H, W, NF, AC = 192, 768, 5, 48

vae = AutoencoderKLWan.from_pretrained(a.model_id, subfolder="vae", torch_dtype=dt)
tf = CasualWorldActionTransformer.from_pretrained(a.transformer_dir).to(device=dev, dtype=dt).eval()
if a.tier == "fp8":
    print("fp8 swapped", swap_linears_to_fp8(tf.blocks))
for mod in tf.modules():
    if hasattr(mod, "fuse_projections") and hasattr(mod, "set_processor"):
        try: mod.fuse_projections()
        except Exception: pass
pipe = WAPipeline.from_pretrained(a.model_id, vae=vae, transformer=tf, text_encoder=None, torch_dtype=dt).to(dev)
runner = AnsPrefixRunner(tf)
runner.compile_prepare("reduce-overhead"); runner.compile_step_ans("reduce-overhead")

img_raw = torch.rand(1, 3, H, W, device=dev)            # 合成相机帧 [0,1]
state = torch.randn(1, 1, 14, device=dev, dtype=dt)
enc = torch.randn(1, 64, tf.config.text_dim, device=dev, dtype=torch.float32)
def sync(): torch.cuda.synchronize()
def run_once(prof):
    pipe.scheduler.set_timesteps(a.tv, device=dev); pipe.action_scheduler.set_timesteps(a.ta, device=dev)
    tt, at = pipe.scheduler.timesteps, pipe.action_scheduler.timesteps
    t = {}
    sync(); s = time.perf_counter()
    img = pipe.video_processor.preprocess(img_raw, height=H, width=W).to(dev, torch.float32)
    sync(); t["preprocess"] = time.perf_counter() - s
    s = time.perf_counter()
    lat, cond, ffm, action = pipe.prepare_latents(img, 1, pipe.vae.config.z_dim, H, W, NF,
                                                   torch.float32, dev, None, None, None, AC)
    action = action.to(dt); sync(); t["vae+prep_latents"] = time.perf_counter() - s
    refc = ((1 - ffm) * cond + ffm * lat)[:, :, :1].to(dt)
    s = time.perf_counter()
    runner.prepare_ans(refc, lat[:, :, 1:].to(dt), enc.to(dt), state); runner.set_action_rope(AC)
    sync(); t["prepare(prefix30L)"] = time.perf_counter() - s
    s = time.perf_counter()
    for i in range(a.ta):
        noisy = lat[:, :, 1:].to(dt)
        ap_, np_ = runner.step_ans(action, noisy, at[i].to(dt), tt[i].to(dt))
        ln = pipe.scheduler.step(np_.float(), tt[i], lat[:, :, 1:], return_dict=False)[0]
        lat = torch.cat([lat[:, :, :1], ln], dim=2)
        action = pipe.action_scheduler.step(ap_.float(), at[i], action.float(), return_dict=False)[0].to(dt)
    sync(); t["steps(%d)" % a.ta] = time.perf_counter() - s
    return t

for _ in range(5): run_once(False)   # warmup(含 compile)
agg = {}
for _ in range(a.iters):
    for k, v in run_once(True).items(): agg.setdefault(k, []).append(v * 1000)
import statistics as st
total = 0
print(f"\n=== gwp_ans {a.tier} T_a={a.ta} 延迟剖析(ms, {a.iters} 次)===")
for k, vs in agg.items():
    m = st.mean(vs); total += m
    print(f"  {k:22s} {m:6.1f} ± {st.stdev(vs):.1f}")
print(f"  {'TOTAL':22s} {total:6.1f}  | VAE+preproc 占比 {(agg['preprocess'][0]*0+st.mean(agg['preprocess'])+st.mean(agg['vae+prep_latents']))/total*100:.0f}%")
