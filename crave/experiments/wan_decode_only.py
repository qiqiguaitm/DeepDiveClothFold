"""直接读 PNG → WanVAEEncoder 编码→平均→解码。用 srpo env(已确认可加载 WanVAEEncoder)。"""
import sys, warnings, gc
from pathlib import Path
import numpy as np
import cv2
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.home() / "workspace/deepdive_kai0/crave/src"))
sys.path.insert(0, str(Path.home() / "workspace/deepdive_kai0/crave/experiments"))
from crave.config.encoders import EncoderSpec
from crave.encoders.wan_vae import WanVAEEncoder

mat_dir = Path("/vePFS/tim/workspace/deepdive_kai0/crave/docs/visualization/pipeline_material")
pngs = sorted(mat_dir.glob("ep*.png"))
if not pngs: print("no ep*.png!"); raise SystemExit(1)
print(f"读取 {len(pngs)} 帧: {[p.stem for p in pngs]}")

imgs = []
for p in pngs:
    im = cv2.imread(str(p))[..., ::-1]
    if im.shape[:2] != (256, 256):
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    imgs.append(im)
print(f"帧尺寸: {len(imgs)}×{imgs[0].shape}")

spec = EncoderSpec("wan-vae","wan_vae",
    str(Path.home() / "workspace/deepdive_kai0/kai0/checkpoints/Wan2.2-TI2V-5B-Diffusers"),
    48*16*16,"fp16",256,16,0)
wan = WanVAEEncoder(spec, "cuda")
zs = wan.encode_latents(imgs, bs=6)
print(f"latent: {zs.shape}")
z_mean = zs.mean(0, keepdims=True)       # (1,48,16,16)
dec_mean = wan.decode(z_mean)[0]          # (256,256,3)
dec_first = wan.decode(zs[:1])[0]
print(f"decoded: {dec_mean.shape}")

from PIL import Image
Image.fromarray(dec_mean).save(str(mat_dir / "wan_decoded_centroid.png"))
Image.fromarray(dec_first).save(str(mat_dir / "wan_decoded_nearest.png"))
Image.fromarray(imgs[0]).save(str(mat_dir / "wan_nearest_real.png"))
print(f"SAVED to {mat_dir}")
