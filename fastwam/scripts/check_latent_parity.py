"""latent 缓存数值对拍(放量前必须通过):
  对同一窗口:fast path(缓存)样本 vs 原路径(视频解码)样本
  1) action/proprio/context 逐位一致(同一 processor 链);
  2) 缓存 latents == VAE.encode(原路径 video)(确定性 VAE,bf16 严格相等)。
前置:compute_latents.py --smoke 1 已产出 episode_000000.pt(修复后的像素链)。
用法:CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python scripts/check_latent_parity.py
"""
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))
sys.path.insert(0, "src")


def main():
    from hydra import compose, initialize
    from hydra.utils import instantiate
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train",
                      overrides=["data=visrobot01_fold", "model=fastwam", "task=visrobot01_fold_uncond_1e-4"])

    cache_dir = cfg.data.train["latent_cache_dir"]
    # A) fast-path 数据集(带缓存)
    ds_fast = instantiate(cfg.data.train)
    assert ds_fast._cache_index is not None and len(ds_fast._cache_index) > 0, "cache index empty"
    # B) 原路径数据集(同配置去掉缓存)
    ds_orig = instantiate(cfg.data.train, latent_cache_dir=None)

    # 取 ep0 的第 0、第 7 个缓存窗口对拍
    picks = [i for i, (ep, gs, wi) in enumerate(ds_fast._cache_index) if ep == 0][:8:7]
    print(f"checking {len(picks)} windows: {[(ds_fast._cache_index[i]) for i in picks]}", flush=True)

    vae = None
    for i in picks:
        ep, gstart, wi = ds_fast._cache_index[i]
        fast = ds_fast[i]
        orig = ds_orig[gstart]          # 原路径 idx = 全局帧号

        for k in ("action", "proprio"):
            a, b = fast[k], orig[k]
            ok = torch.allclose(a, b, atol=1e-6)
            print(f"  win@{gstart} {k}: max|Δ|={float((a-b).abs().max()):.2e} {'OK' if ok else 'FAIL'}", flush=True)
            assert ok, f"{k} mismatch"
        assert fast["prompt"] == orig["prompt"], "prompt mismatch"
        assert torch.equal(fast["context"], orig["context"]), "context mismatch"

        # latent vs 在线 VAE
        if vae is None:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(REPO / "checkpoints")
            os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
            from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
            _, _, vae_cfg, _ = _resolve_configs("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B",
                                                redirect_common_files=False)
            vae_cfg.download_if_necessary()
            vae = _load_registered_model(str(vae_cfg.path), "wan_video_vae",
                                         torch_dtype=torch.bfloat16, device="cuda").eval()
        v = orig["video"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)   # [1,C,13,384,320]
        with torch.no_grad():
            z = vae.encode(v, device="cuda")
        z = (z[0] if isinstance(z, list) else z).cpu().squeeze(0)
        zc = fast["video_latents"]
        d = (z.float() - zc.float()).abs()
        print(f"  win@{gstart} latent: shape cache={tuple(zc.shape)} online={tuple(z.shape)} "
              f"max|Δ|={float(d.max()):.3e} mean|Δ|={float(d.mean()):.3e}", flush=True)
        assert zc.shape == z.shape, "latent shape mismatch"
        assert float(d.max()) < 1e-2, "latent numeric mismatch(像素链不一致?)"

    print("PARITY OK", flush=True)


if __name__ == "__main__":
    main()
