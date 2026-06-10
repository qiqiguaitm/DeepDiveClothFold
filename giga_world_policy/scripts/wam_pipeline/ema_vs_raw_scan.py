"""实测 EMA vs raw 的 held-out MAE 差随 step 的收敛(判断 EMA delay 何时可忽略)。
对给定/全部 ckpt,各用 transformer(raw) 与 transformer_ema 在相同抽样 window 上算 mae@{1,10,24,48},
打印对比 + |ema-raw|@48。常规 eval 已改用 raw,本脚本是一次性诊断(可在续训出更多 ckpt 后重跑)。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/wam_pipeline/ema_vs_raw_scan.py \
    --model_dir runs/visrobot01_fold_aihc_latent_5x/models --steps all --n 16 \
    --model_id "$WAN_DIFFUSERS" --stats_path assets_visrobot01/norm_stats_vis.json \
    --val_root "$GWP_DATA/visrobot01_val" --t5_pkl "$GWP_DATA/visrobot01_val/t5_embedding/episode_000000.pt"
"""
import argparse, os, sys, glob, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache, _hwc_to_chw01
VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--steps", default="all")
    ap.add_argument("--model_id", required=True); ap.add_argument("--stats_path", required=True)
    ap.add_argument("--val_root", required=True); ap.add_argument("--t5_pkl", required=True)
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--steps_inf", type=int, default=10); ap.add_argument("--delta_mask", default="")  # 空=从 --stats_path 内嵌 delta_mask 取;传 "1,1,..,0" 覆盖
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    from giga_datasets import load_dataset
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    from world_action_model.pipeline.utils import (extract_normalization_tensors, load_stats,
        load_t5_embedding_from_pkl, denormalize_action, add_state_to_action, normalize_state, build_ref_image)
    from diffusers.models import AutoencoderKLWan
    stats = load_stats(args.stats_path); norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=64).to(dev, torch.float32)
    if args.delta_mask.strip():
        _dm = [c == "1" for c in args.delta_mask.split(",")]
    else:
        from world_action_model.pipeline.utils import resolve_delta_mask
        _dm = resolve_delta_mask(stats, 14).tolist()
    dm = torch.tensor(_dm, device=dev, dtype=torch.bool)
    ve = dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": args.action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    ds = load_dataset([ve]); idx, _, info = build_window_indices(args.val_root, "exec", 0, args.action_chunk, 16)
    fc = EpisodeFrameCache(args.val_root, VK, 4)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    import random; random.seed(0); samp = random.sample(idx, args.n)

    def ev(sub):
        tf = CasualWorldActionTransformer.from_pretrained(sub).to(dt)
        pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev)
        H = {1: [], 10: [], 24: [], 48: []}
        for gi in samp:
            d = ds[int(gi)]; ep, f = info[int(gi)]; fr = fc.get(ep)
            ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in VK}, dst_size=(768, 192), crop_mode="center")
            st = d["observation.state"].float().unsqueeze(0).to(dev); ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
            with torch.no_grad():
                _, act = pipe(height=192, width=768, action_chunk=args.action_chunk, state=ns, num_frames=5,
                              guidance_scale=0.0, num_inference_steps=args.steps_inf, image=ref, action_only=True,
                              return_dict=False, prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32))
            pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"),
                                     st[0].float().to(act.device), action_chunk=args.action_chunk, mask=dm).cpu().numpy()
            gt = d["action"].float().numpy()[:, :14]; L = min(len(pa), len(gt)); ae = np.abs(pa[:L] - gt[:L])
            for h in H:
                if h <= L: H[h].append(ae[h - 1].mean())
        del pipe, tf; torch.cuda.empty_cache()
        return {h: float(np.mean(v)) for h, v in H.items()}

    allsteps = sorted(int(d.split("step_")[1]) for d in glob.glob(args.model_dir + "/checkpoint_*_step_*"))
    steps = allsteps if args.steps == "all" else [int(s) for s in args.steps.split(",")]
    print(f"{'step':>7} | {'raw@1':>7} {'raw@48':>7} | {'ema@1':>7} {'ema@48':>7} | {'|d|@48':>7}  (n={args.n})")
    for s in steps:
        base = glob.glob(args.model_dir + f"/checkpoint_*_step_{s}")
        if not base: continue
        raw = os.path.join(base[0], "transformer"); ema = os.path.join(base[0], "transformer_ema")
        r = ev(raw) if os.path.isdir(raw) else None
        e = ev(ema) if os.path.isdir(ema) else None
        rr = f"{r[1]:.4f} {r[48]:.4f}" if r else "  --      --  "
        ee = f"{e[1]:.4f} {e[48]:.4f}" if e else "  --      --  "
        dd = f"{abs(r[48]-e[48]):.4f}" if (r and e) else "  --  "
        print(f"{s:>7} | {rr} | {ee} | {dd:>7}")


if __name__ == "__main__":
    main()
