"""Score gwp_ans (fp8+T_a3) on the dumped comparison frames. Run in gwp venv.
Metric (episode_report convention): mae@h = |pred[h-1]-gt[h-1]|.mean(), averaged over windows.
"""
import argparse, glob, json, os
import numpy as np
import torch

from diffusers.models import AutoencoderKLWan
from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.pipeline.wa_pipeline import WAPipeline
from world_action_model.pipeline.utils import (add_state_to_action, build_ref_image, denormalize_action,
    extract_normalization_tensors, load_stats, load_t5_embedding_from_pkl, normalize_state, resolve_delta_mask)

HOR = [1, 10, 24, 48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--transformer_path", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--t5_pkl", required=True)
    ap.add_argument("--steps_act", type=int, default=3)
    ap.add_argument("--steps_inf", type=int, default=10)
    ap.add_argument("--opt_tier", default="fp8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    W, H, AC = 768, 192, 48

    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    dm = torch.tensor(resolve_delta_mask(stats, 14).tolist(), device=dev, dtype=torch.bool)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=64).to(dev, torch.float32)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    tf = CasualWorldActionTransformer.from_pretrained(args.transformer_path).to(dt)
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev)
    LOOK = bool(getattr(tf.config, "action_attends_video", False))

    from scripts.opt_ans import AnsPrefixRunner, opt_call
    if args.opt_tier == "fp8":
        from scripts.fp8_linear import swap_linears_to_fp8
        swap_linears_to_fp8(tf.blocks)
    for m in tf.modules():
        if hasattr(m, "fuse_projections") and hasattr(m, "set_processor"):
            try: m.fuse_projections()
            except Exception: pass
    runner = AnsPrefixRunner(tf)
    runner.compile_prepare("reduce-overhead"); runner.compile_step_ans("reduce-overhead")

    files = sorted(glob.glob(os.path.join(args.frames_dir, "win_*.npz")))
    acc = {h: [] for h in HOR}
    for i, fp in enumerate(files):
        z = np.load(fp)
        imgs = {"observation.images.cam_high": torch.from_numpy(z["top_head"]).permute(2, 0, 1).float() / 255.0,
                "observation.images.cam_left_wrist": torch.from_numpy(z["hand_left"]).permute(2, 0, 1).float() / 255.0,
                "observation.images.cam_right_wrist": torch.from_numpy(z["hand_right"]).permute(2, 0, 1).float() / 255.0}
        ref = build_ref_image(images=imgs, dst_size=(W, H), crop_mode="center")
        st = torch.from_numpy(z["state"]).float().unsqueeze(0).to(dev)
        ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
        with torch.no_grad():
            act = opt_call(pipe, runner, image=ref, state=ns, prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32),
                           height=H, width=W, num_frames=5, action_chunk=AC, num_inference_steps=args.steps_inf,
                           action_num_inference_steps=args.steps_act, is_ans=LOOK, bac_skip=0)
        pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"),
                                 st[0].float(), action_chunk=AC, mask=dm).cpu().numpy()
        gt = z["gt"]
        L = min(len(pa), len(gt)); ae = np.abs(pa[:L] - gt[:L])
        for h in HOR:
            if h <= L: acc[h].append(float(ae[h - 1].mean()))
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(files)}", flush=True)
    res = {"model": "gwp_ans_fp8_ta3", "n_win": len(files), "mae": {h: float(np.mean(acc[h])) for h in HOR}}
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
