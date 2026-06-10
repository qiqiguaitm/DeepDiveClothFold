"""轨迹诊断:对一个 ckpt,在 held-out window 上跑 action-only 推理,画 pred vs GT vs stay-put
的 14 维 action 轨迹(over action_chunk 步),并给定量:每维 pred-GT 形状相关性 + 相对 stay-put 的
改善比。用于判断 action 头到底"学到运动形状没"(开环 MAE 劣于 stay-put ≠ 没学到形状)。

用法:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.wam_pipeline.viz_traj \
    --transformer_dir runs/visrobot01_fold_aihc_latent_5x/models/checkpoint_epoch_1_step_15000/transformer_ema \
    --model_id "$WAN_DIFFUSERS" --stats_path assets_visrobot01/norm_stats_vis.json \
    --val_root "$GWP_DATA/visrobot01_val" --t5_pkl "$GWP_DATA/visrobot01_val/t5_embedding/episode_000000.pt" \
    --n_windows 4 --out_dir runs/visrobot01_fold_aihc_latent_5x/traj_vis
"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))        # scripts/ (wam_pipeline)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root (world_action_model)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache, _hwc_to_chw01

VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
DIM_NAMES = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer_dir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--t5_pkl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_windows", type=int, default=4)
    ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--delta_mask", default="")  # 空=从 --stats_path 内嵌 delta_mask 取(默认);传 "1,1,..,0" 覆盖
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev, dt = "cuda", torch.bfloat16
    from giga_datasets import load_dataset
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    from world_action_model.pipeline.utils import (extract_normalization_tensors, load_stats,
        load_t5_embedding_from_pkl, denormalize_action, add_state_to_action, normalize_state, build_ref_image)
    from diffusers.models import AutoencoderKLWan
    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=64).to(dev, torch.float32)
    if args.delta_mask.strip():
        _dmask = [c == "1" for c in args.delta_mask.split(",")]
    else:
        from world_action_model.pipeline.utils import resolve_delta_mask
        _dmask = resolve_delta_mask(stats, 14).tolist()
    dmask = torch.tensor(_dmask, device=dev, dtype=torch.bool)
    ve = dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": args.action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    ds = load_dataset([ve]); idx, _, info = build_window_indices(args.val_root, "exec", 0, args.action_chunk, 16)
    fc = EpisodeFrameCache(args.val_root, VK, 4)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    tf = CasualWorldActionTransformer.from_pretrained(args.transformer_dir).to(dt)
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev)

    import random; random.seed(0)
    samp = random.sample(idx, args.n_windows)
    agg_corr, agg_pred_mae, agg_stay_mae = [], [], []
    for wi, gi in enumerate(samp):
        d = ds[int(gi)]; ep, f = info[int(gi)]; fr = fc.get(ep)
        ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in VK}, dst_size=(args.width, args.height), crop_mode="center")
        st = d["observation.state"].float().unsqueeze(0).to(dev)
        ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
        with torch.no_grad():
            _, act = pipe(height=args.height, width=args.width, action_chunk=args.action_chunk, state=ns,
                          num_frames=args.num_frames, guidance_scale=0.0, num_inference_steps=args.steps,
                          image=ref, action_only=True, return_dict=False,
                          prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32))
        pred = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"),
                                   st[0].float().to(act.device), action_chunk=args.action_chunk, mask=dmask).cpu().numpy()
        gt = d["action"].float().numpy()[:, :14]; L = min(len(pred), len(gt)); pred, gt = pred[:L], gt[:L]
        stay = np.repeat(st[0, :14].cpu().numpy()[None, :], L, axis=0)
        # 定量:每维形状相关 + mae
        for dd in range(14):
            g = gt[:, dd]
            if g.std() > 1e-4 and pred[:, dd].std() > 1e-4:
                agg_corr.append((dd, float(np.corrcoef(pred[:, dd], g)[0, 1])))
            agg_pred_mae.append((dd, float(np.abs(pred[:, dd] - g).mean())))
            agg_stay_mae.append((dd, float(np.abs(stay[:, dd] - g).mean())))
        # 画 14 维
        fig, axes = plt.subplots(7, 2, figsize=(13, 16)); axes = axes.flatten()
        xs = np.arange(L)
        for dd in range(14):
            ax = axes[dd]
            ax.plot(xs, gt[:, dd], "k--", lw=2, label="GT")
            ax.plot(xs, pred[:, dd], "r-", lw=1.5, label="pred")
            ax.plot(xs, stay[:, dd], "b:", lw=1, label="stay-put")
            c = next((cc for d2, cc in agg_corr if d2 == dd), float("nan"))
            ax.set_title(f"{DIM_NAMES[dd]}  corr={c:.2f}", fontsize=9)
            if dd == 0: ax.legend(fontsize=8)
        fig.suptitle(f"window {wi} (ep{ep} f{f})  pred vs GT vs stay-put", fontsize=12)
        fig.tight_layout(); p = os.path.join(args.out_dir, f"window{wi}_ep{ep}_f{f}.png"); fig.savefig(p, dpi=80); plt.close(fig)
        print(f"[viz] saved {p}")
    # 汇总
    import collections
    corr_by = collections.defaultdict(list); pmae = collections.defaultdict(list); smae = collections.defaultdict(list)
    for dd, c in agg_corr: corr_by[dd].append(c)
    for dd, m in agg_pred_mae: pmae[dd].append(m)
    for dd, m in agg_stay_mae: smae[dd].append(m)
    print("\n=== 定量(每维, n_windows 平均): corr=pred-GT形状相关, pred_mae vs stay_mae, ratio=pred/stay(<1=优于stayput) ===")
    cors = []
    for dd in range(14):
        c = float(np.mean(corr_by[dd])) if corr_by[dd] else float("nan")
        pm = float(np.mean(pmae[dd])); sm = float(np.mean(smae[dd]))
        cors.append(c if c == c else 0.0)
        print(f"  {DIM_NAMES[dd]:7s} corr={c:+.2f}  pred_mae={pm:.4f}  stay_mae={sm:.4f}  ratio={pm/(sm+1e-9):.2f}")
    valid = [c for _, c in agg_corr]
    print(f"\n>>> 平均形状相关(运动维) = {np.mean(valid):.3f} (>0.5=形状跟得不错,~0=没学到形状)")
    print(f">>> 优于 stay-put 的维度数 = {sum(1 for dd in range(14) if np.mean(pmae[dd])<np.mean(smae[dd]))}/14")


if __name__ == "__main__":
    main()
