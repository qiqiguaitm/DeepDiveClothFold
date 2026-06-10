"""诊断:逐维 mae(delta vs abs),验证 abs@1k 比 delta@1k 差是表示效应还是 bug。
- 夹爪(idx6/13)在两种配置都是 absolute → 两个模型逐维 mae 应基本一致(控制组)。
- 关节(其余12维)delta vs abs 的 mae 差,应≈各维 denorm-std 比(scale 效应)。
小样本(coverage=sample, --n 窗口),单卡,与 eval_watch 同一套重建(denorm + add_state·mask)。
"""
import argparse
import numpy as np
import torch

from scripts.wam_pipeline.eval_watch import EpisodeFrameCache, build_window_indices, _hwc_to_chw01
from world_action_model.pipeline.utils import (
    add_state_to_action, build_ref_image, denormalize_action, extract_normalization_tensors,
    load_stats, load_t5_embedding_from_pkl, normalize_state, resolve_delta_mask,
)


def run(ckpt, stats_path, n, val_root, model_id, t5_pkl, device="cuda"):
    from diffusers.models import AutoencoderKLWan
    from giga_datasets import load_dataset
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    dtype = torch.bfloat16
    AC = 48
    vk = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    val_ds = load_dataset([dict(_class_name="LeRobotDataset", data_path=val_root,
                                delta_info={"action": AC}, skip_video_decoding=True,
                                embodiment="visrobot01", tolerance_s=1e-3)])
    fc = EpisodeFrameCache(val_root, vk, cache_size=2)
    stats = load_stats(stats_path)
    norm = extract_normalization_tensors(stats, device=device, state_dim=14, action_dim=14)
    mask = torch.tensor(resolve_delta_mask(stats, 14).tolist(), device=device, dtype=torch.bool)
    t5 = load_t5_embedding_from_pkl(t5_pkl, target_len=64).to(device=device, dtype=torch.float32)
    all_idxs, _, info = build_window_indices(val_root, "exec", 0, AC, 16)
    sel = [all_idxs[i] for i in np.unique(np.linspace(0, len(all_idxs) - 1, n).astype(int))]
    sel = sorted(sel, key=lambda gi: info[gi])

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    tf = CasualWorldActionTransformer.from_pretrained(ckpt).to(dtype)
    pipe = WAPipeline.from_pretrained(model_id, vae=vae, transformer=tf, torch_dtype=dtype).to(device)

    ae1 = np.zeros(14); aeC = np.zeros(14); nw = 0
    for gi in sel:
        d = val_ds[int(gi)]; ep, f = info[int(gi)]
        epf = fc.get(ep); Lf = epf[vk[0]].shape[0]
        ref = build_ref_image(images={k: _hwc_to_chw01(epf[k][f]) for k in vk}, dst_size=(768, 192), crop_mode="center")
        state = d["observation.state"].float().unsqueeze(0).to(device)
        nstate = normalize_state(state, norm, mode="zscore").to(device=device, dtype=dtype)
        with torch.no_grad():
            _, action = pipe(height=192, width=768, action_chunk=AC, state=nstate, num_frames=5,
                             guidance_scale=0.0, num_inference_steps=10, image=ref, action_only=False,
                             return_dict=False, prompt_embeds=t5.unsqueeze(0).to(device=device, dtype=torch.float32))
        pa = add_state_to_action(denormalize_action(action[0].float(), norm, mode="zscore"),
                                 state[0].float().to(action.device), action_chunk=AC, mask=mask).cpu().numpy()
        ga = d["action"].float().numpy(); L = min(len(pa), len(ga))
        ae = np.abs(pa[:L] - ga[:L])
        ae1 += ae[0]; aeC += ae.mean(0); nw += 1
    return ae1 / nw, aeC / nw, nw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--val_root", default="../kai0/data/wam_fold_v1/visrobot01_val")
    ap.add_argument("--model_id", default="../checkpoints/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--t5_pkl", default="../kai0/data/wam_fold_v1/visrobot01_val/t5_embedding/episode_000000.pt")
    args = ap.parse_args()
    D = "runs/cmp_delta_1k/models/checkpoint_epoch_1_step_1000/transformer"
    A = "runs/cmp_abs_1k/models/checkpoint_epoch_1_step_1000/transformer"
    print(f"[diag] n={args.n} windows per model")
    d1, dC, nd = run(D, "assets_visrobot01/norm_stats_vis.json", args.n, args.val_root, args.model_id, args.t5_pkl)
    a1, aC, na = run(A, "assets_visrobot01/norm_stats_vis_abs.json", args.n, args.val_root, args.model_id, args.t5_pkl)
    JOINTS = [i for i in range(14) if i not in (6, 13)]
    print(f"\n{'dim':>4} {'kind':>7} | {'delta mae@1':>11} {'abs mae@1':>10} {'ratio':>6} | {'delta maeC':>10} {'abs maeC':>9} {'ratio':>6}")
    for i in range(14):
        kind = "GRIP" if i in (6, 13) else "joint"
        r1 = a1[i] / d1[i] if d1[i] > 1e-9 else float('nan')
        rC = aC[i] / dC[i] if dC[i] > 1e-9 else float('nan')
        print(f"{i:>4} {kind:>7} | {d1[i]:>11.4f} {a1[i]:>10.4f} {r1:>6.2f} | {dC[i]:>10.4f} {aC[i]:>9.4f} {rC:>6.2f}")
    print(f"\n[GRIPPERS idx6,13] delta mae@1={d1[[6,13]].mean():.4f} abs mae@1={a1[[6,13]].mean():.4f} "
          f"(ratio {a1[[6,13]].mean()/d1[[6,13]].mean():.2f}; expect ~1.0 if no global bug)")
    print(f"[JOINTS 12dim]     delta mae@1={d1[JOINTS].mean():.4f} abs mae@1={a1[JOINTS].mean():.4f} "
          f"(ratio {a1[JOINTS].mean()/d1[JOINTS].mean():.2f}; denorm-std ratio ~1.87)")
