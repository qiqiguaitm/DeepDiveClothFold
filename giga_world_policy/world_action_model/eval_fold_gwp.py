"""gwp 内联 fold 评测核心:全 val 集 sharded MAE@{1,10,chunk/2,chunk}(action-only)。

镜像 scripts/wam_pipeline/episode_report.py 的 metric 路径(stock 引擎、action_only、exec stride、
per-window cumulative mae@h=ae[:h].mean()、per-ep 均值再全局均值、denorm+add_state),但:
  - 用【live transformer】(训练中 unwrap 出来)+ 训练已驻留的 VAE 包成 WAPipeline(只建一次,权重原地更新);
  - 用现有训练 rank 分片(shard_id=process_index, num_shards=num_processes);
  - 不重载 ckpt、不抢卡。

被 CasualWATrainer.inline_eval_fold 调用;try/except 由调用方负责(eval 失败不应中断训练)。
"""
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # giga_world_policy/


def _ensure_scripts_path():
    sp = os.path.join(_REPO, "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)


def horizons(action_chunk):
    return sorted({h for h in (1, 10, action_chunk // 2, action_chunk) if h <= action_chunk})


def build_eval_pipeline(model_id, vae, transformer, dtype, device):
    """用 live transformer + 已驻留 vae 包 WAPipeline(scheduler/config 从 model_id 读)。只建一次复用。"""
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    pipe = WAPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=dtype)
    # vae/transformer 已在 device 上;pipe 其余组件 to(device)
    try:
        pipe = pipe.to(device)
    except Exception:
        pass
    return pipe


def eval_fold_gwp(pipe, transformer, val_root, view_keys, stats_path, t5_pkl,
                  shard_id, num_shards, action_chunk=48, steps_inf=10, exec_horizon=16,
                  n_eps=100, max_win_per_ep=6, width=768, height=192, frame_cache=2,
                  device="cuda", dtype=torch.bfloat16, hor=None, log=print):
    _ensure_scripts_path()
    from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache, _hwc_to_chw01
    from world_action_model.pipeline.utils import (
        extract_normalization_tensors, load_stats, load_t5_embedding_from_pkl,
        denormalize_action, add_state_to_action, normalize_state, build_ref_image, resolve_delta_mask)
    from giga_datasets import load_dataset

    if hor is None:
        hor = horizons(action_chunk)
    stats = load_stats(stats_path)
    norm = extract_normalization_tensors(stats, device=device, state_dim=14, action_dim=14)
    t5 = load_t5_embedding_from_pkl(t5_pkl, target_len=64).to(device, torch.float32)
    dm = torch.tensor(resolve_delta_mask(stats, 14).tolist(), device=device, dtype=torch.bool)

    ve = dict(_class_name="LeRobotDataset", data_path=val_root, delta_info={"action": action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    ds = load_dataset([ve])
    fc = EpisodeFrameCache(val_root, view_keys, frame_cache)
    idx, _, info = build_window_indices(val_root, "exec", exec_horizon, action_chunk, exec_horizon)
    ep2win = OrderedDict()
    for gi in idx:
        ep2win.setdefault(info[gi][0], []).append(gi)
    eps = list(ep2win.keys())[:n_eps]
    my = eps[shard_id::num_shards]

    LOOKAHEAD = bool(getattr(transformer.config, "action_attends_video", False))
    ANS = bool(getattr(transformer.config, "async_noise", False))
    STEPS_ACT = 5 if ANS else None

    metric = {}
    for ep in my:
        wins = ep2win[ep]
        if len(wins) > max_win_per_ep:
            wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, max_win_per_ep).astype(int))]
        fr = fc.get(ep)
        em = {f"mae@{h}": [] for h in hor}
        for gi in wins:
            d = ds[int(gi)]
            _, f = info[int(gi)]
            ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in view_keys},
                                  dst_size=(width, height), crop_mode="center")
            st = d["observation.state"].float().unsqueeze(0).to(device)
            ns = normalize_state(st, norm, mode="zscore").to(device, dtype)
            with torch.no_grad():
                out = pipe(height=height, width=width, action_chunk=action_chunk, state=ns, num_frames=5,
                           guidance_scale=0.0, num_inference_steps=steps_inf, image=ref,
                           action_only=not LOOKAHEAD, action_num_inference_steps=STEPS_ACT,
                           finish_video=(STEPS_ACT is None), return_dict=False,
                           prompt_embeds=t5.unsqueeze(0).to(device, torch.float32))
            act = out[1]
            pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"),
                                     st[0].float().to(act.device), action_chunk=action_chunk, mask=dm).cpu().numpy()
            gt = d["action"].float().numpy()[:, :14]
            L = min(len(pa), len(gt)); ae = np.abs(pa[:L] - gt[:L])
            for h in hor:
                if h <= L:
                    em[f"mae@{h}"].append(float(ae[:h].mean()))
        metric[ep] = {f"mae@{h}": (float(np.mean(em[f"mae@{h}"])) if em[f"mae@{h}"] else None) for h in hor}
        log(f"[gwp_eval shard{shard_id}] ep{ep} mae@{hor[-1]}={metric[ep].get(f'mae@{hor[-1]}')}")
    return metric, hor


def aggregate(metrics_list, hor):
    eps = {}
    for m in metrics_list:
        eps.update(m)
    out = {}
    for h in hor:
        v = [r[f"mae@{h}"] for r in eps.values() if r.get(f"mae@{h}") is not None]
        out[h] = float(np.mean(v)) if v else None
    return out, len(eps)
