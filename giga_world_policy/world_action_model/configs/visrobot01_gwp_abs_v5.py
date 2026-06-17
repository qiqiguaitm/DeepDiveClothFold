"""gwp_abs_v5 —— 用【相机顺序修正】的 latent 重训 gwp_abs(对照 v4)。

根因(2026-06-17,代码+像素双重确认):v3 的 vae_latent 由 fastwam compute_latents 用字母序探测
相机生成 → 把 hand_left(腕部)当 256×320 主图、top_head(俯视全局)压进 128×160 腕部小槽(降采样 4×)。
gwp_abs_v4 经 lerobot_dataset 的 fastwam-format 分支读这份 latent,即也在"俯视图被压扁"的次优 latent 上训练
(仍收敛 0.104,因三路相机都拍桌面)。

v5 = v4 完全相同配方,仅把 latent_dir 指向【修正版】vae_latent_v3fix(top_head 全分辨率当主图),
验证修正 latent 是否进一步提升 gwp_abs。eval 口径不变(visrobot01_v3_val)。
"""
import copy

from world_action_model.configs.visrobot01_gwp_abs_v4 import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/gwp_abs_v5"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

# 指向相机顺序修正后的 latent 缓存(vae_latent → vae_latent_v3fix)
for _ent in config["dataloaders"]["train"]["data_or_config"]:
    _ent["latent_dir"] = _ent["latent_dir"].replace("/vae_latent", "/vae_latent_v3fix")

# 内联 fold 评测(全 val sharded MAE@{1,10,24,48},复用训练 rank;见 world_action_model/eval_fold_gwp.py)
# 正式:every=1000 + n_eps=100(内联 eval 已验证跑通)。
config["models"]["eval_fold"] = dict(
    enabled=True,
    every=1000,
    n_eps=100,
    val_root="../kai0/data/wam_fold_v3/visrobot01_v3_val",
    stats_path="assets_visrobot01_v3/norm_stats_vis_abs.json",
    t5_pkl="../kai0/data/wam_fold_v3/visrobot01_v3_val/t5_embedding/episode_000000.pt",
    view_keys=[
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ],
    action_chunk=48,
    steps_inf=10,
    exec_horizon=16,
    max_win_per_ep=6,
    width=768,
    height=192,
    frame_cache=2,
)
