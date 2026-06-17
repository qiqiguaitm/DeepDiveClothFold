"""gwp-ori 基于 wam_fold_v3 重训 —— 纯 visrobot01 v3 数据，abs joint angle。

数据:
  vis_base_v3 (1840 ep) + vis_dagger_v3 (513 ep) → merge_and_split.py → visrobot01_v3_train (2353 ep)
  测试集:固定抽取 100 ep → visrobot01_v3_val (100 ep)，不进训练集

继承链: visrobot01_fold_abs_best → visrobot01_fold_abs_50k → visrobot01_fold_aihc_latent_5x
超参完全对齐 abs_best(batch=320, LR=4.3e-4, 50k steps, WarmupCosine, λ_action=5)。
变更:
  1. data_path → wam_fold_v3/visrobot01_v3_train
  2. 单 embodiment visrobot01，norm_path 用 v3 stats
  3. project_dir → runs/visrobot01_v3_abs
  4. view_keys → v3 相机名(top_head/hand_left/hand_right)
  5. skip_video_decoding=False，改用 DefaultSampler（v3 latent 尚未预计算）
"""
import copy

from world_action_model.configs.visrobot01_fold_abs_best import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/visrobot01_v3_abs"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

DATA = "../kai0/data/wam_fold_v3"

# v3 相机键
view_keys = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]


def _entry(emb, data_path):
    return dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        data_size=None,
        embodiment=emb,
        delta_info={"action": 48},
        delta_frames={k: list(range(0, 49, 4)) for k in view_keys},  # 13 帧/窗,stride=4
        tolerance_s=1e-3,
        skip_video_decoding=True,
        latent_dir=f"{data_path}/vae_latent",
        latent_cache_size=3,
    )


# visrobot01_v3_train × 4
config["dataloaders"]["train"]["data_or_config"] = [
    _entry("visrobot01", f"{DATA}/visrobot01_v3_train"),
    _entry("visrobot01", f"{DATA}/visrobot01_v3_train"),
    _entry("visrobot01", f"{DATA}/visrobot01_v3_train"),
    _entry("visrobot01", f"{DATA}/visrobot01_v3_train"),
]

# v3 norm stats
config["dataloaders"]["train"]["transform"]["norm_path"] = [
    "./assets_visrobot01_v3/norm_stats_vis_abs.json",
]

config["dataloaders"]["train"]["transform"]["robotype_to_embed_id"] = {
    "visrobot01": 0,
}

# 更新 view_keys（替换 transform 中的相机配置）
config["dataloaders"]["train"]["transform"]["view_keys"] = view_keys

# latent 缓存模式必须用 LatentEpisodeSampler（全局 index 匹配 ep["starts"]）
# num_frames=50: latent computed with range(0,L-49,4); sampler range(0,L-num_frames+1,4)=range(0,L-49,4) ✓
config["dataloaders"]["train"]["sampler"] = dict(type="LatentEpisodeSampler", stride=4, num_frames=50)

# Gaussian training_weight — aligns with fastwam-v4; upweights low-noise regime (σ<0.5)
# to fix abs-action convergence under flow_shift=5 (which puts 88% mass at σ>0.4).
config["models"]["training_weight"] = True

# Independent action sigma — aligns with fastwam-v4: t_action sampled completely fresh
# each step, decoupled from t_video (unlike ANS which enforces t_video >= t_action).
config["models"]["independent_action_sigma"] = True
