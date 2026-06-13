"""gwp-ori 重训 —— 只用 visrobot01_train(去掉 kairobot01 跨 embodiment 混训)。

与 abs-best 配方完全相同的超参(batch=320, LR=4.3e-4, 50k steps, WarmupCosine);
变更:
  1. 去掉 kairobot01 条目 → 单 embodiment {"visrobot01": 0}
  2. norm_path 只留 norm_stats_vis_abs.json
  3. 数据:visrobot01_train × 4(4 slot 结构不变,kairobot01 slot 换成 train)

动机:kairobot01(6512 ep)与 visrobot01 机型不同(关节角 mapping 差异),
  混训时梯度互相干扰 → action loss 难收敛;去掉后梯度全部来自目标 domain。
"""
import copy

from world_action_model.configs.visrobot01_fold_abs_best import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/visrobot01_abs_visonly"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

DATA = "../kai0/data/wam_fold_v1"


def _entry(emb, data_path):
    return dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        data_size=None,
        embodiment=emb,
        delta_info={"action": 48},
        tolerance_s=1e-3,
        skip_video_decoding=True,
        latent_dir=f"{data_path}/vae_latent",
        latent_cache_size=3,
    )


# visrobot01_train × 4(去掉 kairobot01,全部用 train,4 slot 结构不变)
config["dataloaders"]["train"]["data_or_config"] = [
    _entry("visrobot01", f"{DATA}/visrobot01_train"),
    _entry("visrobot01", f"{DATA}/visrobot01_train"),
    _entry("visrobot01", f"{DATA}/visrobot01_train"),
    _entry("visrobot01", f"{DATA}/visrobot01_train"),
]

# 单 embodiment → 只需 visrobot01 norm stats
config["dataloaders"]["train"]["transform"]["norm_path"] = [
    "./assets_visrobot01/norm_stats_vis_abs.json",
]

# 去掉 kairobot01 embed id
config["dataloaders"]["train"]["transform"]["robotype_to_embed_id"] = {
    "visrobot01": 0,
}
