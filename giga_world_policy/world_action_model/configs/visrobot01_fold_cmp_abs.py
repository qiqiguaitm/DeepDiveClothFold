"""小规模 delta vs abs 对比 —— ABS 端(b1, 单节点 8卡, 1000 步)。

与 visrobot01_fold_cmp_delta **完全一致**,唯一区别:norm_path 指向 *_abs.json
(全 False mask → 关节也走 absolute、夹爪仍 abs)。同一初始化(action head from scratch +
Wan backbone),故 delta/abs 仅差动作表示与其 stats,是干净的对照实验。
验证目标:abs 路径能端到端训练 + eval,且与 delta 端可比 mae。
"""
import copy

from world_action_model.configs.visrobot01_fold_aihc_latent_5x import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/cmp_abs_1k"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
# 唯一区别:ABS stats(全 False delta_mask → 关节 absolute)
config["dataloaders"]["train"]["transform"]["norm_path"] = [
    "./assets_visrobot01/norm_stats_vis_abs.json",
    "./assets_visrobot01/norm_stats_kai_abs.json",
]

config["schedulers"] = dict(type="WarmupCosineScheduler", warmup_steps=100, decay_steps=1000)
config["train"].update(
    dict(
        resume=False,
        max_steps=1000,
        checkpoint_interval=500,
        checkpoint_total_limit=3,
        with_ema=False,
        log_interval=10,
    )
)
