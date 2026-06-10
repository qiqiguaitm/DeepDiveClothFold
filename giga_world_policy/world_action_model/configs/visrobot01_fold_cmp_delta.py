"""小规模 delta vs abs 对比 —— DELTA 端(b0, 单节点 8卡, 1000 步)。

继承生产 5x latent 配方(5:1 loss、warmup-cosine、EMA off、cross-rig vis×3+kai),只:
  - 缩到 1000 步、warmup 100(让 LR 真正 ramp,1000 步内能看到 action_loss 下降);
  - 独立 project_dir;norm_path 保持 DELTA stats(关节 delta、夹爪 abs)。
配对端见 visrobot01_fold_cmp_abs(norm_path 换 *_abs.json,其余完全一致)。
验证目标:重构后(delta_mask 内嵌 stats + line-308 action_dim_mask 修复)训练端到端不回退。
"""
import copy

from world_action_model.configs.visrobot01_fold_aihc_latent_5x import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/cmp_delta_1k"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
# norm_path 保持 base 的 DELTA stats(norm_stats_vis.json / norm_stats_kai.json)

config["schedulers"] = dict(type="WarmupCosineScheduler", warmup_steps=100, decay_steps=1000)
config["train"].update(
    dict(
        resume=False,
        max_steps=1000,
        checkpoint_interval=500,   # 第 500、1000 步落 ckpt
        checkpoint_total_limit=3,
        with_ema=False,
        log_interval=10,
    )
)
