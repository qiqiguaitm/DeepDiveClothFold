"""ABS 全量训练(50k)—— 对标 delta-5x 的 absolute-action 版本。

与生产基线 `visrobot01_fold_aihc_latent_5x` **完全相同的配方**(5:1 loss、warmup-cosine、
EMA off、cross-rig vis×3+kai),**唯一差别 = norm_path → *_abs.json**(全 False delta_mask →
关节也走 absolute;夹爪本就 abs)。

本任务 **4节点×8卡×bs2 = global batch 64 = delta-5x 实测 batch 64**(其 train log num_processes:32
batch_size:64)→ **batch/LR/samples 完全一致**,LR 即 delta-5x 的 2^-13.5,无需缩放。最干净的 A/B。

checkpoint:每 1k 落盘、保留 30 个(供 b0 单机 per-1k eval;watcher 及时评完即可,不必留全 50)。
对比基线:delta-5x report_step50000 raw_mae mae@1=0.0028 / mae@48=0.113(待 abs 对标)。
"""
import copy

from world_action_model.configs.visrobot01_fold_aihc_latent_5x import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/visrobot01_fold_abs_50k"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

# (1) ABS stats —— 唯一的表示差别
config["dataloaders"]["train"]["transform"]["norm_path"] = [
    "./assets_visrobot01/norm_stats_vis_abs.json",
    "./assets_visrobot01/norm_stats_kai_abs.json",
]

# LR = delta-5x 原值(4节点 batch 64 与 delta-5x 完全一致,无需缩放)
config["optimizers"]["lr"] = 2 ** (-13.5)

# 50k、每 1k 落 ckpt、保留 30 个供 per-1k eval;EMA 仍关(eval 用 raw),resume 开(容错续跑)
config["train"].update(
    dict(
        resume=True,
        max_steps=50000,
        checkpoint_interval=1000,
        checkpoint_total_limit=30,
        with_ema=False,
    )
)
