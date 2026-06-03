"""visrobot01 叠衣服 full-FT —— 百度 AIHC 生产配置(节点数无关)。

GPU 数/拓扑由 AIHC 注入的 WORLD_SIZE/RANK + run_train_aihc.sh 的 accelerate CLI 决定,
不读 config.launch。当前目标:4 节点 × 8 A100 = 32 卡(global batch = 32 × bs2 = 64)。
本机 b2 跑解耦 eval-agent,watch 本 config 的 project_dir(共享 PFS)。

继承 visrobot01_fold(visrobot01_train 1898 + 修正 norm_stats + 双 embodiment + held-out 物理切分)。
"""
import copy

from world_action_model.configs.visrobot01_fold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/visrobot01_fold_aihc"
config["models"]["view_dir"] = "runs/visrobot01_fold_aihc"
config["dataloaders"]["train"]["batch_size_per_gpu"] = 2
# num_workers 是 per-rank;每节点 8 rank × 124 核 → 每 rank ~12(×8≈96/节点,留余量给主进程+VAE)。
# 设 32-48/rank 会 256-384 进程/节点严重超核 → 反而更慢。根治在 #1 latent 缓存(读小 latent 而非解 mp4)。
config["dataloaders"]["train"]["num_workers"] = 12
config["train"].update(
    dict(
        max_steps=50000,
        checkpoint_interval=2000,        # 每 2000 步落一个 → eval-agent 触发一次评估
        checkpoint_total_limit=15,
        checkpoint_safe_serialization=True,
        with_ema=True,
        resume=True,                     # 多机长跑必开:从 project_dir 最近 ckpt 续训
        activation_checkpointing=True,
        mixed_precision="bf16",
        log_interval=10,
    )
)
