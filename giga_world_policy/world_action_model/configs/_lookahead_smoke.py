"""单节点 8卡 6步 smoke —— 验证 action_attends_video=True 的 _forward_train 端到端跑通
(real data + real 5B 架构,看 visual_loss/action_loss 有限、无 shape/NaN/OOM)。不存 ckpt。"""
import copy

from world_action_model.configs.visrobot01_fold_abs_lookahead import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/_lookahead_smoke"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
config["dataloaders"]["train"]["batch_size_per_gpu"] = 2
config["dataloaders"]["train"]["num_workers"] = 4
config["train"].update(
    dict(resume=False, max_steps=6, checkpoint_interval=100000,
         checkpoint_total_limit=1, with_ema=False, log_interval=1)
)
