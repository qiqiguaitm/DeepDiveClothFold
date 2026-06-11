"""单节点 8卡 6步 smoke —— 验证 async_noise=True 的 _forward_train 端到端(双 timestep、
action 切片、分支A loss 掩码)跑通:loss 有限、无 shape/NaN。不存 ckpt。"""
import copy

from world_action_model.configs.visrobot01_fold_abs_ans import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/_ans_smoke"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
config["dataloaders"]["train"]["batch_size_per_gpu"] = 2
config["dataloaders"]["train"]["num_workers"] = 4
config["models"]["ans_p"] = 0.5   # smoke 用大 p,保证分支A被覆盖到
config["train"].update(
    dict(resume=False, max_steps=6, checkpoint_interval=100000,
         checkpoint_total_limit=1, with_ema=False, log_interval=1)
)
