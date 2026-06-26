"""visrobot-only AC-WM 正式训练,GPU 0-5(留 6-7 给后台 kairobot 统一重抽,无 GPU 重叠)。
继承 acwm_clothfold(visrobot vae_latent_v3fix, num_history=2, lambda_action=0, eval_fold)。"""
import copy
from world_action_model.configs.acwm_clothfold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/acwm_clothfold_vis"
config["models"]["view_dir"] = "runs/acwm_clothfold_vis"
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5]   # 6 卡训练;6-7 留给后台 unify
