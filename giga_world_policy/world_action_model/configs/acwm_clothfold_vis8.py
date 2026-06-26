"""visrobot-only AC-WM 正式训练 —— 8 卡(全机),resume=false。
修复 6 卡版在 step1000 checkpoint 保存时 OOM 的问题(8 卡 zero2 分片更小 + expandable_segments)。
继承 acwm_clothfold(visrobot vae_latent_v3fix, num_history=2, lambda_action=0, eval_fold)。"""
import copy
from world_action_model.configs.acwm_clothfold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/acwm_clothfold_vis"
config["models"]["view_dir"] = "runs/acwm_clothfold_vis"
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]   # 全机 8 卡(本机专给 AC-WM)
config["train"]["resume"] = False                         # 全新训练,勿尝试 resume 残缺 checkpoint
