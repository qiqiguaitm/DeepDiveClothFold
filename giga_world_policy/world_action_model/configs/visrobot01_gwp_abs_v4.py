"""gwp_abs_v4 —— 在【已修复】的干净 v3 数据上重训 abs,训练策略对齐 fastwam-v4。

根因复盘(2026-06-15):v3_abs/旧 v4 的 flat MAE 真因是 **v3 train parquet 的 episode_index
陈旧**(合并前值未重写),上游 LeRobot __getitem__ 按陈旧 episode_index 算 delta 区间 →
2353/2353 集 action 标签被夹成常量 → 模型只会吐常量块。已用
scripts/wam_pipeline/fix_v3_train_parquet_index.py 重写干净并端到端验证(动作恢复为真实轨迹)。

训练策略保持与 fastwam-v4 一致(继承自 visrobot01_v3_abs):
  1. training_weight=True          —— Gaussian bell(t=500) 加权
  2. independent_action_sigma=True —— t_action 独立采样,不与 t_video 共享(fastwam-v4 style)
注:这两项当初是按【已证伪的 flow_shift 理论】加的,但为与 fastwam-v4 配方对齐而保留。

其余超参(batch=320, LR=4.3e-4, 50k steps, WarmupCosine, λ_action, latent cache, v3 data)不变。
"""
import copy

from world_action_model.configs.visrobot01_v3_abs import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/gwp_abs_v4"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ
