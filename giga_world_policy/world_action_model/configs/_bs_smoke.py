"""单节点 8卡 显存 smoke(测 batch_size_per_gpu 上限)。SMOKE_BS 环境变量控制 bs/gpu。
跑 ~12 步、不存 ckpt;从训练日志 'cuda: alloc/reserved G' 读峰值 reserved / 看是否 OOM。
单节点 ZeRO-2 把 optimizer/grad 分到 8 卡(prod 4节点是 32 卡)→ per-gpu 显存比 prod 略高,
故此处能跑通 = prod 一定能跑通(保守上界)。"""
import copy
import os

from world_action_model.configs.visrobot01_fold_abs_50k import config as _base

config = copy.deepcopy(_base)
BS = int(os.environ.get("SMOKE_BS", "4"))
PROJ = f"runs/_bs_smoke/bs{BS}"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
config["dataloaders"]["train"]["batch_size_per_gpu"] = BS
config["dataloaders"]["train"]["num_workers"] = 4
config["train"].update(
    dict(resume=False, max_steps=12, checkpoint_interval=100000,
         checkpoint_total_limit=1, with_ema=False, log_interval=1)
)
