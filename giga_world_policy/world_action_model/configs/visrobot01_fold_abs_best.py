"""abs-BEST —— 把 absolute-action 推到最好(非受控对比,目标=abs 终质量最高)。

基于 `visrobot01_fold_abs_50k`(abs stats + 5:1 + EMA off),为"最好效果"改三处:
  1. batch_size_per_gpu 2→8 + 5节点 → global batch 320(smoke 实测 bs/gpu=8 单节点峰值 71G/80G,
     prod 40-way ZeRO-2 更省 ~1.5G → ~70G,偏紧但可跑;AIHC 容错兜 ckpt-save 偶发 spike)。
     flow-matching 每样本梯度因随机 timestep 极噪,大 batch 直接降噪 → 终质量更好(diffusion 通用经验)。
  2. peak LR 线性 ×5(batch 320 vs delta-5x 64)= 2^-13.5×5 ≈ 4.3e-4;warmup 2000→5000(大 LR 需更长热身)。
  3. max_steps 50000 不变 → batch320 = 5× 样本(~5× 墙钟);若算力紧可减步数。

收尾:跑完对**末段若干 ckpt 离线权重平均**(EMA 的事后版)再出终值,通常再降一档 mae。
"""
import copy

from world_action_model.configs.visrobot01_fold_abs_50k import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/visrobot01_fold_abs_best"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

# (1) 大 batch:bs/gpu 2→8(5节点 → global batch 320)
config["dataloaders"]["train"]["batch_size_per_gpu"] = 8
config["dataloaders"]["train"]["num_workers"] = 6

# (2) 线性 5× LR + 更长 warmup(batch 320 vs delta-5x 64)
config["optimizers"]["lr"] = 2 ** (-13.5) * 5.0          # ≈ 4.3e-4
config["schedulers"] = dict(type="WarmupCosineScheduler", warmup_steps=5000, decay_steps=50000)

# (3) 50k 步、每 1k ckpt、保留 30(供 b0 per-1k eval + 末段平均)
config["train"].update(
    dict(resume=True, max_steps=50000, checkpoint_interval=1000, checkpoint_total_limit=30, with_ema=False)
)
