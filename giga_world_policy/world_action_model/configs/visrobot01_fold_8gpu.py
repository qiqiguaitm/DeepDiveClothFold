"""visrobot01 叠衣服微调 —— 8×A100 全量训练变体。

在 visrobot01_fold(冒烟默认 gpu_ids=[0])基础上做生产级覆盖,保持基础 config 不变:
  - launch.gpu_ids = 0..7(8 卡 DeepSpeed ZeRO-2)
  - train.activation_checkpointing = True(5B 全参微调 + 视频 latent,省激活显存)
  - batch_size_per_gpu = 4(保守取值规避 OOM;显存富余可调回 8。global = 4×8 = 32)

数据/transform/norm_path/动作维度等全部沿用 visrobot01_fold。
启动建议用 scripts.wam_pipeline.launch_train(带显存门控,等 GPU 空出再开训):
  python -m scripts.wam_pipeline.launch_train \
      --config world_action_model.configs.visrobot01_fold_8gpu.config \
      --gpu_memory_mb 70000 --seconds 60
"""
import copy

from world_action_model.configs.visrobot01_fold import config as _base

config = copy.deepcopy(_base)

config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
config["dataloaders"]["train"]["batch_size_per_gpu"] = 4
config["train"]["activation_checkpointing"] = True
