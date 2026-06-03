"""WAM 训练启动器 —— 在共享 GPU 节点上等待显存空出后再启动。

scripts/train.py 直接 launch_from_config(config) 不带显存门控;本机 8×A100 常被其他
租户的作业占满(每卡仅 ~14GB 空闲)。本启动器调用 launch_from_config 时传入 gpu_memory
阈值,giga_train 的 wait_for_gpu_memory 会阻塞轮询,直到 config.launch.gpu_ids 里每张卡
都有 >= gpu_memory(MB)空闲,再真正拉起训练 —— 从而可后台挂起、GPU 一空出即自动开训。

用法:
  python -m scripts.wam_pipeline.launch_train \
      --config world_action_model.configs.visrobot01_fold.config \
      --gpu_memory_mb 70000 --seconds 60
"""
import argparse

from giga_train import launch_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dotted config path, e.g. world_action_model.configs.visrobot01_fold.config")
    ap.add_argument("--gpu_memory_mb", type=float, default=70000.0,
                    help="每张卡需空闲的显存(MB),达不到则阻塞等待。设 0 关闭门控(立即启动)。")
    ap.add_argument("--seconds", type=int, default=60, help="显存轮询间隔(秒)")
    args = ap.parse_args()

    gpu_memory = args.gpu_memory_mb if args.gpu_memory_mb and args.gpu_memory_mb > 0 else None
    launch_from_config(args.config, gpu_memory=gpu_memory, seconds=args.seconds)


if __name__ == "__main__":
    main()
