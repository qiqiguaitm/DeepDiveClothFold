"""GigaWorld-Policy(WAM)叠衣服微调 config —— 目标本体 visrobot01。

数据:../kai0/data/wam_fold_v1/{visrobot01,kairobot01}(由 scripts/build_wam_dataset.py 构建,
已去 depth、相机标准化为 cam_*、合并连续编号)。
双 embodiment 联合训练:visrobot01=embed_id 0(目标), kairobot01=embed_id 1(辅助),
本体标识通过 LeRobotDataset 的 embodiment= 参数注入(见 third_party/giga-datasets 改动)。

所有路径相对 repo root(运行 `python -m scripts.train` 时 cwd 应在 giga_world_policy/)。
"""

dst_size = (256, 192)
num_frames = 48
action_dim = 14

DATA = "../kai0/data/wam_fold_v1"
CKPT = "../checkpoints"

view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

# norm_path 顺序即 embed_id 索引:[0]=visrobot01, [1]=kairobot01(由 compute_norm_stats 生成)
norm_path = [
    "./assets_visrobot01/norm_stats_vis.json",
    "./assets_visrobot01/norm_stats_kai.json",
]


def _entry(emb):
    return dict(
        _class_name="LeRobotDataset",
        data_path=f"{DATA}/{emb}",
        data_size=None,
        embodiment=emb,  # 本体标识 → WAM 路由 norm_stats / delta_mask
        delta_info={"action": num_frames},
        delta_frames={k: image_frame_offsets for k in view_keys},
        # timestamp 经 regularize 后为 frame_index/30 的 float32,相邻 diff 因 float32 量化在
        # 0.0332~0.0334 间抖动(±~1.2e-4),超过 lerobot 默认 tolerance 1e-4 会报 timestamps
        # violate tolerance。设 1e-3:> float32 量化噪声、<< 半帧距 0.0167s,既过检查又不会取错帧。
        tolerance_s=1e-3,
    )


# visrobot01(目标域 ~2101 ep)上采样 3x 平衡 kairobot01(~6512 ep ≈ 3:1)
data_or_config = [_entry("visrobot01")] * 3 + [_entry("kairobot01")]

config = dict(
    project_dir="runs/visrobot01_fold",
    runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
    launch=dict(
        gpu_ids=[0],  # 冒烟单卡;全量训练改成全部可用卡,如 [0,1,2,3,4,5,6,7]
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(deepspeed_config_file='accelerate_configs/zero2.json'),
        until_completion=True,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=8,
            num_workers=8,
            transform=dict(
                type='WATransformsLerobot',
                robotype_to_embed_id={"visrobot01": 0, "kairobot01": 1},
                dst_size=dst_size,
                num_frames=num_frames,
                is_train=True,
                norm_path=norm_path,
                model_action_dim=action_dim,
                num_views=3,
                t5_len=64,
                view_keys=view_keys,
                image_cfg=dict(mask_generator=dict(max_ref_frames=1, start=1, factor=4)),
            ),
            sampler=dict(type='DefaultSampler'),
            collator=dict(is_equal=True),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=f"{CKPT}/Wan2.2-TI2V-5B-Diffusers",
        strict=False,
        action_dim=action_dim,
        flow_shift=5.0,
        expand_timesteps=True,
        view_dir="runs/visrobot01_fold",
        state_repeats=1,
    ),
    optimizers=dict(type='CAME8Bit', lr=2 ** (-14.5), weight_decay=1e-2),
    schedulers=dict(type='ConstantScheduler'),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=100000,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
        checkpoint_interval=5000,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=False,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
