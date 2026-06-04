"""visrobot01 叠衣服 full-FT —— λ_action=5 变体(对齐官方 GigaWorld-Policy 后训练权重)。

诊断依据(见 docs/gigaworld_policy_recipe_vs_experiment.md):官方 GigaWorld-Policy 后训练用
λ_action=5, λ_video=1("emphasizing action prediction"),而本项目原配置为 **1:1** → action 欠监督。
此前 warmup+2×lr 实验已证 lr 非瓶颈(action_loss 仍 ~0.3)。本配置在 warmup+cosine 基础上把
**λ_action 提到 5**(λ_video=1),直接针对 action 欠加权;其余(latent 缓存/双 embodiment/bs/
warmup-cosine/peak 8.6e-5/resume=False/独立 project_dir)与 _warmup 变体一致,便于 A/B。
注:训练日志里 action_loss 打印的是**加权后**值(≈5×真实 velocity-MSE),看真实收敛需 ÷5。
"""
dst_size = (256, 192); num_frames = 48; action_dim = 14
DATA = "../kai0/data/wam_fold_v1"; CKPT = "../checkpoints"
PROJ = "runs/visrobot01_fold_aihc_latent_5x"
view_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
norm_path = ["./assets_visrobot01/norm_stats_vis.json", "./assets_visrobot01/norm_stats_kai.json"]


def _entry(emb, data_path):
    return dict(_class_name="LeRobotDataset", data_path=data_path, data_size=None, embodiment=emb,
                delta_info={"action": num_frames}, tolerance_s=1e-3,
                skip_video_decoding=True, latent_dir=f"{data_path}/vae_latent", latent_cache_size=6)


data_or_config = [_entry("visrobot01", f"{DATA}/visrobot01_train")] * 3 + [_entry("kairobot01", f"{DATA}/kairobot01")]

config = dict(
    project_dir=PROJ,
    runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
    launch=dict(gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], distributed_type='DEEPSPEED',
                deepspeed_config=dict(deepspeed_config_file='accelerate_configs/zero2.json'), until_completion=True),
    dataloaders=dict(train=dict(
        data_or_config=data_or_config,
        batch_size_per_gpu=2, num_workers=8, prefetch_factor=6, persistent_workers=True,
        transform=dict(type='WATransformsLerobot', robotype_to_embed_id={"visrobot01": 0, "kairobot01": 1},
                       dst_size=dst_size, num_frames=num_frames, is_train=True, norm_path=norm_path,
                       model_action_dim=action_dim, num_views=3, t5_len=64, view_keys=view_keys,
                       image_cfg=dict(mask_generator=dict(max_ref_frames=1, start=1, factor=4))),
        sampler=dict(type='LatentEpisodeSampler', stride=4),
        collator=dict(is_equal=True)), test=dict()),
    models=dict(pretrained=f"{CKPT}/Wan2.2-TI2V-5B-Diffusers", strict=False, action_dim=action_dim,
                flow_shift=5.0, expand_timesteps=True, view_dir=PROJ, state_repeats=1,
                lambda_action=5.0, lambda_video=1.0),   # ← 官方后训练权重 5:1
    optimizers=dict(type='CAME8Bit', lr=2 ** (-13.5), weight_decay=1e-2),
    schedulers=dict(type='WarmupCosineScheduler', warmup_steps=2000, decay_steps=50000),
    train=dict(resume=False, max_epochs=0, max_steps=50000, gradient_accumulation_steps=1, mixed_precision='bf16',
               checkpoint_interval=1000, checkpoint_total_limit=20, checkpoint_safe_serialization=True,
               with_ema=True, activation_checkpointing=True, log_with='tensorboard', log_interval=10),
    test=dict(),
)
