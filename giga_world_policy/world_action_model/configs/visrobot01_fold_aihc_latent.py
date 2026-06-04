"""visrobot01 叠衣服 full-FT —— AIHC 生产配置(VAE latent 缓存版,根治数据瓶颈)。

读离线缓存的 VAE latent(scripts/wam_pipeline/compute_latents 预抽,S=4),跳过 mp4 解码 + VAE 编码,
配 episode-grouped sampler + dataset LRU → 单步接近算力上限(b2 实测 mp4 ~8.6s → latent ~1-2s,~10x)。

双 embodiment:visrobot01_train×3 + kairobot01;skip_video_decoding;latent_dir=各自 vae_latent。
拓扑由 run_train_aihc.sh + AIHC 注入决定(不读 config.launch)。
"""
dst_size = (256, 192); num_frames = 48; action_dim = 14
DATA = "../kai0/data/wam_fold_v1"; CKPT = "../checkpoints"
view_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
norm_path = ["./assets_visrobot01/norm_stats_vis.json", "./assets_visrobot01/norm_stats_kai.json"]


def _entry(emb, data_path):
    return dict(_class_name="LeRobotDataset", data_path=data_path, data_size=None, embodiment=emb,
                delta_info={"action": num_frames}, tolerance_s=1e-3,
                skip_video_decoding=True, latent_dir=f"{data_path}/vae_latent", latent_cache_size=6)


# visrobot01_train ×3 上采样平衡 kairobot01(≈3:1)
data_or_config = [_entry("visrobot01", f"{DATA}/visrobot01_train")] * 3 + [_entry("kairobot01", f"{DATA}/kairobot01")]

config = dict(
    project_dir="runs/visrobot01_fold_aihc_latent",
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
                flow_shift=5.0, expand_timesteps=True, view_dir="runs/visrobot01_fold_aihc_latent", state_repeats=1),
    optimizers=dict(type='CAME8Bit', lr=2 ** (-14.5), weight_decay=1e-2),
    schedulers=dict(type='ConstantScheduler'),
    train=dict(resume=True, max_epochs=0, max_steps=50000, gradient_accumulation_steps=1, mixed_precision='bf16',
               checkpoint_interval=1000, checkpoint_total_limit=20, checkpoint_safe_serialization=True,
               with_ema=True, activation_checkpointing=True, log_with='tensorboard', log_interval=10),
    test=dict(),
)
