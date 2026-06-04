"""visrobot01 叠衣服 full-FT —— warmup + cosine-decay 变体(诊断"早期 plateau"假设)。

诊断结论(见对话/eval 分析):原 latent 配置(ConstantScheduler, lr=2^-14.5, 无 warmup)下
train action_loss 在 ~step2400 即跌到 ~0.3 后**长期 plateau**,held-out action MAE 平坦且**劣于
"原地不动"基线**——典型的「低 lr 无 warmup 早早陷入劣解」。本配置:
  - WarmupCosineScheduler(warmup 2000 步 → 峰值,再 cosine 衰减到 0 @ 50000),
  - 峰值 lr 提到 2^-13.5≈8.6e-5(2×,给逃离 plateau 的机会;warmup 抑制早期不稳),
  - resume=False **从 base Wan 全新开始**(原 30k 步已 plateau,无续训价值),
  - 独立 project_dir,便于与原 run 的 eval 曲线 A/B。
其余(latent 缓存/双 embodiment/bs/workers)与 visrobot01_fold_aihc_latent 一致。
"""
dst_size = (256, 192); num_frames = 48; action_dim = 14
DATA = "../kai0/data/wam_fold_v1"; CKPT = "../checkpoints"
PROJ = "runs/visrobot01_fold_aihc_latent_warmup"
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
                flow_shift=5.0, expand_timesteps=True, view_dir=PROJ, state_repeats=1),
    optimizers=dict(type='CAME8Bit', lr=2 ** (-13.5), weight_decay=1e-2),   # 峰值 ≈8.6e-5(2×原)
    schedulers=dict(type='WarmupCosineScheduler', warmup_steps=2000, decay_steps=50000),
    train=dict(resume=False, max_epochs=0, max_steps=50000, gradient_accumulation_steps=1, mixed_precision='bf16',
               checkpoint_interval=1000, checkpoint_total_limit=20, checkpoint_safe_serialization=True,
               with_ema=True, activation_checkpointing=True, log_with='tensorboard', log_interval=10),
    test=dict(),
)
