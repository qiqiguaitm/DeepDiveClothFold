# PyTorch 原生训练 pi05 vis_v2_full (R1 absolute + R2 delta)

> **决策日期**: 2026-05-23 PM。
> **状态**: ⏳ pending — config 待新加, smoke test 未跑。
> **关联 task**: `#18 PyTorch 原生训练: pi05 vis_v2_full (R1 absolute + R2 delta)`。

## 1. 动机

已实现 20Hz JAX→PyTorch 转换部署, 但**转换有精度损失**。需 PyTorch 原生训练直出 2 个 ckpt (delta + absolute), 用于 `realtime_vla_optimization_analysis.md` §1.4 选项 X "双推理架构并存" 路线 — 新 ckpt 走 PyTorch+Triton 5-10× 加速 (V1 论文 105ms→27ms)。

参考: [realtime_vla_optimization_analysis.md](../../../deployment/realtime_vla_optimization_analysis.md) §1.4 选项 X + §3.4.1 PyTorch 等效性 POC。
战略上下文: [cross_embodiment_strategy.md](../../../deployment/cross_embodiment_strategy.md)。

## 2. 数据与框架

| 项 | 配置 |
|---|---|
| 数据集 | `vis_v2_full` (1406 ep, 1.93M frames, 16 v2 dates 04-23 → 05-22) |
| 路径 (gf3 cnbj) | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full` |
| 路径 (gf0 cnsh) | `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full` |
| 训练框架 | **PyTorch DDP** (`scripts/train_pytorch.py`, 646 行) ⭐ 不是 JAX |
| 模型类 | `openpi.models_pytorch.pi0_pytorch.PI0Pytorch` (已支持 `pi05=True`) |
| Init | `pi05_base` PyTorch weights (从 JAX ckpt 转一次, 之后 PyTorch native 训练) |
| 训练 base config | 复用 `pi05_flatten_fold_vis_v2_full` (config.py:1300) hparams |

## 3. 两个新训练 config

### Exp R1: `pi05_pytorch_vis_v2_full_absolute`

```python
TrainConfig(
    name="pi05_pytorch_vis_v2_full_absolute",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_full",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,        # ← absolute
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    pytorch_weight_path=None,                  # 首次 PyTorch 训, 从 JAX ckpt 转 init
    pytorch_training_precision="bfloat16",
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
    ),
    ema_decay=0.9999,
    num_train_steps=50_000,
    keep_period=10_000,
    save_interval=2_000,
    num_workers=8,
    batch_size=128,
    fsdp_devices=8,
    inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_merged_val",
    inline_eval_n_frames=200,
    inline_eval_every=4,
),
```

### Exp R2: `pi05_pytorch_vis_v2_full_delta`

```python
TrainConfig(
    name="pi05_pytorch_vis_v2_full_delta",
    # 其他全同 R1
    data=LerobotAgilexDataConfig(
        ...
        use_delta_joint_actions=True,         # ← delta (唯一区别)
    ),
),
```

## 4. 训练参数对比表 (vs 历史 baseline)

| 参数 | 历史 smooth_800 (JAX) | **R1/R2 (PyTorch)** | 备注 |
|---|---|---|---|
| 训练框架 | JAX/Flax NNX | **PyTorch DDP (torchrun)** | ⭐ 关键差异 |
| 数据集 | vis_clean 800 ep | **vis_v2_full 1406 ep** | +76% 数据 |
| Init | mixed_1 (jax) | **pi05_base** (PyTorch) | 干净起点 |
| Steps | 50k | 50k | 同 |
| Batch | 128 | 128 | 同 |
| LR schedule | 1.5e-5 → 1.5e-6 cosine, warmup 1k | 同 | 一致 |
| EMA | 0.9999 | 0.9999 | 同 |
| Action 表示 | absolute | R1=absolute / **R2=delta** | R2 关键变体 |
| 推理用途 | JAX inference (现有) | **PyTorch+Triton (选项 X 路径 A, 5-10×加速)** | ⭐ 部署不同 |

## 5. 启动命令

```bash
# 16 GPU on Robot-North-H20 单节点
torchrun --standalone --nnodes=1 --nproc_per_node=16 \
  scripts/train_pytorch.py pi05_pytorch_vis_v2_full_absolute \
  --exp_name vis_v2_full_pytorch_absolute_v1 \
  --save_interval 2000

# R2 (delta) 同
torchrun --standalone --nnodes=1 --nproc_per_node=16 \
  scripts/train_pytorch.py pi05_pytorch_vis_v2_full_delta \
  --exp_name vis_v2_full_pytorch_delta_v1 \
  --save_interval 2000
```

## 6. 资源 + 时间预估

| 项 | R1 (absolute) | R2 (delta) |
|---|---|---|
| Resource | Robot-North-H20 **16 H20** | 同 |
| Training time | ~40-50h (PyTorch 比 JAX 慢 2-3×) | 同 |
| Storage (ckpt) | ~25 GB (keep_period 5 ckpts) | 同 |
| 并发 vs 串行 | **串行推荐** (避免抢资源 + bug 隔离) | — |
| 总 ETA | — | **~4-5 day 串行** |

## 7. 实施 TODO

| Task | 内容 | ETA | 阻塞 |
|---|---|---|---|
| T1 | 写 2 个 config 到 `config.py` | 0.5h | — |
| T2 | JAX pi05_base → PyTorch 权重 (验证 PI0Pytorch.from_pretrained) | 2-4h | — |
| T3 | uc02 / gf3 1 GPU smoke (~1k step, loss + 速度) | 4-6h | T1, T2 |
| T4 | R1 启动 16 H20 on Robot-North-H20 | ~40h | T3 OK |
| T5 | R2 启动 16 H20 (T4 完成后) | ~40h | T4 done |
| T6 | best ckpt 选 + 转 PyTorch+Triton 推理 | 1 day | T4/T5 |
| T7 | 真机评估 R1 vs R2 vs JAX→PyTorch 转换 ckpt | 1 day | T6 |

## 8. 风险

| # | 风险 | 应对 |
|---|---|---|
| 1 | PyTorch 训练等效性 (vs JAX) 未充分 POC | T3 比对 loss/速度; 若发散 → fallback 走 Y (JAX 训 + JAX→ONNX→TRT) |
| 2 | PyTorch 训练慢 (50k step JAX ~20h vs PyTorch ~40-50h) | 接受, 用 bf16 + DDP + grad_ckpt 优化 |
| 3 | `pytorch_weight_path` init 转换损失 | T2 一次性转换损失可控, vs 训完 ckpt 多次转换 |
| 4 | OOM (model 3B + batch 128) | bf16 已开; 不够 → batch 64 + grad_accum 2 |
| 5 | R2 (delta) norm_stats 重算 | `use_delta_joint_actions=True` 触发自动 transforms; data loader 内部 |

## 9. 决策点

- **D1**: T3 smoke 接受标准: 同 step loss 差 < 5%, 速度 < 3× JAX
- **D2**: R1/R2 训完真机评估 vs JAX→PyTorch 转换 ckpt — 量化"原生 PyTorch 训 vs 转换"精度收益
- **D3**: 若 R1+R2 都成功 → 标准化 PyTorch pipeline 作为 realtime_vla 选项 X 长线路径
