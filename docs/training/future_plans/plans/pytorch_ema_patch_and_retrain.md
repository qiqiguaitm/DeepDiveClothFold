# PyTorch EMA Patch + 重训 (修复 train_pytorch.py 缺 EMA)

> # ❌❌ 本 plan 已作废 (2026-05-31) — EMA 假说被 model-soup 实测证伪
>
> **不要执行本 plan。** 实测 (见 [`../../history/experiments/task_a_new_pure_200_new_norm_results.md`](../../history/experiments/task_a_new_pure_200_new_norm_results.md) §8.4 + [`../../analysis/pytorch_vs_jax_eval_postmortem.md`](../../analysis/pytorch_vs_jax_eval_postmortem.md)):
> - **model-soup (≈EMA) ≈ plain (差<1%)** → EMA **不是**主因, 假说证伪。
> - PyTorch 比 JAX 真差 (同协议 @50 4.1×), 根因疑为 **flow-matching sampler 实现差异**, 不是 EMA。加 EMA patch + 68h 重训对修复 gap **无用**。
> - 保留本文档仅作 "错误路线轨迹" + EMA patch 写法参考。真要修 gap 走 postmortem §5 (sampler 逐行对比)。
>
> ---
>
> **(以下为原 plan 内容, 已作废)**
> **状态**: ❌ 作废 (2026-05-31)
> **日期**: 2026-05-31

---

## 0. 为什么必须做 (实证依据)

| ckpt | @1 | @50 |
|---|---:|---:|
| JAX (真 EMA) | 0.0065 | 0.0087 |
| PyTorch plain 50k (no EMA) | 0.0121 | 0.0646 |
| **PyTorch model-soup 40k-50k (粗糙 EMA)** | **0.0073** | **0.0201** |

- soup (仅平均 6 个末段 ckpt) 已把 @50 收复 80% (0.0646→0.0201)
- 真 streaming EMA(0.9999) 预期把 PyTorch 拉到 ≈ JAX 0.0065/0.0087
- **所有后续 PyTorch 训练 (R1/R2 vis_v2_full 等) 都依赖此 patch**, 否则长 horizon 都会退化

---

## 1. EMA Patch 规格 (`kai0/scripts/train_pytorch.py`)

> 设计: 只在 `is_main` rank 维护 EMA shadow (DDP 各 rank params 同步一致, main 的即全局); EMA 权重存为 `model.safetensors` (与 JAX 一致, eval/deploy 自动取 EMA); raw 权重存为 `model_raw.safetensors` (resume 用)。

### Patch 1 — 初始化 EMA shadow (替换 line 513 的 "not supported" 日志)

**Anchor (line 513, 在 `if is_main:` 块内, 8-space 缩进)**:
```python
        logging.info("EMA is not supported for PyTorch training")
```

**替换为**:
```python
        logging.info(f"EMA enabled: decay={config.ema_decay}")
```

**并在训练循环开始前 (line 515 `model.train()` 之前, 4-space 函数体缩进) 插入**:
```python
    # --- EMA shadow (mirror JAX ema_decay behavior) ---
    ema_decay = config.ema_decay
    ema_state = None
    if ema_decay is not None:
        _m = model.module if hasattr(model, "module") else model
        # float params only; on same device, detached
        ema_state = {k: v.detach().clone().float() for k, v in _m.state_dict().items() if v.is_floating_point()}
```

### Patch 2 — 每步 EMA 更新 (在 `optimizer.step()` 之后)

**Anchor (line 530-531)**:
```python
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
        optimizer.step()
```

**替换为 (追加 EMA 更新)**:
```python
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
        optimizer.step()
        if ema_state is not None:
            _m = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                msd = _m.state_dict()
                for k in ema_state:
                    ema_state[k].mul_(ema_decay).add_(msd[k].detach().float(), alpha=1.0 - ema_decay)
```

### Patch 3 — 保存时写 EMA 为主 ckpt (在 save_checkpoint 调用后)

**Anchor (line 537)**:
```python
            save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
```

**替换为**:
```python
            save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
            if is_main and ema_state is not None:
                _write_ema_weights(config.checkpoint_dir / str(global_step), model, ema_state)
```

> ⚠️ save_checkpoint 用的目录名是 `global_step` (确认: ckpt 目录是 `2000/4000/.../50000`)。若 save_checkpoint 内部用了别的命名 (如 tmp→rename), 以其 final dir 为准 — 核对 line 155-183 后定 `_write_ema_weights` 的目标路径。

### Patch 4 — 新增 helper (在 `def save_checkpoint` 之前插入, line 152)

**Anchor (line 152)**:
```python
def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
```

**之前插入**:
```python
def _write_ema_weights(ckpt_dir, model, ema_state):
    """Overwrite ckpt model.safetensors with EMA weights; keep raw as model_raw.safetensors.

    EMA-as-primary mirrors JAX openpi (eval/deploy auto-pick model.safetensors = EMA).
    Raw weights preserved for exact-trajectory resume.
    """
    import safetensors.torch as _st
    from pathlib import Path as _P
    ckpt_dir = _P(ckpt_dir)
    model_path = ckpt_dir / "model.safetensors"
    raw_path = ckpt_dir / "model_raw.safetensors"
    if not model_path.exists():
        return
    _m = model.module if hasattr(model, "module") else model
    full = _m.state_dict()
    # 1) save raw (full state, original dtypes)
    _st.save_file(full, str(raw_path))
    # 2) build EMA full state: float keys from ema_state (cast back to orig dtype), non-float from raw
    out = {}
    for k, v in full.items():
        if k in ema_state:
            out[k] = ema_state[k].to(v.dtype)
        else:
            out[k] = v
    _st.save_file(out, str(model_path))
    logging.info(f"Wrote EMA -> {model_path} (raw -> {raw_path})")


```

### Patch 5 (resume 正确性, 可选但推荐)

resume 时 model 应载入 **raw** 权重 (继续优化轨迹), EMA shadow 从 `model.safetensors` 恢复:
- `load_checkpoint` 优先读 `model_raw.safetensors` (若存在) 载入 model
- ema_state 初始化后, 若 resume 且 `model.safetensors` 存在, 用它覆盖 ema_state

> 若本次只跑全新 50k (不 resume), Patch 5 可暂缓。

---

## 2. Smoke test (应用 patch 后, 重训前必做)

```bash
cd kai0
CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache \
  .venv/bin/python scripts/train_pytorch.py pi05_pytorch_a_new_pure_200 \
  --exp_name ema_smoke --save_interval 50 --num_train_steps 100
```
**验收**:
1. log 出现 `EMA enabled: decay=0.9999` (不再是 "not supported")
2. step 50/100 的 ckpt 目录里同时有 `model.safetensors` (EMA) + `model_raw.safetensors` (raw)
3. 两文件 size 均 ~12GB, 且**内容不同** (EMA ≠ raw): `python -c "import safetensors.torch as s; a=s.load_file('.../model.safetensors'); b=s.load_file('.../model_raw.safetensors'); import torch; ks=[k for k in a if a[k].is_floating_point()][0]; print((a[ks].float()-b[ks].float()).abs().mean())"` 应 > 0
4. 无 NaN, loss 正常下降

---

## 3. 重训

```bash
cd kai0
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_pytorch_a_new_pure_200 \
  --exp_name A_mirror200_pi05_pytorch_ema \
  --save_interval 2000
```
- 8× GPU, batch 128, 50k step, lr 1.5e-5→1.5e-6, ema_decay 0.9999 (config 已有)
- ETA ~68h (与无 EMA 版相近, EMA 更新开销极小)

---

## 4. 验收标准 (重训后 eval)

同 `eval_val_action_mse.py` on A_new_pure_200_val:

| 指标 | plain (no EMA) | soup (粗 EMA) | **EMA 重训目标** | JAX 参考 |
|---|---:|---:|---:|---:|
| MAE@1 | 0.0121 | 0.0073 | **≤ 0.0075** | 0.0065 |
| MAE@50 | 0.0646 | 0.0201 | **≤ 0.012** | 0.0087 |

- 若 EMA 重训 @50 ≤ 0.012 (接近 JAX 0.0087) → **PyTorch 路径修复确认**, 可放心用于 R1/R2 + 生产
- 残余 @1 gap (vs JAX 0.0065) 若仍在, 排查 PyTorch 训练 image aug (§8.4.3b) 是否需对齐 JAX

---

## 5. 后续连锁

- ✅ EMA 修复后, [`pytorch_native_vis_v2_full.md`](pytorch_native_vis_v2_full.md) R1/R2 才有意义 (否则长 horizon 必退化)
- realtime_vla 选项 X 部署侧也应确认用 EMA 权重 (`model.safetensors`), 而非 raw

---

## 附录: 临时 model-soup 验证脚本 (已用于 §0 验证)

`train_scripts/kai/eval/model_soup_ema_probe.py` — 均匀平均末段 ckpt 模拟 EMA, 已验证 soup 40k-50k → @50 0.0201。EMA 修复后此脚本不再需要 (但可留作快速诊断)。
