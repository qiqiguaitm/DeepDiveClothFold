# V1 Triton 推理集成 (35 ms target)

> deepdive_kai0 pi05 推理走 V1 (Running VLAs at Real-time Speed) 手写 Triton kernel + CUDA Graph
> 实测 5090 3-view = **35.4 ms** (vs PyTorch E max-autotune 41.0 ms, -14%)
> 数值: rel error 1.42%, per-dim MAE < 0.01 (部署级)

## 文件清单

| 文件 | 用途 |
|---|---|
| `pi05_infer.py` | V1 主推理类 (Pi05Inference, 22+ Triton kernel + CUDA Graph capture) |
| `pi0_infer.py` | V1 Pi0 推理类 (含基础 Triton kernel, 被 pi05 import) |
| `dm0_infer.py` | V1 DM0 推理类 (benchmark.py 需要 import) |
| `convert_from_jax_pi05.py` | V1 原版 JAX → V1 weight dict 转换 (依赖 HF AutoTokenizer) |
| `convert_kai0_to_v1.py` | **我们的 adapter**: deepdive_kai0 ckpt → V1 pkl, 用 sentencepiece (绕开 HF AutoTokenizer) |
| `benchmark_kai0_v1.py` | 真 ckpt 100-iter benchmark |
| `benchmark.py` | V1 原版 benchmark (synthetic random weight) |
| `test.py` | V1 原版 PiModelEvaluator (含 JAX vs Triton 对比) |
| `numerical_compare.py` | **我们的数值对比**: 3 phase (jax / triton / compare), shared noise, V1 test.py 风格 input/output 对齐 |

## 使用流程

### 一次性: 转换 deepdive_kai0 ckpt → V1 pkl

```bash
cd /home/tim/workspace/deepdive_kai0
CKPT=/data1/DATA_IMP/checkpoints/task_a_mix_b6000_p1200_mixed_1_step49999
OUT=optimize/results/task_a_mix_b6000_p1200_v1.pkl

kai0/.venv_5090_trt/bin/python optimize/v1_triton/convert_kai0_to_v1.py \
    --jax_path "$CKPT" \
    --output "$OUT" \
    --prompt "Flatten and fold the cloth" \
    --tokenizer_model /data1/tim/workspace/deepdive_kai0/openpi_cache/big_vision/paligemma_tokenizer.model
```

输出: 6.7 GB pkl, 含 V1 格式 weights + embedding_weight + time_embeds (10 步) + language_embeds.

注意: `language_embeds` 在 pkl 里是基于 fixed task prompt 预算的。**真实推理时**, 若 state 变化需要重新生成 language_embeds (因为 deepdive_kai0 prompt 含 digitized state tokens). 见 `numerical_compare.py:phase_triton` 的处理方式。

### benchmark (speed)

```bash
CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090_trt/bin/python optimize/v1_triton/benchmark_kai0_v1.py \
    --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \
    --num-views 3 --chunk-size 50 --n-test 100
```

期望输出:
```
Mean: 35.385 ms, P50: 35.393 ms, P99: 35.489 ms, Std: 0.046 ms
```

### 数值对齐验证

```bash
# Phase JAX (主 venv)
OPENPI_EXTRA_CONFIG=$CKPT/train_config.json CUDA_VISIBLE_DEVICES=3 \
  kai0/.venv/bin/python optimize/v1_triton/numerical_compare.py jax \
    --ckpt $CKPT \
    --base-config-name pi05_flatten_fold_mix_b6000_p1200_init_mixed_1 \
    --out /tmp/jax_out.npz

# Phase Triton (.venv_5090_trt)
NORM_STATS=$CKPT/assets/mix_b6000_p1200/norm_stats.json
TOK=/data1/tim/workspace/deepdive_kai0/openpi_cache/big_vision/paligemma_tokenizer.model
CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090_trt/bin/python optimize/v1_triton/numerical_compare.py triton \
    --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \
    --inputs /tmp/jax_out.npz \
    --norm-stats $NORM_STATS \
    --tokenizer-model $TOK \
    --out /tmp/triton_out.npz

# Phase Compare
kai0/.venv_5090_trt/bin/python optimize/v1_triton/numerical_compare.py compare \
    --jax /tmp/jax_out.npz --triton /tmp/triton_out.npz
```

期望输出:
```
maxabs diff: 2.7e-2
mean abs diff: 3.7e-3
rel error: 1.42%
per-dim MAE: all < 0.01
```

## 关键技术点

### 1. V1 kernel 在 5090 上无需修改

V1 用通用 Triton, BLOCK_SIZE 在调用处传 (constexpr), Triton 编译器自动针对 sm_120 生成 PTX。phantom env smoke test + 真 ckpt benchmark 都验证可行, 5090 上 35.4ms (vs 4090 的 39.2ms README 报告, 5090 快 ~10%)。

### 2. deepdive_kai0 输入 pipeline 差异

V1 test.py 默认 PaliGemma 3-camera 命名 `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`, 但 deepdive_kai0 (agilex_policy) 用 `top_head / hand_left / hand_right`. 转换器无需关心 (V1 内部不区分 camera 名称, 只看 num_views=3)。

### 3. Output post-process: 不要加 ori_state

V1 test.py 在 Triton output 上 `actions = unnormalize + ori_state` 是针对 V1 reference Pi05 训练目标 (delta action). **deepdive_kai0 训练目标是绝对 action**, 所以只需要 unnormalize, **不要** + ori_state. 这是数值对齐的关键 (没改之前 Triton 输出是 JAX 的 1.7×)。

### 4. Language embeds 需要包含 digitized state

deepdive_kai0 PaligemmaTokenizer 把 state digitize 后塞进 prompt: `"Task: {prompt}, State: {state_tokens};\nAction: "`. 所以 `language_embeds` 不能纯文本预算 — 必须用真实 (task + digitized state) 重新 encode. `numerical_compare.py:phase_triton` 第 156-175 行处理。

生产部署时, 每次推理前需重新 generate language_embeds based on current state (~1ms overhead).

### 5. 数值差异 1.42% 在哪

主要来源:
- bf16 vs fp64 (JAX 默认 fp32 内部, 输出 cast 到 fp64 numpy)
- 10 步去噪累积误差 (V1 用 bf16 路径, JAX 用 fp32)
- 22 个手写 kernel 内 fused fp32 accumulator → bf16 store (V1) vs JAX 单 op fp32

部署级精度对比:
| 量级 | 含义 |
|---|---|
| per-dim MAE 0.001-0.01 rad | ~ 0.05-0.5 度, 远小于 Piper 控制精度 (~ 1-2 度) |
| Action chunk 50 步累积漂移 | 假设独立, 总漂移 ~ 0.07 rad ≈ 4 度 — 可控 |

## 下一步 (可选)

### Step 4: 集成到 start_autonomy_from_ckpt.sh

加 `backend` sidecar 字段:
```jsonc
// train_config.json
{
  "base_config_name": "...",
  "override_asset_id": "...",
  "backend": "triton"  // 新增: "jax" (默认) / "pytorch" / "triton"
}
```

启动脚本读 backend 字段, dispatch 到:
- `jax`: 现 `serve_policy.py` (.venv) :8000
- `pytorch`: 阶段 3 实施 (尚未做)
- `triton`: 新 `serve_policy_triton.py` (.venv_5090_trt + V1 pkl) :8002

工程量: 1-2 天.

### Step 5: sim01 真机回归测试

跑 5-10 个完整 task A episode, 对比:
- 推理 wall time (35ms 是否稳定)
- chunk 节奏 (CUDA Graph 抖动 < 0.05 ms 应该让 RTC 极稳)
- 真实任务成功率 (vs JAX baseline)

工程量: 1-2 天.

## 修订历史

| 版本 | 时间 | 内容 |
|---|---|---|
| v0.1 | 2026-05-20 | V1 集成完整完成: 35.4ms 速度 + 1.42% rel error 数值对齐 |
