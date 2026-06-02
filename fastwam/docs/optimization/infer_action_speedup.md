# FastWAM 动作快路径（action-only）推理加速

> 目标：在 **默认去噪步数** 下，把 `infer_action`（不做 test-time 未来视频想象的动作快路径）单次推理延迟优化到 **≈50 ms**，且 **不重新训练、不损失精度**。
>
> 结论速览：在部署默认的 **4 步** 下，从 **258 ms → 34.4 ms（7.5×）**，已达标；2 步 26.1 ms，8 步 ≈51 ms。优化全部为 **无损系统级**（CUDA Graphs + 算子融合 + max-autotune kernel 选择 + cudnn.benchmark），输出与 eager 一致（仅 bf16 融合核舍入误差，rel≈8e-3）。

---

## 1. 实验环境与方法

| 项 | 值 |
|---|---|
| 机器 / GPU | sim01，单卡 **RTX 5090 32GB**（物理 GPU2，`CUDA_VISIBLE_DEVICES=2`） |
| 框架 | torch **2.7.1+cu128**，conda 环境 `fastwam`（python 3.10），`pip install -e .` |
| 模型 | Wan2.2-TI2V-5B 架构，**RoboTwin profile**（`robotwin_uncond_3cam_384_1e-4`：3 相机拼接 384×320，action_dim=14，proprio=14），与 deepdive_kai0 的双臂数据（`Task_A`，3 相机 + 14 维状态）结构一致 |
| 权重 | **随机初始化、零下载**（绕过会强制下载 Wan VAE 的 loader；video DiT / action DiT / VAE 全部随机初始化，不加载文本编码器，喂 dummy cached context）。推理延迟只取决于张量形状、与权重数值无关，因此随机权重的计时与训练权重一致 |
| 输入数据 | 取自 `deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val` 的真实一帧 + 14 维状态（按部署方式拼成 384×320） |
| 精度 | bf16 |
| 计时 | 每组 `warmup` 后 `torch.cuda.synchronize()` 包夹，30 次取均值 |

被测路径：`FastWAM.infer_action`，忠实复刻部署代码
`experiments/robotwin/fastwam_policy/deploy_policy.py` 的调用（input_image `[1,3,384,320]`∈[-1,1]、proprio `[1,14]`、`action_horizon=32`）。

---

## 2. 基线性能（eager）

| 去噪步数 | 平均延迟 | 吞吐 | 有效控制率（replan=4） |
|---|---|---|---|
| 2 | 157.8 ms | 6.34 chunks/s | 25.3 Hz |
| 4 | 258.5 ms | 3.87 chunks/s | 15.5 Hz |
| 8 | 460.4 ms | 2.17 chunks/s | 8.7 Hz |
| 10 | ~504 ms | — | — |

峰值显存 **12.71 GiB**（这正是 GPU2 仅 ~12 GB 空闲时 OOM 的原因）。

### 2.1 分阶段 profile（eager，10 步）

| 阶段 | 单次耗时 | 调用次数/chunk | 占比 |
|---|---|---|---|
| VAE encode | 8.77 ms | 1 | — |
| video pre_dit | 0.84 ms | 1 | — |
| **prefill KV cache** | **50.09 ms** | 1 | 大 |
| **per denoise step** | **44.33 ms** | ×steps | 主导 |
| scheduler step | 0.08 ms | ×steps | 可忽略 |

`video_seq_len=120`、`action_tokens=32`。

---

## 3. 瓶颈诊断：**kernel-launch / Python 开销绑定**

仅 **120 个视频 token + 32 个动作 token** 却要 44–50 ms/层堆——计算量极小，时间几乎全花在 **大量小 kernel 的启动开销和 Python 调度** 上，而非 FLOPs。判据：

- attention 已是 `F.scaled_dot_product_attention`（SDPA），并非朴素 softmax；
- `robotwin_uncond` 设 `mot_checkpoint_mixed_attn=false`，且 gradient checkpointing 由 `self.training` 门控 → 推理期本就不触发，无重算开销；
- 30 层、序列长仅 ~150 的前向耗时数十毫秒，只能是 launch-bound。

**launch-bound 的正解是 CUDA Graphs**（把固定形状的 kernel 序列录制后重放，几乎消除启动开销），且 **完全无损**（重放的是同一串 kernel）。

---

## 4. 优化方案（逐步）

### 4.1 把 CPU 常驻张量搬到 GPU（解锁 CUDA Graphs 的前置条件）

首次 `torch.compile(mode="reduce-overhead")` 时 CUDA Graphs 被 **跳过**，日志报 “skipping cudagraphs due to cpu device”。根因是热路径里有 **CPU 常驻、非 buffer 的张量**，每次调用都触发 CPU→GPU 拷贝：

| 张量 | 位置 | 现象 |
|---|---|---|
| `vae.scale / mean / std` | `wan_video_vae.py` `__init__`（普通属性，非 buffer，`module.to(device)` 漏掉） | `scale=[s.to(device) ...]` 每次拷贝 |
| `action_expert.freqs`（RoPE，complex64） | `action_dit.py:99` | `self.freqs[:L].to(tokens.device)` 每次拷贝 |
| `video_expert.freqs`（3D RoPE，tuple×3） | `wan_video_dit.py:386` | 同上 |

修复（推理期一次性搬运，**不改库代码**，在构建脚本里 monkey-patch 实例）：

```python
model.action_expert.freqs = model.action_expert.freqs.to(device)
model.video_expert.freqs  = tuple(f.to(device) for f in model.video_expert.freqs)
for obj in (model.vae, getattr(model.vae, "model", None)):
    if obj is None: continue
    if hasattr(obj, "mean"):  obj.mean  = obj.mean.to(device)
    if hasattr(obj, "std"):   obj.std   = obj.std.to(device)
    if isinstance(getattr(obj, "scale", None), (list, tuple)):
        obj.scale = [s.to(device) for s in obj.scale]
```

> 这是上游可改进点：把这些张量改为 `register_buffer`，`model.to(device)` 即可自动搬运，CUDA Graphs 开箱即用。

### 4.2 torch.compile 算子融合（reduce-overhead）

对热函数 `VAE encode / prefill / per-step` 施加 `torch.compile(mode="reduce-overhead")`（底层走 CUDA Graphs）。

- **仅融合、CUDA Graphs 仍被跳过时**（即 4.1 未做）：2/4/8/10 步 = 53.0 / 84.4 / 151.5 / 178.7 ms，约 **2.7–2.9×**（纯 inductor 融合收益）。

### 4.3 启用 CUDA Graphs + KV cache 克隆（关键收益）

做完 4.1 后再 compile，CUDA Graphs 生效，但出现经典 **跨图别名错误**：
`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`——prefill 产出的 **KV cache 落在 cudagraph 内存池**，被随后的 step 图读取时已被覆写。

修复：**把 KV cache 从托管内存池克隆出来**，并把上游 cudagraph 输出（VAE 的 first-frame latent）在被 eager 段消费前 `.clone()`：

```python
kv = [{k: v.clone() for k, v in d.items()} for d in kv]   # prefill 后立即克隆
first = enc_fn(input_image=img, tiled=False).clone()       # VAE 输出克隆
```

收益（搬 buffer + 全阶段 CUDA Graphs）：

| 步数 | eager | 优化后 | 加速 | 控制率(replan=4) |
|---|---|---|---|---|
| 2 | 159 ms | **29.7 ms** | 5.4× | 135 Hz |
| **4（部署默认）** | 258 ms | **37.8 ms** | **6.8×** | 106 Hz |
| 8 | 450 ms | 54.9 ms | 8.2× | 73 Hz |
| 10 | 614 ms | 64.1 ms | 9.6× | 62 Hz |

### 4.4 单图全融合实验（未带来额外收益）

尝试把 `VAE→prefill→N 步去噪` 整体编进 **单个** `torch.compile` 图（步数与 token 长度作为编译期常量、注意力 mask 与 schedule 预计算缓存）：
2/4/8/10 = 29.6 / 37.8 / 78.9 / 64.3 ms。与 4.3 持平甚至更差（8 步出现图捕获抖动），**说明残余开销不在“图与图之间的 Python 胶水”**，而在阶段本身的真实计算。脚本保留为 `benchmark_infer_action_fused.py` 备查。

### 4.5 max-autotune kernel 选择 + cudnn.benchmark（最后的无损榨取）

在 4.3 基础上，把 `torch.compile` 模式从 `reduce-overhead` 换成 **`max-autotune`**（更优 matmul/epilogue kernel + CUDA Graphs），并开启 `torch.backends.cudnn.benchmark=True`（自动选最快卷积算法，利好 VAE）。

**优化后分阶段实测（GPU3 / RTX 5090，单 stage 单独计时）：**

| 阶段 | eager | reduce-overhead | **max-autotune** | 说明 |
|---|---|---|---|---|
| VAE encode | 8.77 ms | 4.91 ms | **4.80 ms** | 真实卷积计算，地板 |
| **prefill KV cache** | 50.09 ms | 16.06 ms | **13.54 ms** | 30 层、hidden=3072 视频专家，**新瓶颈** |
| per denoise step | 44.33 ms | 5.06 ms | **4.75 ms** | 30 层动作专家，kernel-count 受限 |
| 估算 4 步合计 | ~258 ms | 41.2 ms | **37.3 ms** | |
| 估算 10 步合计 | ~504 ms | 71.6 ms | **65.8 ms** | |

`max-autotune` 主要收益在 **prefill（16.1→13.5 ms）** 与每步（5.06→4.75 ms）；编译耗时显著更长（多 kernel 变体搜索），属一次性成本。

**端到端实测（max-autotune + cudnn.benchmark，含精度校验，GPU3）：**

| 步数 | eager | reduce-overhead | **max-autotune（最终）** | 加速* | ≤50 ms |
|---|---|---|---|---|---|
| 2 | ~158 ms | 29.7 ms | **26.1 ms** | 6.0× | ✅ |
| **4（部署默认）** | ~258 ms | 37.8 ms | **34.4 ms** | **7.5×** | ✅ |
| 8 | ~460 ms | 54.9 ms | **51.0 ms** | 9.0× | ≈50 |
| 10 | ~504 ms | 64.1 ms | **59.3 ms** | 8.5× | — |

\* 加速相对 GPU2 eager 基线。精度校验：`max_abs_diff≈3.1e-2, mean≈3.6e-3, rel_max≈7.8e-3`（bf16 融合核舍入级，无损）。峰值显存 12.81 GiB。

> 关键转变：经过 4.1–4.5，**瓶颈已从“launch 开销”转为“真实计算”**——VAE(4.8) + prefill(13.5) 构成 ~18 ms 一次性地板，且每步 ~4.75 ms 已是 30 层小张量的 kernel 数量下限。继续无损压缩的空间已很小。

---

## 5. 精度验证（无损）

同一输入、同一固定初始噪声、同一 schedule 下，对比优化 vs eager 的动作输出：

```
max_abs_diff ≈ 1.9e-2,  mean_abs_diff ≈ 3.9e-3,  rel_max ≈ 5.5e-3
```

差异量级即 **bf16 融合核的舍入误差**（同一算法、同一步数，不同 kernel 的归约顺序）。**无重训练、无减步、无量化 → 视为无损。**

---

## 6. 是否还有优化空间？

**已接近无损极限。** 经过 4.1–4.5，瓶颈从 launch 开销转为 **真实计算**：
- 一次性地板 ≈18 ms = VAE encode(4.8) + prefill(13.5)；
- 每步 ≈4.75 ms（30 层动作专家的 kernel 数量下限，max-autotune 后）。

仍属无损、但收益已很小 / 待验证：
- **flash 友好的结构化 mask**：当前 SDPA 用显式 bool mask（`first_frame_causal`），可能落到 math 后端；若改写成 causal/block-diagonal 让其走 flash kernel，可能小幅降低 prefill 与每步的注意力耗时。需谨慎验证数值（flash 与 math 归约顺序不同，仍属 bf16 量级误差）。
- **prefill 是当前最大单项（13.5 ms）**，但它是 30 层、hidden=3072 视频专家的真实前向（FFN 主导，~630 GFLOP），已编译+autotune，进一步压缩需手写融合 kernel，工程量大、收益有限。
- 编译 `video pre_dit`（0.8 ms，收益可忽略）。

会**损失精度 / 被本任务约束排除**的（仅记录，不采用）：
- 减少去噪步数（最直接，但改变输出）；
- fp8 / int8 量化；
- 降低 VAE 输入分辨率 / 更小的 VAE / VAE 蒸馏；
- 步间缓存 / 一致性蒸馏（DeepCache、Consistency 类）。

> **结论**：在“不重训练、不损精度”约束下，**已基本触底**。最终（max-autotune + cudnn.benchmark）部署默认 **4 步 → 34.4 ms 达标（7.5×）**，2 步 26.1 ms，**8 步 51.0 ms 已贴近 50 ms**；10 步 59.3 ms 受 VAE+prefill 的真实计算地板限制，难再无损压到 50 ms 以下。要进一步必须放开精度约束（减步 / 量化 / 蒸馏）。

---

## 7. 复现

```bash
conda activate fastwam
cd FastWAM

# 1) 最终推荐：max-autotune + CUDA Graphs + cudnn.benchmark（最快，含精度校验）
PYTHONPATH=scripts python scripts/benchmark_infer_action_opt.py \
    --gpu 2 --mode max-autotune --num-inference-steps 2 4 8 10 --iters 30 --warmup 8
#   （首次 max-autotune 编译较慢；想快速验证可用默认 --mode reduce-overhead）

# 2) eager 分阶段 profile
PYTHONPATH=scripts python scripts/profile_infer_action.py --gpu 2 --num-inference-steps 10

# 3) 优化后分阶段 profile / compile 模式对比（reduce-overhead vs max-autotune）
PYTHONPATH=scripts python scripts/profile_infer_action_opt.py --gpu 2

# 4) 纯随机初始化、零下载的基线 benchmark
python scripts/benchmark_infer_action.py --gpu 2 --num-inference-steps 2 4 8
```

相关脚本（均在 `scripts/`）：

| 脚本 | 用途 |
|---|---|
| `benchmark_infer_action.py` | 基线 benchmark；默认 **随机初始化、零下载**；用 kai0 真实帧 |
| `profile_infer_action.py` | eager 分阶段计时 |
| **`benchmark_infer_action_opt.py`** | **推荐**：CUDA Graphs + 算子融合 + cudnn.benchmark；`--mode max-autotune` 为最快（含 eager 对比与精度校验） |
| `benchmark_infer_action_fused.py` | 单图全融合实验（未优于 opt，备查） |
| `profile_infer_action_opt.py` | 优化后分阶段 + reduce-overhead vs max-autotune 模式对比 |

---

## 8. 注意事项

- **显存**：模型峰值 **~12.7 GiB**。GPU2 与他人的 docker grasp 服务栈（`graspgen/da3/dinox/tracker-worker` 等，镜像 `grasp/forge`、`grasp/sif`）共享，常驻 ~19 GB；这些容器 `restart=unless-stopped` 且会被自动拉起，跑本测试需先 `docker stop` 释放显存（或改用空闲卡）。
- **默认步数**：仓库内 `eval_num_inference_steps=10`（`configs/train.yaml`），`infer_action()` 签名默认 20，RoboTwin 评测示例与本 benchmark 取 **4**。本文“默认”按 **部署默认 4 步** 计。
- 计时与权重数值无关，故随机初始化用于纯速度评测完全有效。
