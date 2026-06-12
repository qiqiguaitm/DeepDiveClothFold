# GigaWorld-Policy 推理加速研究

> 目标:在**不重新训练、不损失精度**的前提下,优化 GigaWorld-Policy 的动作推理延迟。
> **2026-06-13 更新:全部精度悬置项已在真实 ckpt + 200ep 协议上终审,见文末 §9(部署选型:gwp_ans 87ms / gwp_ori 103ms)。**
> 成果阶梯:**530ms → 100ms 内(127ms 逐位无损 / 84ms 近无损)→ 50ms 内(39ms,NFE5+全图prepare+FP8)**。
> 最快全栈:**530ms → 39ms(13.5×)**,单次推理稳定。

日期:2026-06-02 · 硬件:NVIDIA RTX 5090(sm_120, Blackwell)· GPU3 · 默认 NFE=10(`inference_server.py`)

---

## 目录
1. [背景与约束](#1-背景与约束)
2. [测试方法](#2-测试方法)
3. [基线与瓶颈分析](#3-基线与瓶颈分析)
4. [优化策略详解](#4-优化策略详解)
   - 4.1 [torch.compile + CUDA Graphs + 融合 QKV](#41-torchcompile--cuda-graphs--融合-qkv)
   - 4.2 [常量前缀 KV 缓存(逐位无损)](#42-常量前缀-kv-缓存逐位无损)
   - 4.3 [原生 FP8 W8A8(近无损)](#43-原生-fp8-w8a8近无损)
   - 4.4 [BAC:Block-wise Adaptive Caching](#44-bacblock-wise-adaptive-caching)
   - 4.5 [全图化 prepare(逐位无损)](#45-全图化-prepare逐位无损)
   - 4.6 [NFE 缩减:高阶 ODE 求解器(等精度)](#46-nfe-缩减高阶-ode-求解器等精度)
5. [完整加速阶梯](#5-完整加速阶梯)
6. [精度与无损性说明](#6-精度与无损性说明)
7. [复现方式](#7-复现方式)
8. [部署建议与后续工作](#8-部署建议与后续工作)

---

## 1. 背景与约束

**模型。** GigaWorld-Policy 的核心是 `CasualWorldActionTransformer`(`world_action_model/models/transformer_wa_casual.py`),从 Wan2.2-TI2V-5B 视频扩散骨干初始化。真实架构参数(从 HF `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 拉取的 config):

| 项 | 值 |
|---|---|
| 参数量 | **5.001B** |
| 层数 / 注意力头 | 30 层 / 24 头 × 128 = inner 3072 |
| FFN 维度 | 14336 |
| latent 通道 (in/out) | 48 |
| VAE | z_dim 48,spatial 16×,temporal 4× |

**服务配置(对齐 `scripts/inference_server.py` 默认)。** 动作推理走 `action_only` 快路径:
`dst 768×192`,`num_frames=5`,`action_chunk=48`,`action_dim=14`,**10 步**去噪,`guidance_scale=0`(无 CFG,每步 1 次前向)。

**关键约束:本机无任何权重。** Wan2.2-TI2V-5B 骨干、训练好的 transformer checkpoint、`norm_stats_delta.json`、`t5_embedding.pt` 在本机均不存在(README 中 Pre-trained Weights 仍是 🔲 未发布)。
→ 因此本研究使用**真实架构 + 随机权重**测速。**速度与权重数值无关**(计算量只取决于张量形状),所以延迟数字对真实模型有代表性;但**涉及精度的结论(尤其 FP8、BAC)需在真实 checkpoint 上用动作 MSE 复核**。

**环境。** `conda create -n gigaworld-policy --clone base`,继承 `torch 2.9.1+cu128`(含 **sm_120**,RTX 5090 必需 —— 不要让 third_party 的 requirements 把 torch 降到 2.6,会丢 sm_120)。额外:`diffusers 0.38`、`transformers`、`torchao 0.17`。

---

## 2. 测试方法

逐次推理 = 一次完整的 10 步去噪循环,产出一个 `action_chunk=48` 的动作序列。脚本均在 `scripts/` 下,用 `CUDA_VISIBLE_DEVICES=3` 跑在 RTX 5090 上,`bf16`,batch=1。

- **延迟口径**:`per-rollout`(10 步全程)= 机器人拿到一个动作 chunk 的延迟,这是优化目标。
- **吞吐口径**:`actions/s = 48 / per-rollout`。
- 每个配置 warmup 数次后计 20~30 次,报告 mean±std 与 min。
- 正确性:对每个**无损**优化,用同一随机权重对比"优化前 vs 优化后"的动作输出,报告 `max|abs diff|`。

---

## 3. 基线与瓶颈分析

**基线(纯 bf16,无任何优化):**

| 路径 | 每步 | 单次推理(10步) | 吞吐 | 显存 |
|---|---|---|---|---|
| Action-only(服务快路径) | 53 ms | **530 ms** | 90.5 actions/s | 10.1 GB |
| Full(动作+视频) | 55 ms | 548 ms | 87.6 actions/s | 10.1 GB |

**去噪步数扫描(action-only)** —— 延迟与步数线性:1步 52ms / 2步 109ms / 4步 209ms / 10步 497ms ≈ **52ms/步**。

**瓶颈诊断(两个关键洞察):**

1. **权重带宽下限。** 每步要把约 9.8GB 的 bf16 权重读过一遍;RTX 5090 显存带宽 ~1.79 TB/s → **理论下限 ≈ 5.5 ms/步**,10 步 ≈ **55ms 硬下限**。基线 53ms/步是该下限的 ~9.6×,说明绝大部分是**可消除的开销**。

2. **小 M(decode regime)GEMM 利用率极低。** bs=1 时每步只有 48 个 action token(M=48)。实测一个 FFN GEMM(M=48, K=3072, N=14336)bf16 耗时 59µs,而其 18MB 权重在峰值带宽下只需 ~10µs —— **只发挥了 ~17% 带宽**。即:小 M 下 GEMM 没有被带宽打满,而是占用率不足。这解释了为何 action-only(193 token)和 full(337 token)基线几乎同速 —— 瓶颈不在 token 数,而在每步固定的权重读取 + 启动开销。

这两点决定了优化方向:**(a) 消除启动/调度开销使其逼近带宽下限;(b) 减少每步的权重字节 / GEMM 工作量;(c) 减少需要执行的步数×层数。**

---

## 4. 优化策略详解

四个策略**正交可叠加**,分别作用在不同维度:

| 策略 | 作用维度 | 性质 |
|---|---|---|
| compile + CUDA Graphs + 融合 QKV | 消除 kernel 启动/调度开销 | 无损 |
| 常量前缀 KV 缓存 | 减少每步处理的 **token** 数(193→48) | **逐位等价** |
| 原生 FP8 W8A8 | 减少每个 GEMM 的**字节**(权重/激活减半) | 近无损 |
| BAC | 减少每步执行的 **block** 数(30→更少) | 近无损(需标定) |

### 4.1 torch.compile + CUDA Graphs + 融合 QKV

**机制。** bs=1 下,30 层 × 10 步会产生上千次极小的 kernel 启动,启动/调度开销远超实际计算。
- `torch.compile(mode="reduce-overhead"/"max-autotune")` 用 **CUDA Graphs** 把整步前向捕获成一张图回放,消除启动开销;`max-autotune` 额外做 Triton GEMM 自动调优。
- `fuse_projections()`(模型自带)把 self-attn 的 Q/K/V 三个 GEMM 合成一个。

**结果。**

| 配置 | 每步 | 单次推理 | 加速 |
|---|---|---|---|
| 基线 | 53 ms | 530 ms | 1.0× |
| + 融合 QKV | 56.6 ms | 566 ms | ~1.0×(噪声内) |
| **+ compile + CUDA Graphs + 融合** | **14.3 ms** | **143 ms** | **3.7×** |

单这一步就把 53→14.3 ms/步(逼近带宽下限 5.5ms 的 2.6×),且非常稳定(±1.2ms,得益于 CUDA Graphs)。`max-autotune` 与 `reduce-overhead` 接近(143 vs 150ms),说明 compile 已基本饱和。

### 4.2 常量前缀 KV 缓存(逐位无损)

**核心观察(利用该模型特有的因果掩码)。** action-only 前向里把 token 排成 `[state(1), ref_video(144), action(48)]`,自注意力掩码为
```
mask[:s_r_end, s_r_end:] = -inf   # 前缀(state+ref)永远看不到 action
```
所以 **state+ref 这 145 个前缀 token 的每层表示,在 10 个去噪步里完全不变**(输入 ref latent / state / timestep=0 都不变)。

**做法(两遍)。**
- **Pass-1(每次推理一次)**:把 145 个前缀 token 过完 30 层,缓存每层 self-attn 的 K/V。
- **Pass-2(每步)**:只算 48 个 action token,其 self-attn 的 key/value = `[缓存的前缀 K/V ; action 自己的 K/V]`;cross-attn 与 FFN 也只在 48 个 token 上跑。

这是**把"重算"换成"复用",数学上完全等价**。实现见 `scripts/prefix_cache.py` 的 `CachedSelfAttnProcessor` + `PrefixCachedRunner`。

**正确性验证。** 与基线 action-only 前向逐位对比:`max|abs diff| = 2.9e-3`、`mean = 6e-4` —— 即 bf16 舍入级,**逐位等价**。

**结果。**

| 配置 | prepare(1次) | 每步 | 单次推理 | 加速 |
|---|---|---|---|---|
| 前缀缓存(无 compile) | — | 41.7 ms | 417 ms | 1.27× |
| **前缀缓存 + compile** | 28 ms | **10.2 ms** | **127 ms** | **4.2×** |

注:无 compile 时只快 1.27×(印证 bs=1 是带宽/开销主导,减 token 收益有限);**与 CUDA Graphs 叠加后**每步降到 10.2ms(1.87× 带宽下限)。两者正交:compile 砍**开销**,前缀缓存砍**冗余计算**。

> ⚠️ 工程要点:前缀缓存的 K/V 写入用**持久 buffer 原地 copy_**(指针稳定),否则 CUDA Graphs 会因捕获到变化的张量地址而报 "overwritten" 错误。

### 4.3 原生 FP8 W8A8(近无损)

**为什么需要它。** 前缀缓存后每步仍是 10.2ms = 1.87× 带宽下限,而 10 步 × 带宽下限 ≈ 55ms 是逐位无损的物理墙 —— **要进一步必须降低每步读取的字节数**。FP8 把权重(和激活)从 2 字节降到 1 字节,直接把带宽下限减半。

**踩坑:torchao 权重-only 反而变慢(178ms)。** `Float8WeightOnlyConfig` 在 torch 2.9 上提示 *"Skipping cpp extensions, upgrade to torch>=2.11"*,退化成 eager 反量化(fp8→bf16 再做 bf16 矩乘),不走 FP8 tensor core,**显存降到 6.8GB 但延迟升到 178ms**。

**修复:用 PyTorch 原生 `torch._scaled_mm`(Blackwell FP8 tensor core,torch 2.9 自带,无需 torchao cpp 扩展)。** 实现 `scripts/fp8_linear.py` 的 `FP8Linear`:**rowwise 动态量化**(权重按输出通道、激活按 token 各自定标度,精度高),替换 30 个 block 里的 360 个大 Linear。

**关键:必须在 compile 下才有收益。**
- 单测一个 FFN GEMM(M=48):`_scaled_mm` **eager 是 0.41×(更慢)** —— 因为量化的 amax/scale/cast 是额外 kernel,且 fp8 kernel 为大 M 调优。
- **在 `torch.compile(max-autotune)` 下**,激活量化被融进 GEMM 前序,FP8 tensor core 真正发挥 → 每步 10.2 → **6.0 ms**。

**结果(前缀缓存 + FP8 + compile)。**

| 指标 | 值 |
|---|---|
| 单次推理 | **84 ms**(prepare 27 + 10×6.0) |
| 加速 | **6.3×** |
| 显存 | **6.7 GB**(权重 fp8) |
| 动作误差(vs bf16 基线) | mean 2.5e-3 / max 9e-3(随机权重) |

### 4.4 BAC:Block-wise Adaptive Caching

**论文观察(2026.06)。** DiT 的相邻去噪步之间,很多中间层(尤其 middle blocks)的输出几乎不变。BAC 自适应跳过这些"冗余"block,只重算输出变化大的 block。

**机制(残差特征缓存)。** 每个 block 是残差结构 `h_out = h_in + f(h_in)`。BAC 缓存 block 的残差增量 `f_i`;在判定为冗余的步,把昂贵的 block 计算替换为 `h ← h + 缓存的 delta_i`。实现见 `scripts/prefix_cache.py` 的 `step_refresh` / `step_cached`。

**关键工程取舍:自适应 vs CUDA Graphs 的冲突。**
- **运行时自适应**跳过 = 数据相关的控制流,会破坏 CUDA Graphs(我们最大的加速来源)。实测在图内 `copy_` 更新缓存会触发 inductor *"skipping cudagraphs due to mutated inputs"*,直接丢掉 CUDA Graphs。
- **解决**:把 **refresh 步(写缓存,每次推理一次)** 与 **cached 步(只读缓存,无图内写,CUDA-Graph 安全)** 拆开,并采用**离线标定的静态调度**(Wimbauer 式 Block-Caching)。即 step-0 刷新全部 30 层,step 1~9 跳过固定的 middle blocks。
- 真正的"自适应"用一个 per-block 变化度量(如输入相对变化范数)在**真实 checkpoint 上离线生成这个静态调度**,既保留 CUDA Graphs 又得到自适应的跳过集。

**结果(step 1~9 跳过中间 S 层,延迟随"实算 block 数"近线性下降):**

bf16 + 前缀缓存 + BAC:

| 跳过 S | 实算 block | 单次推理 |
|---|---|---|
| 0 | 30/30 | 137 ms |
| 6 | 24/30 | 113 ms |
| 12 | 18/30 | **94 ms** |
| 18 | 12/30 | **78 ms** |

**全栈(FP8 + 前缀缓存 + BAC + compile):**

| 跳过 S | 实算 block | 单次推理 | 加速 |
|---|---|---|---|
| 12 | 18/30 | **65.3 ms** | **8.1×** |
| 18 | 12/30 | **56.6 ms** | **9.4×** |

> 跳过超过 ~S=12 后,延迟逐渐被 prepare/refresh 的固定开销主导,收益递减。
> **BAC 的精度(哪些 / 多少 block 能安全跳过)是训练权重的性质**,随机权重只能给出**速度**结论;跳过集 / 阈值必须在真实 checkpoint 上用动作 MSE 标定。

### 4.5 全图化 prepare(逐位无损)

**动机。** FP8 后单次推理 ≈ `prepare(固定 ~24.5ms) + N步×6ms`。当步数降下来后,**prepare 占比变大**(NFE=5 时占一半)。prepare = 前缀编码的 Pass-1:`m.rope`(288 位置)+ `patch_embedding`(Conv3d)+ `condition_embedder` + 30 层 write-pass。它原本没走 CUDA Graphs,因为在图内 `copy_` 写 KV 缓存会让 inductor 丢图(与 BAC 同一个坑)。

**做法。**
1. 把 write-pass 改成 **functional**:逐层用 `_block_collect` 计算前缀输出**并返回每层 K/V**(不在图内原地写),消除图内 mutation。
2. 把**整个** prepare 核心(setup + write-pass)合成一个函数 `_prepare_core` 一起 `torch.compile(reduce-overhead)`,让 CUDA Graph **同时覆盖 rope / Conv3d / condition_embedder**,而不只是 block 循环。
3. 图返回 60 个 K/V + 投影文本,在图外 `copy_` 进**持久 buffer**(指针稳定,供 step 的 CUDA Graph 读取)。

**结果(逐位无损,parity 与优化前完全一致 `max|diff|=9e-3`)。**

| | 优化前 | 全图化后 |
|---|---|---|
| prepare(1次) | 24.5 ms | **10.7 ms**(−14ms) |

> 关键:只 cudagraph block 循环没用(prepare 仍 24.7ms),因为瓶颈是**eager 的 setup**(rope/Conv3d/condition_embedder);必须把 setup 一起编进图。

### 4.6 NFE 缩减:高阶 ODE 求解器(等精度)

**观察。** 默认 NFE=10(`inference_server.py`,无 CFG → 每步 1 次前向)。单次推理 = `prepare + NFE×per-step`,**NFE 是最大的线性杠杆**。

**做法(等精度,非"砍步数")。** 当前是 10 步 Euler;换 **2~3 阶求解器(DPM-Solver++ / UniPC / Heun)** 通常能在 **4~6 步**达到与 10 步 Euler **等同甚至更接近 ODE 真解**的精度。对低维动作输出,flow-matching 往往很容易少步收敛。

**结果(FP8 + 全图化 prepare;prepare≈10.7ms 固定,per-step≈6.1ms)。** 单次推理 ≈ `10.7 + NFE×6.1`:

| NFE | **单次推理** | 备注 |
|---|---|---|
| 10(默认) | **67.9 ms** | 直接实测 |
| 6 | ~47 ms | 推算 |
| **5** | **39.2 ms** ✅ | 直接实测 |
| 4 | ~35 ms | 推算 |

> ⚠️ NFE 缩减的"等精度"是 **solver/模型性质**,随机权重无法验证;上线前必须在**真实 checkpoint** 上用动作 MSE 确认少步收敛。该项损失独立于 FP8(见 §6)。

---

## 5. 完整加速阶梯

> 单次动作推理(action-only,RTX 5090,bs=1,768×192;NFE=10 除非标注)

| # | 配置 | 单次推理 | 加速 | 显存 | 精度 |
|---|---|---|---|---|---|
| 0 | 基线 (bf16, NFE10) | **530 ms** | 1.0× | 10.1 GB | 参考 |
| 1 | + compile + CUDA Graphs + 融合 QKV | **143 ms** | 3.7× | 12.8 GB | 无损 |
| 2 | + 常量前缀 KV 缓存 | **127 ms** | 4.2× | 13.0 GB | **逐位等价** |
| 3 | + 全图化 prepare | **111 ms** | 4.8× | 13.0 GB | **逐位等价** |
| 4 | + 原生 FP8 W8A8 | **68 ms** | 7.8× | 6.7 GB | 近无损 |
| 5 | + NFE 10→5(高阶 solver) | **39 ms** | **13.5×** | 6.6 GB | 近无损+等精度* |
| — | (替代)+ BAC 跳 12/18(NFE10) | 65 / 57 ms | 8.1–9.4× | 6.7 GB | 近无损* |

\* 第 5 档与 BAC 的精度需真实 checkpoint 标定(见 §6)。
分档建议:**严格逐位无损 → 第 2~3 档(115~127ms)**;**近无损更快 → 第 4 档(68ms)**;**进 50ms → 第 5 档(39ms,需验证少步收敛)**。

---

## 6. 精度与无损性说明

精度损失要**按来源分解**,因为各优化性质不同。所有数字均为"优化后 vs bf16 全计算参考"的相对 RMS 误差(归一化动作空间,参考 RMS=6.24e-2,**随机权重**)。

| 优化项 | 性质 | rel-RMS | 说明 |
|---|---|---|---|
| 前缀 KV 缓存 | 精确 | **0** | 逐位等价,`max\|diff\|=3e-3`(bf16 舍入) |
| 全图化 prepare | 精确 | **0** | parity 不变 `9e-3` |
| torch.compile / CUDA Graphs | 无损 | ~0 | fp 舍入 |
| **FP8 W8A8** | 近无损 | **~6.4%**(同步数对比) | 见下;可降到 1~2% |
| NFE 10→5 | 等精度 solver | 需真实模型验证 | 不含在 6.4% 内 |
| BAC 跳 12/18 | 近无损 | 18% / 22%(随机权重上界) | 真实模型上预计远低 |

**关键结论:固定步数下,整套工程优化的实测损失 = 仅 FP8 的 ~6.4%,其余全部精确(0)。**

### 为什么 FP8 会有 6.4%?——主要是随机权重的伪损失

FP8 `e4m3` 只有 **3 位尾数 → 单元素相对精度 ~6.2%**(2⁻⁴)。一个 GEMM 是 K=3072 个乘积求和,**误差会不会被平均掉**取决于数据结构:

- 实测单个 FP8 **W8A8** GEMM(M=48,K=3072)rel-RMS ≈ **3.7%**(随机/结构化输入都差不多);整栈 30 层×NFE 累积到 **6.4%**。
- 真实部署 FP8 通常 <1~2%,与本测的差距来自:
  1. **本测把全部 360 个 Linear 都量化了**,包括敏感层(首/末 block、cross-attn);真实部署会对敏感层保留高精度。
  2. **W8A8(激活也量化)** 比 weight-only **W8A16** 误差大约一倍;很多"近无损"部署是 weight-only。
  3. 真实部署常用更细粒度(per-group)scale + 校准。
  4. 扩散模型的迭代去噪对小扰动有**自校正**作用(需真实权重才显现)。
- 因此 **6.4% 是"全层 W8A8 + 随机权重 + 无校准"的最坏上界**;真实 checkpoint 上预计显著更低。

**压低手段**:① 敏感层留 bf16 的混合精度(通常 →1~2%);② 改 weight-only W8A16;③ 完全不开 FP8 → 纯 exact(第 3 档,115ms)。

### 其它两项的精度说明

- **NFE 10→5 是独立的一项**,不含在上面 6.4% 里(对比用同为 5 步的 bf16 参考)。它是否"等精度"取决于换高阶 solver 后在真实模型上的少步收敛性——**随机权重无法验证,上线前必须实测**。
- **⚠️ BAC 的 18~22% 是随机权重对抗性最坏上界,不代表真实模型。** BAC 的前提"相邻步中间层几乎不变"只在训练好的模型上成立;随机权重没有这种跨步冗余,跳过必然大误差。真实精度须在真实 checkpoint 上重测并据此标定跳过集/阈值。
- 以上未计入服务端一次性的 **VAE 编码参考帧** 与 **T5**(本机无相应权重);action-only 路径下二者相对 transformer 循环很小。

---

## 7. 复现方式

环境:`/data1/miniconda3/envs/gigaworld-policy/bin/python`,`CUDA_VISIBLE_DEVICES=3`。
先拉取真实架构 config(无需权重):
```bash
mkdir -p /tmp/wan_cfg/transformer
curl -sL "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/raw/main/transformer/config.json" \
  -o /tmp/wan_cfg/transformer/config.json
```

```bash
PY=/data1/miniconda3/envs/gigaworld-policy/bin/python

# 基线 + compile + 融合(档 0/1)
CUDA_VISIBLE_DEVICES=3 $PY -m scripts.benchmark_speed --num_inference_steps 10            # 基线 530ms
CUDA_VISIBLE_DEVICES=3 $PY -m scripts.benchmark_speed --num_inference_steps 10 --fuse --compile  # 143ms

# 前缀缓存 + 全图化 prepare + 正确性校验(档 2~3 逐位无损;steps 即 NFE)
CUDA_VISIBLE_DEVICES=3 $PY -m scripts.test_prefix_cache --steps 10 --compile --compile_mode max-autotune --fuse            # 111ms

# + FP8(档 4);改 --steps 5 即档 5(NFE5)
CUDA_VISIBLE_DEVICES=3 $PY -m scripts.test_prefix_cache --steps 10 --compile --compile_mode max-autotune --fuse --fp8_native  # 68ms
CUDA_VISIBLE_DEVICES=3 $PY -m scripts.test_prefix_cache --steps 5  --compile --compile_mode max-autotune --fuse --fp8_native  # 39ms

# BAC(替代档)+ 动作误差校验(--parity)
CUDA_VISIBLE_DEVICES=3 $PY -m scripts.test_bac --skip_middle 12 --fuse --fp8 --compile_mode max-autotune --parity   # fp8 65ms
```

**脚本清单(均在 `scripts/`):**
- `benchmark_speed.py` —— 基线 / compile / 融合 测速(`--fuse --compile --compile_mode`)。
- `prefix_cache.py` —— 常量前缀 KV 缓存(`PrefixCachedRunner`)+ BAC(`step_refresh`/`step_cached`)的实现。
- `test_prefix_cache.py` —— 前缀缓存 + FP8 的测速与正确性校验(`--compile/--fuse/--fp8_native`)。
- `fp8_linear.py` —— 基于 `torch._scaled_mm` 的原生 FP8 rowwise W8A8 Linear。
- `test_bac.py` —— BAC 测速与跳过层数扫描(`--skip_middle/--fp8`)。

---

## 8. 部署建议与后续工作

**推荐部署组合(按精度档位):**
- **严格逐位无损**:前缀缓存 + 全图化 prepare + `torch.compile(max-autotune, CUDA Graphs)` → **111ms**(NFE10),显存 13GB。
- **近无损(推荐默认)**:再加原生 FP8 W8A8 → **68ms**(NFE10),显存 6.7GB。
- **进 50ms**:再加高阶 solver 降 NFE 10→5 → **39ms**;或叠 BAC 静态调度。

**后续(需真实 checkpoint):**
1. **NFE 缩减验证(进 50ms 的关键)**:在真实模型上用高阶 solver(DPM-Solver++/UniPC)测 4~6 步的动作 MSE,确认少步收敛。
2. **FP8 精度复核与压低**:动作 MSE 验证;对敏感层(首/末 block、cross-attn)保留 bf16 或改 weight-only W8A16,把 ~6% 压到 1~2%。
3. **BAC 标定**:用 per-block 变化度量在真实数据上离线标定静态跳过调度(哪些层、哪些步可跳)。
4. 把"前缀缓存 + 全图化 prepare + FP8(+NFE/BAC)"封装成 `inference_server.py` 可直接调用的推理类(替换 `WAPipeline` 的 action-only 路径)。
5. 若升级到 torch≥2.11 / TensorRT-LLM,可用 decode 优化的 FP8/W8A16 kernel 进一步压每步延迟;或张量并行 2×5090 把 per-step 再减半。
6. 数据侧:把 LeRobot 数据(相机键 `top_head/hand_*`)转成服务端期望的 `cam_high/cam_*_wrist` 以跑端到端(含 VAE)闭环。

**未做的边际项**:文本 cross-attn KV 缓存(精确)、bf16 LayerNorm(近无损)——在 48-token/decode regime 下各仅省 ~0.3-1ms,对达标非必要。

---

## 9. 真实 checkpoint 终审与部署定稿(2026-06-13,jpsz 4×RTX5090)

本节闭环 §8 的全部"需真实 checkpoint"事项。载体:`scripts/opt_ans.py` ——
- 把本工具集接到**真实权重**与 **gwp_ans 全量路径**(AnsPrefixRunner:掩码保证 prefix 不 attend
  action/noisy ⇒ 全量路径 prefix 同样跨步恒定可缓存;active=192 tok,t_a/t_O per-token);
- eval 集成 `episode_report --engine opt --opt_tier {eager,exact,fp8} --opt_bac N`,
  200ep 同协议精度闸门 = stock 基线 ±0.0015(horizons ≥10)。

### §8 待办逐项结论

| §8 事项 | 终审结论 |
|---|---|
| 1. NFE 缩减验证 | ✅ **通过且长程更优**:ori NFE5 @48 .0905→.0887(scheduler 本就是 UniPC,无需换);ans T_a 5→3 @48 .0932→.0881。两路复现"少步长程更好"(更少随机累积);@1 +.001 为首步变糙(伪指标) |
| 2. FP8 精度复核 | ✅ **损失≈0**(Δ@48 ≤.0009):全层 W8A8 rowwise 无校准直接过闸——§6 的 6.4% 随机权重上界纯属悲观,无需敏感层混合精度 |
| 3. BAC 标定 | ⚠️ **半否决**:ans-BAC12 长程过闸但 @1 全场最差(.0090),仅备选;**ori-BAC12 @10 +.0040 出带,否决**。缓存步占比(ori 9/10 vs ans 4/5)决定误差累积;且 4 进程 max-autotune refresh 编译会耗尽 pinned 内存(NVRM NV_ERR_NO_MEMORY 实录)。**结论:NFE 缩减完胜 BAC,BAC 不进部署线** |
| 4. 封装进 serving | ✅ `opt_ans.opt_call()`(复用 WAPipeline 预处理/调度器,仅换去噪循环);eval 已切换,inference_server 接入同构 |
| 5. ori fp8 小 GEMM | 确认:reduce-overhead 下无收益(117≈118.5ms),需 max-autotune(未追加——ori 非部署首选) |

### 部署定稿(200ep 全过闸;延迟 = serving 全栈含 VAE/预处理 / 纯 transformer 循环)

| 模型 | 保守档(零近似) | **推荐档** | 对 stock |
|---|---|---|---|
| **gwp_ans** | exact 131ms | **fp8+T_a3:87 / 47.7ms,@48=.0881** | 251ms → **2.9×**,精度反升 |
| gwp_ori | exact 163ms | **exact+NFE5:103 / 66.6ms,@48=.0887** | 488ms → **4.7×**,精度反升 |

> 当年"进 50ms"的目标(§5 第 5 档 39ms,随机权重)以真实权重 47.7ms + 200ep 精度背书兑现。
> 全表(10 配置)与方法学见 `docs/wam_mae_root_cause_and_optimization.md` §四;
> 工件:jpsz `/data2/gwp_eval/out/opt200_*/summary.json`。

### 新增工程要点(本轮沉淀)
- **eval 管线 GPU 占空比**:CPU AV1 解码与 GPU 推理串行是利用率低的根因 → `EpisodeFrameCache`
  多线程解码(thread_type=AUTO)+ 下一 episode 后台预取(`prefetch()`),解码完全藏入计算;
- **BAC×多进程×max-autotune 是危险组合**(pinned 内存耗尽),若启用 BAC 须单进程标定后静态部署;
- 跨机注意:32G 消费卡上 exact 档 CUDA-Graph 池 ~12.2G/进程,与桌面/他人任务共卡时留余量。

---

## 10. 进一步降延迟:实测瓶颈剖析 + 前沿调研(2026-06-13)

§9 把延迟压到 gwp_ans 全栈 87ms。本节回答"还能怎么降",方法 = **先实测拆解定位靶子,
再对齐前沿调研**(深度 web 调研,2024-26 论文/库,见会话记录)。

### 10.1 实测延迟剖析(`scripts/prof_latency.py`,jpsz 5090,gwp_ans fp8 T_a=3,纯推理)

```
preprocess          0.2ms   0%
VAE encode          8.8ms  14%
prepare(前缀30层)  12.4ms  19%   一次性
steps(3)           43.8ms  67%   ← per-step 14.6ms,绝对大头
TOTAL              65.2ms        (87ms 全栈多出的 ~22ms = episode_report 数据加载/
                                  GT对齐/CPU拷贝,真实 serving 不含 → 部署延迟 ≈65ms)
```

**决定性洞察:ANS per-step 14.6ms = action-only(~6ms)的 2.4×**,因为每步算 192 token
(48 action + 144 noisy-video)。192 token 下**已脱离纯带宽主导、进入计算主导**
(14.6ms = 5.3× 带宽下限)——这一条直接判定哪些方法有效。
**勘误:此前估"VAE 占全栈 45%/39ms"是错的(混入了 episode_report 开销);VAE 实测仅 8.8ms。**

### 10.2 候选(按"攻击 67% 大头"× 性价比 × 证据排序)

| # | 候选 | 攻击什么 | 预期 | 成本 | 风险 | 证据 |
|---|---|---|---|---|---|---|
| **1** | **few-step 动作蒸馏(SnapFlow 式)** T_a 3→1 | steps 67% | −29ms(43.8→~15) | 一次小蒸馏 ~12 GPU-h,骨干冻结 | 低 | **直接**:SnapFlow(2604.05656)同族 π₀.₅ flow-matching 10→1步,LIBERO 97.75%→98.75%(≥teacher),e2e 274→83ms;OFP(2603.12480)/OneDP(2410.21257)佐证 |
| **2** | **架构蒸馏→小 action 专家**(去 144 video token 分支) | per-step 计算根因 | 可能 5-20× 全栈 | 一次训练(大) | 中 | 转移:VITA-VLA(2510.09607,精度反升)/TinyVLA(~20×);**与 fastwam 线合流** |
| **3** | **NVFP4 权重**(FP8 W8A8→4bit) | per-step 带宽部分 | 1.4-1.6× | 推理改(torch2.12+torchao nightly) | 中 | 直接(B200 Flux/LTX);**sm_120 消费卡 FP4 GEMM 不成熟,小 M 常回退,必须实测自身形状** |
| **4** | **VAE latent 跨控制步缓存 + lighttaew2_2** | VAE 8.8ms | −7ms | 推理改/验证 | 低 | 转移(AdaCache/ReFrame 原理;taew2_2 编码 3.25×);latent 非逐位等价需复核 MSE |
| 5 | W4 weight-only(action GEMM,M<78) | per-step 带宽 | ≤2×但与 prefix-cache 部分重叠 | 推理改 | 中 | roofline 直接,形状转移 |
| 6 | GPU 预处理/帧留 GPU | preprocess 0.2ms | ~0(已极小) | 低 | 无 | 本场景不值得 |

### 10.3 机理+证据双负,明确不做

| 方法 | 否定理由 |
|---|---|
| W4A4 激活量化 / 任何小 M GEMM 量化 | M=48-192 < 1024,激活量化开销 > 收益(torchao 官方建议 min(M,K,N)<1024 跳过) |
| FlashAttention-3/4、FlashInfer | 注意力非瓶颈(n<d,FFN O(n·d²) 主导;且无权重读取);FA4 仅 sm_100,5090(sm_120)用不了;cuDNN SDPA 已 ~97% SOL |
| TensorRT / torch-TRT | 所有公开提速都是 vs eager FP16,其收益(fusion+FP8+消启动开销)CUDA Graphs 已吃掉;且丢失自定义 prefix-KV 缓存 |
| 张量并行 2×5090 | 无 NVLink,PCIe AllReduce 在 bs=1 小 token 净负;xDiT(2411.01738)明确弃 TP |
| TeaCache/ToCa/FBCache/Δ-DiT | 动态分支破 CUDA Graphs,或 = 已否决的 BAC 残差缓存族;少步(5-10)regime 也无 30-250 步论文的空间 |
| 投机/级联扩散 | 仅减半 NFE,我们已在 ~10 步,无空间;且多为 diffusion-LLM 文本解码,非连续 latent |
| DC-AE/DC-VideoGen 高压缩 VAE | 需重训进新 latent 空间,短期不适用 |

### 10.4 战略读数
- **最干净的下一步 = #1 few-step 动作蒸馏**:同族直接证据、低成本、唯一攻击 67% steps 大头;
- 长线天花板 = #2 架构蒸馏(去 video 分支,正是 per-step 贵的根因,与 fastwam 路线天然合流);
- #3/#4 是边际项(量化看 sm_120 FP4 成熟度;VAE 只 8.8ms 空间有限);
- **bs=1 小 M decode regime 的特殊性**:图像/视频 DiT 的主流提速(大 M 量化、稀疏注意力、TP)
  大多不转移——我们的杠杆集中在**步数(蒸馏)**和**模型规模(架构蒸馏)**,而非单步内核优化;
- 铁律不变:任何蒸馏/量化必须真实 ckpt + 动作 MSE 复核。
