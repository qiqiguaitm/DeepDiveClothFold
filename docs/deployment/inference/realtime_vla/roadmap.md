# Realtime-VLA 推理优化 — 5 阶段实施路线图 + 真机测试

> 本文档是 `realtime_vla_optimization_analysis.md` (1687 行, 已拆) 拆分后的"路线图层"。包含 5 阶段实施路线 + 真机测试方案。
>
> **同 series 文档**: `strategy.md` (战略与决策) / `v1_triton_log.md` (V1 已实施日志) / `layer_b_plan.md` (Layer B 未来 plan)

---

## 3. 实施路线图 — 5 阶段

### 3.1 总体阶段图

> **当前推理基线 (2026-05-20, v0.11)**: V1 Triton 路径已落地, P50 = **32.05 ms** (8.00× vs eager, 详见 §6). 阶段 3.2 工程量从原计划"自写 PyTorch+Triton (1-2 周)"实际收敛到 3 天 (V1 复用 + sentencepiece adapter + 5090 重 autotune). 阶段 3 剩余主要是: §3.4.1 PyTorch 训练等效性 POC (1-2 周), §3.4.3 sidecar framework 字段 (1 天).

```
阶段 0 (已完成):
  ✅ venv_5090 / .venv_5090_trt   ← sm_120 兼容
  ✅ benchmark_pi05_inference.py  ← PyTorch 5-backend baseline (E=41ms)
  ✅ V1 Triton 集成               ← P50=32.05ms (§6)
       │
       ↓
阶段 1 (短期, 1 周, 并行可做):
  ⏳ #8 延迟标定        ← 1-2 个 autonomy session
  ⏳ #6 JAX 浅层优化     ← AOT/cache/bf16, 现有 serve_policy 拉满
  ⏳ 真机测试 2         ← Piper t_motion (决定阶段 5 #7 是否上)
  ⏳ inference_rate 调参 ← 3Hz → 8-10Hz, 看 RTC 是否更稳
       │
       ↓
阶段 2 (任务速度主线, 1-2 周):
  ⏳ #5 时间轴 QP       ← 任务速度 1.5-2× (Q5 主要诉求)
       │
       ↓
阶段 3 (选项 X 落地, 剩余 1-2 周):
  ⏳ 3.1 PyTorch 训练等效性 POC (1-2 周)
  ✅ 3.2 推理 serve (V1 路径已完成, P50=32ms)  — 详 §6
  ⏳ 3.3 sidecar framework 字段 (1 天)
  ⏳ 3.4 新 fine-tune 切 PyTorch (持续)
       │
       ↓
阶段 4 (任务质量, 3-6 周):
  ⏳ #4 速度自适应      ← 油门数采 + 速度回归 head
       │
       ↓
阶段 5 (可选, 已降级):
  ⏳ #3 Flash 推测推理   ← 研究性: baseline 32ms × 3× = ~11ms, 边际有限
  ⏳ #7 客户端 MPC       ← 仅 真机测试 2 测出 t_motion > 50ms 时上
```

### 3.1.1 已知前置工程清单

> v0.11 后状态: 走 V1 路径无需修 PI0Pytorch (因为不再用 PI0Pytorch 跑 inference, V1 直接 dict-of-tensors). 下表保留作为"备选路径 max-autotune 实施前置"参考.

阶段 3.2 备选路径 (PyTorch + max-autotune) 的 model code fix:

| # | 文件:行 | 现状 | 需改成 | 影响 |
|:---:|---|---|---|---|
| 1 | `pi0_pytorch.py:172` | `sample_noise` 硬编码 `dtype=torch.float32` | `dtype=next(self.parameters()).dtype` | sample_actions bf16 闭环 |
| 2 | `pi0_pytorch.py:402,405` | `sample_actions` 的 `dt`/`time` 硬编码 fp32 | 跟随 model dtype | bf16 denoising loop |
| 3 | `pi0_pytorch.py:461` | `denoise_step` 返回 `action_out_proj(suffix_out)` 但 suffix_out 是 fp32 | 加 `suffix_out = suffix_out.to(model.dtype)` cast | 修 fp32/bf16 mismatch |
| 4 | `pi0_pytorch._prepare_attention_masks_4d` | `torch.where(mask, 0.0, -inf)` 输出 fp32 | 用 `torch.tensor(..., dtype=model.dtype)` 包 literal | Inductor SDPA 严格 dtype 检查 |
| 5 | `pi0_pytorch.py:229 embed_prefix` | `torch.tensor(att_masks_list, ...)` host→device copy | 预分配 GPU tensor | manual CUDA Graph (backend C) 需要 |
| 6 | `preprocessing_pytorch.py:160` | `class SimpleProcessedObservation:` nested class | 提到模块级别 | torch.compile fullgraph=True 才需要 |

工时估算: 1-3 天 (备选路径才需要做).

### 3.1.2 子任务清单 (阶段 0-1)

| 子任务 | 阶段 | 状态 |
|---|:---:|---|
| ✅ kai0/.venv_5090 / .venv_5090_trt (PyTorch nightly cu128) | 0 | 完成 |
| ✅ benchmark_pi05_inference.py (5-backend) | 0 | 完成 |
| ✅ 真实 JAX ckpt → V1 pickle 转换 (`convert_kai0_to_v1.py`) | 3.2 | 完成 |
| ✅ V1 Triton 集成 + 5090 重 autotune (5 kernel) | 3.2 | 完成 (P50=32.05ms, 见 §6) |
| ⏳ ROS2 推理节点 latency 实测 (#8 标定) | 1 | 待做 |
| ⏳ Piper t_motion chirp 测试 | 1 | 待做 |
| ⏳ inference_rate 3→10 Hz 调参 + RTC 行为观察 | 1 | 待做 |
| ⏳ sidecar framework 字段 + 启动脚本分发 | 3.3 | 待做 |
| ⏳ V1 推理路径包装为 serve_policy_v1.py + WebSocket | 3.2 后 | 待做 |

### 3.2 阶段 1 — 短期热身 (1 周)

#### #8 延迟标定 + 感知对齐

**目标**: `latency_k=8` 从经验值改为数据驱动值, RTC chunk 边界抖动 -30-50%

**改动文件**:
- `ros2_ws/src/piper/scripts/policy_inference_node.py::_get_synced_frame` — 接收图像时用 `header.stamp` 反查 joint deque, 而非"图像到达时"的关节
- 新增 `start_scripts/kai/diag/measure_latencies.py` — 一次性 autonomy session 标定脚本

**步骤**:
1. 跑一次完整 autonomy 周期, 记录:
   - 相机 ROS timestamp vs `header.stamp` (D435/D405 曝光延迟)
   - 关节 state ROS timestamp vs CAN 接收时刻
   - `_publish_action` 发出时刻 vs Piper `/joint_state` 回环响应
2. 落表: `t_camera`, `t_readout`, `t_proprio`, `t_motion` (期望与 V2 数量级一致: 50/33/50/150 ms)
3. 改 `_get_synced_frame`: 用 timestamp 从 joint state deque 反查"曝光那一刻"的关节
4. `latency_k = round((t_camera + t_readout + t_motion) × publish_rate)`, 推测 ~7-10

**验证**: 重启 autonomy, 看 chunk 边界 RMS 抖动 (`StreamActionBuffer` 切换时刻的关节速度跳变) 是否下降。

#### #6 JAX 浅层优化

**目标**: 把现有 JAX serve_policy 推理时间压到该栈下最低 (期望 333ms → 150-200ms)

**改动文件**: `kai0/scripts/serve_policy.py`

**4 项改动**:
- (a) **AOT compile**: 服务启动时 `jit(fn).lower(args).compile()` 预编译, 不靠首次 trace 触发编译
- (b) **持久化 XLA cache 命中确认**: 已有 `start_server_xla_cache.sh`, 但需确认每次重启服务都命中 (常见坑: cache 路径变 / JAX 版本变 / device 列表变都会 invalidate)
- (c) **局部 bf16/fp16**: pi05 base 已 bf16, 但 normalization apply / tokenizer embedding lookup / image preprocessing 多半还在 fp32, 局部转 bf16 省 10-15%
- (d) **VLM prefill 与 AE 第 1 步并发**: 论文 V1 §5.2 数据显示 stream overlap 提升约 3.7%; JAX 下拆两个 jit 函数 + multi-stream

**验证**: WebSocket 服务 stress test, 推理 RTT P50/P95/P99 分布对比改动前后。

#### 真机测试 1 + 2
见 §4。期望产出:
- 测试 1: 模型实际推理时间 P50/P95/P99 分布表
- 测试 2: Piper t_motion 数值, 决定阶段 5 #7 是否做

### 3.3 阶段 2 — 任务速度主线 (1-2 周)

#### #5 时间轴 QP 重参数化 (V2 §4.3.1 + `realtime-vla-v2/server/optimizer.py`)

**目标**: chunk 几何不变, 重分配 Δt_i, 任务耗时 1.5-2×

**改动文件**:
- `kai0/src/openpi/serving/websocket_policy_server.py` — 推理后处理添加 OSQP 调用
- 新增 `kai0/src/openpi/policies/timeaxis_optimizer.py` — 复用 `realtime-vla-v2/server/optimizer.py::TimeParameterizationMPC`
- `ros2_ws/src/piper/scripts/policy_inference_node.py::_publish_action` — 定时器从 `1/publish_rate` 改为 `dt[k]` 驱动

**步骤**:
1. 集成 OSQP 到服务端推理后处理, V2 cloth config 参考参数:
   ```yaml
   dt_ref: 0.016    # 62.5 Hz 参考
   dt_min: 0.008    # 125 Hz 上限
   dt_max: 0.025    # 40 Hz 下限
   lambda_acc: 10.0
   horizon: 50
   v_max: <Piper 关节最大速度>  # 防 QP 违反硬件
   ```
2. 输出协议扩展: `actions: (50, 14)` → `actions: (50, 14) + dt: (50,)`
3. 客户端定时器改造: 从均匀 `1/publish_rate` 改为按 `dt[k]` 累积时间驱动 (或用插值)
4. `latency_k` 单位从"步数"改为"时间 (s)", 内部反查丢哪几步 — **本阶段主要工程量**

**与 RTC 兼容性**: 正交 — RTC 在 chunk index 维度引导, QP 在时间维度拉伸。

**风险**: lambda_acc 先取 10 (V2 cloth 默认) 防过激加速; v_max 需查 Piper 规格书。

**验证**: 折叠任务 (cloth) 实测任务耗时下降比例。

### 3.4 阶段 3 — 选项 X 落地

> **v0.11 实施现状**: §3.4.2 推理 serve 已通过 V1 Triton 路径完成 (P50=32.05 ms, 见 §6), 比原计划"自写 PyTorch+Triton"工程量小一个数量级. 剩余: §3.4.1 PyTorch 训练等效性 POC (1-2 周), §3.4.3 sidecar framework 字段 (1 天).

依赖阶段 1 真机测试 1 结果决定优先级:
- 若模型 P50 < 100ms (推理已很快) → 阶段 3 优先级降低, 可推迟
- 若模型 P50 200ms+ → 阶段 3 拿满 5-10× 才能凸显价值

#### 3.4.1 PyTorch 训练等效性 POC (1-2 周)

**目标**: 验证 `train_pytorch.py` 跑 pi05 fine-tune 与 JAX `train.py` 在 deepdive_kai0 数据集 + DDP 集群上等效。

**步骤**:
1. 选一个已完成 JAX 训练的对照 ckpt (例: `task_a_new_pure_1800_mixed1_step49999`), 记下 config + 数据范围 + 最终 inline-eval MAE
2. 用同一 config 在 PyTorch 跑 (gf0 / uc02 / uc03 任一 8× 节点):
   ```bash
   torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
       <same_config_name> --exp_name=pytorch_parity_test --save_interval 5000
   ```
3. 关键对比指标 (落表):
   | 指标 | 期望 |
   |---|---|
   | 同 1k step 的 loss 曲线对齐性 | max \|Δloss\| < 5% |
   | inline-eval MAE @ 50k step 差异 | < 10% |
   | 单 step 时间 | PyTorch 应快 10-30% (cuDNN + DDP 优化) |
   | 数值稳定性 | 无 NaN / 梯度爆炸 |
4. 验证 PASS → 进入 3.4.2; FAIL → 退回 fallback Y (§5)

**风险点**:
- pi05 PyTorch 路径在 deepdive_kai0 集群是否 DDP/FSDP 稳定 (advantage 管线用过, 主 fine-tune 没用过)
- norm_stats 加载: PyTorch / JAX 共享 `assets/<asset_id>/norm_stats.json`, dtype 转换可能有 bias
- LoRA 适配 (若用): PyTorch peft vs JAX 自实现的等效性

#### 3.4.2 PyTorch 推理 serve 搭建

> **v0.11 实施状态**: 走 V1 Triton 复用路径 (而非自写), 已完成. **P50 = 32.05 ms**, 详见 §6. 本节保留 5-backend baseline 实测数据 (PyTorch 路径上限) 作为路径背景与决策依据.

**2026-05-19 实测结果 (`optimize/benchmark_pi05_inference.py` on 5090, bf16, 100 iter)**

##### 分位数 (P50 / P95 / P99) 含义

`P50` / `P95` / `P99` 是把 100 次推理耗时**按从小到大排序**后, 取第 50 / 95 / 99 个值。

```
[第1名]  [第2名] ... [第50名] ... [第95名] [第96名] ... [第99名] [第100名]
  最快     ...        ↑中位数         ↑P95               ↑P99      最慢
                       P50          (95%快于它)        (99%快于它)
```

| 指标 | 含义 | 通俗说法 |
|---|---|---|
| **P50** | 50% 的推理 ≤ 这个值 | **中位数** — 典型情况 |
| **P95** | 95% 的推理 ≤ 这个值 | "100 次里 5 次比这个慢" — 较慢情况 |
| **P99** | 99% 的推理 ≤ 这个值 | "100 次里只 1 次比这个慢" — **tail latency** |
| Mean | 算术平均 | 受极端值影响大 |
| Std | 标准差 | 衡量波动程度 |

**为什么不只看 Mean**: 真机控制看 **P99 不是 Mean**。
- 两个 backend Mean = 50ms: X (P50=49ms, P99=52ms) 稳定可预期; Y (P50=30ms, P99=200ms) 偶尔的 200ms 推理会打乱 RTC 节奏
- Y 平均一样但灾难性 — 1 次卡顿 chunk 就错位
- **机器人推理只看 Mean 是错的, P99 才是真实风险指标**

##### Backend 做了什么 (按优化层级)

| Backend | Python 层<br>(dispatcher / autograd / op 查表) | XLA / Inductor 层<br>(kernel fusion + Triton 生成) | CUDA Graph 层<br>(消 launch overhead) | autotune 深度 | 一句话定义 |
|:---:|:---:|:---:|:---:|:---:|---|
| **A** eager | ❌ Python 每个 op 都走 dispatcher (1000+ kernel 每个 ~5-10μs Python 开销) | ❌ 无 fusion, kernel 数 1000+ | ❌ 每个 kernel 都 cuLaunchKernel | — | 纯 PyTorch eager, 把所有 model 代码当 Python 程序逐行翻译成 kernel launch, 是 baseline |
| **B** compile-default | ✅ TorchDynamo 把 Python 字节码提取成 FX graph, 后续 call 跳过 dispatcher | ✅ TorchInductor 把 FX graph 编译成融合 kernel (1000+ → ~200-300), 生成 Triton 代码 | ❌ 每个融合 kernel 仍走单次 cuLaunchKernel | min (Inductor 默认 autotune, 每 GEMM 试 ~5-10 种分块) | `torch.compile(model, mode="default")`. 只做 kernel fusion + Triton 生成, **不**自动叠 CUDA Graph |
| **C** cuda-graph (manual) | ❌ kernel 第一次 capture 时走 eager, 之后 replay 跳过 Python | ❌ 无 fusion, capture 进图的是 eager 的 1000+ kernel | ✅ 显式 `torch.cuda.CUDAGraph()` capture-replay, 整张图一次 cuLaunchGraph | — | 仿 V1 论文 §4.2.1: 预分配 buffer + warm-up + 显式 capture-replay。**FAILED** — `embed_prefix:229` 的 `torch.tensor(att_masks_list)` 是 host list → CUDA tensor copy, 违反 capture 内禁止 host-device sync 约束 |
| **D** compile-reduce-overhead | ✅ 同 B | ✅ 同 B (~200-300 融合 kernel) | ✅ **自动**: Inductor 编译完后 `torch.cuda.make_graphed_callables` 自动包 CUDA Graph | std (与 B 同样的 autotune) | `torch.compile(model, mode="reduce-overhead")`. = default mode + 自动 CUDA Graph。"compile + graph 二合一"标准接口 |
| **E** compile-max-autotune | ✅ 同 B/D | ✅ 同 D | ✅ 同 D (自动) | **max**: Inductor 对每个 GEMM/matmul 跑 20+ Triton 模板变体的完整 sweep, 选最快的 | `torch.compile(model, mode="max-autotune")`. = reduce-overhead + 更深 kernel autotune。**这是 deepdive_kai0 `PI0Pytorch.__init__:113` 当前默认** |

##### 实测数据表

| # | Backend | 优化叠加 | Mean (ms) | Std | P50 | P95 | P99 | Min | Speedup | 首次推理<br>(含 compile) |
|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | eager | (baseline) | 240.7 | 3.6 | 239.7 | 246.6 | 254.6 | 234.9 | **1.00×** | 0.6 s |
| B | compile-default | + Inductor fusion | 110.3 | 1.7 | 109.7 | 113.2 | 118.0 | 108.7 | **2.18×** | 40 s |
| C | cuda-graph (manual) | + CUDA Graph (无 fusion) | FAILED | — | — | — | — | — | — | — |
| D | compile-reduce-overhead | + Inductor fusion + auto CUDA Graph | 48.3 | 0.6 | 48.1 | 50.2 | 50.3 | 48.0 | **4.98×** | 37 s |
| E | compile-max-autotune | + Inductor fusion + auto CUDA Graph + 深 autotune | **43.6** | **0.3** | **43.5** | 43.7 | 44.9 | 43.4 | **5.52×** | 39 s |

##### 分量贡献分解

| 边际增量 | Δ (ms) | 倍数 | 解读 |
|:---:|---:|---:|---|
| **B − A** | -130.4 | **2.18×** | 纯 Inductor fusion 单独收益 (kernel 数 1000+ → ~200-300, 减少 launch + matmul fusion) |
| **D − B** | -62.0 | **2.13×** | 在 fusion 之上叠加 CUDA Graph 的边际收益 — **完美印证 V1 论文第一阶段 ~2×** |
| **E − D** | -4.7 | **1.11×** | max-autotune 比 reduce-overhead 多 10% (更深 GEMM 分块搜索) |
| E − A | -197.1 | **5.52×** | 总收益 (deepdive_kai0 当前默认 vs 纯 eager baseline) |

##### 抖动分析 (Std / 稳定性)

| Backend | Std | Std / Mean | P99 − P50 | 抖动级别 |
|:---:|---:|---:|---:|---|
| A eager | 3.6 ms | 1.5% | +14.9 ms | 中等 (Python dispatcher 不稳定) |
| B compile-default | 1.7 ms | 1.5% | +8.3 ms | 中等 |
| D reduce-overhead | 0.6 ms | 1.2% | +2.2 ms | 低 |
| **E max-autotune** | **0.3 ms** | **0.7%** | **+1.4 ms** | **极低** (CUDA Graph 锁定执行路径) |

**关键洞察**: CUDA Graph (D/E) 不只加速, **还显著降低延迟抖动** — P99−P50 从 14.9ms 降到 1.4ms (**抖动降 10×**)。这对真机 RTC 调度极其重要 (P99 接近 P50, 不会出现单次推理超时打乱 chunk 节奏)。

##### 核心结论

- E (deepdive_kai0 当前默认 max-autotune) **P50 = 43.5 ms** — 已远低于 60ms 阈值
- **策略 B (`torch.compile(max-autotune)`) 已饱和**, **不需要走策略 A (V1 手写 Triton 全栈端口)**
- 实际 inference_rate=3Hz timer 节流 333ms, 43.5ms 推理过剩, 可提至 10-20 Hz 让 RTC 更稳

**与 V1 论文对应**: D vs B = 2.13× 印证 V1 第一阶段 CUDA Graph 收益 ~2×; E vs D = 1.11× 反映 max-autotune 多 10% 边际。

##### 首次推理开销 (cold-start vs steady-state)

| Backend | 首次 (ms) | 稳态 P50 (ms) | cold-start 开销 |
|:---:|---:|---:|---|
| A eager | 597 | 239.7 | ~360 ms (cuDNN heuristic 初始化) |
| B compile-default | 40,178 | 109.7 | **40 秒** (TorchDynamo + AOTAutograd + Inductor 编译) |
| D reduce-overhead | 37,175 | 48.1 | 37 秒 (含 CUDA Graph capture) |
| E max-autotune | 39,075 | 43.5 | 39 秒 (autotune 跑 ~20 配置 × 多个 GEMM) |

**生产影响**: 启动 `serve_policy_pytorch` 时需要预热, 加 30-40 秒 cold-start。这是 `start_autonomy_from_ckpt.sh` 启动后需要"等模型 ready"的实际时间。

##### 实测过程踩坑 (前置工程一并落地)

1. **sm_120 不兼容**: kai0/.venv 的 PyTorch 2.7.1+cu126 仅支持到 sm_90, 装独立 `kai0/.venv_5090` (PyTorch nightly 2.12.0.dev + CUDA 12.8 + triton 3.7) 解决
2. **PI0Pytorch 内部 mixed dtype** (sample_noise/dt/time 硬编码 fp32, PaliGemma RMSNorm 强制 fp32 输出, action_out_proj fp32 mismatch action_in_proj bf16): benchmark 中通过 monkey-patch 4 处 (sample_noise / sample_actions / denoise_step / _prepare_attention_masks_4d) 让 bf16 自洽
3. **Dynamo 不能 trace `class SimpleProcessedObservation:` (nested class)**: `torch._dynamo.disable(_preprocessing.preprocess_observation_pytorch)` 让 dynamo 跳过 preprocess (eager 跑)
4. **C 失败 (CUDA Graph capture)**: `embed_prefix:229` 的 `torch.tensor(att_masks_list, ...)` 是 host list→device tensor copy, 违反 capture 约束; 需要把 att_masks 改成预分配 GPU tensor 才能 capture (model code 改动, 阶段 3.2 序后修)

**前置障碍 (已解决)**: `kai0/.venv` 的 PyTorch 2.7.1+cu126 不支持 sm_120 (Blackwell 5090). 解法: 装独立 venv `kai0/.venv_5090` (PyTorch nightly cu128) + `.venv_5090_trt` (稳定 2.7.1+cu128), 与训练 venv 隔离.

**实际实施路径 (v0.11)**: 不自写 Triton, 改走"V1 Triton 复用 + 5090 重 autotune"路径. 见 §6 完整 Step 0-9 记录:
- 工程量: 自写 1-2 周 → V1 复用 3 天
- 性能: PyTorch max-autotune 41 ms → V1 + 5090 tune 32.05 ms
- 代码位置: `optimize/v1_triton/{pi05_infer_tuned.py, convert_kai0_to_v1.py, benchmark_kai0_v1.py}`
- 数值对齐: rel error 1.42%, per-dim MAE < 0.01 rad

**剩余产出**: 把 V1 推理路径包装为 `serve_policy_v1.py` (WebSocket 协议与 JAX serve 一致), 见 §3.1.2 子任务清单 (待做).

#### 3.4.3 sidecar framework 字段 + 启动脚本分支 (1-3 天)

**改动文件**:
- `start_scripts/kai/start_autonomy_from_ckpt.sh`:
  ```bash
  FRAMEWORK=$(python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('framework', 'jax'))")
  if [ "$FRAMEWORK" = "pytorch" ]; then
      export OPENPI_SERVE_ENTRY="serve_policy_pytorch.py"
      export OPENPI_SERVE_PORT=8001
  else
      export OPENPI_SERVE_ENTRY="serve_policy.py"
      export OPENPI_SERVE_PORT=8000
  fi
  ```
- sidecar `train_config.json` 协议扩展:
  ```jsonc
  // 现有
  {"base_config_name": "...", "override_asset_id": "..."}

  // 扩展
  {"base_config_name": "...", "override_asset_id": "...", "framework": "jax"}
  // 或 "framework": "pytorch", 默认 "jax" 向后兼容
  ```
- `start_scripts/kai/start_autonomy.sh` 接受 `config_name:=` 参数时透传给两个 serve

#### 3.4.4 新 fine-tune 切 PyTorch 策略 (持续, per-ckpt)

落地后, 后续 fine-tune 决策树:
- **简单 ablation / 快迭代实验** → JAX (现有管线稳定, 训练速度无差)
- **计划上 sim01 真机长期部署** → PyTorch (拿推理加速)

训练 + 真机测试都走原生 framework, 不跨框架转换。

#### 3.4.5 TensorRT 路径回顾 (2026-05-20)

> **TL;DR**: TRT 路径尝试过, 全部阻塞失败, 转走 V1 Triton 路径成功 (P50=32 ms, §6). 本节沉淀 5 个阻塞点 + 重启条件, 防止重复趟坑.

**Q**: 为什么 V1 Triton 而非 TensorRT?
**A**: TRT 是 §2 排名 #2 (Y fallback), 价值 3-5×. 实际尝试时遇到 5 个阻塞 (3 个工具链 / 2 个技术), 全部解或绕开后**仍卡在 ONNX export pi05 flow loop**. 同期 V1 Triton 路径直接跑通且更快 (8.0×), 故停 TRT 转 V1.

##### A. 已就绪资产 (留作未来重启用)

| 项 | 状态 | 位置 / 备注 |
|---|---|---|
| `kai0/.venv_5090_trt` | ✅ 完整 | Python 3.10 + **PyTorch 2.7.1+cu128 (stable)** + **TensorRT 10.14.1** + tensorrt_bindings/libs |
| `optimize/results/pi05_aoti.pt2` | 🗃 6.3 GB | AOTI 编译产物 (load fail, 留作 PyTorch stable 后重测) |

> TRT pipeline 代码 (`pi05_trt_pipeline.py` 5-stage 流水线)、工具链 smoke test (`trt_smoke_test.py`)
> 与 plan 文档 (`TRT_30ms_PLAN.md`) 已于 2026-05-28 删除 — TRT 路径已废弃 (V1 Triton 取代),
> 关键教训保留在下方 B-E 节. 未来重启 (选项 1, PyTorch 2.13 stable) 时按 B 节阻塞点 + 现有
> `.venv_5090_trt` 重建即可.

##### B. 5 个阻塞点 (按尝试顺序)

| # | 阻塞 | 已解 / 未解 |
|:---:|---|---|
| 1 | `kai0/.venv` PyTorch 2.7.1+cu126 **不支持 sm_120** (Blackwell 5090) | ✅ 装 `kai0/.venv_5090` (PyTorch 2.12 nightly + cu128) |
| 2 | `torch_tensorrt` nightly 强制 CUDA 13 (我们 cu128 / 12.8) | ⚠ 绕开: 不用 torch_tensorrt, 直接走 ONNX → trtexec |
| 3 | NVIDIA pypi `pypi.nvidia.com` 子目录 GET 持续 hang (网络层) | ✅ file-copy `tensorrt + tensorrt_bindings + tensorrt_libs` 从 phantom env |
| 4 | phantom env 是 Python 3.10, `.venv_5090` 是 3.12 → C ext ABI 不通 | ✅ 新建 `kai0/.venv_5090_trt` (Python 3.10 + PyTorch 2.7 stable + cu128) |
| 5 | **ONNX export pi05 flow loop 10 步 denoise** — 真正技术难点 | ❌ **未解** — `torch.export` 对 dynamic shape + control flow + flow matching loop 支持不完善; `onnxscript`/`ml_dtypes` 下游依赖也有 sm_120 兼容问题 |

##### C. Backend H (AOTInductor) 同期阻塞

|  | 状态 |
|---|---|
| compile | ✅ 7 分钟出 6.3 GB `optimize/results/pi05_aoti.pt2` (wrapper.so + 60 cubin) |
| runtime load | ❌ `AOTIModelPackageLoader create_func_ API call failed` (C++ runtime line 123) |
| 推测原因 | PyTorch 2.12 nightly 的 AOTI runtime 与 sm_120 cubin loading 不兼容 (nightly bug), 或 nvcc 12.4 编译的 wrapper.so 与 cu128 ABI 不匹配 |

##### D. 最终决策依据 (来自 `optimize/results/FINAL_30ms_attempt_summary.md`)

V1 §4.2.2 列的 **8/8 优化项全部已被 Inductor max-autotune 自动捕获** (Backend K 手动追加 QKV 融合验证 Δ=-0.1ms 噪声内):

| V1 优化 | 我们栈 (E max-autotune) | 手动追加 (K) |
|---|---|---|
| RMSNorm 融合 | ✅ Inductor 自动 | — |
| **QKV 投影融合** | ✅ **Inductor 自动 (horizontal_fuse pass)** | -0.1 ms (噪声) |
| RoPE 融合 | ✅ Inductor 自动 | — |
| 动作时间编码折叠 | ✅ Inductor 自动 | — |
| GEMM 分块调优 | ✅ max-autotune sweep 21 配置 | — (F coord_descent 退化 6×) |
| 门控线性层融合 | ✅ Inductor 自动 | — |
| Split-k | ✅ max-autotune 覆盖 | — (V1 报告本身 <0.1ms) |
| 标量操作融合 | ✅ Inductor epilogue_fusion 自动 | — |

**结论**: PyTorch 内置工具链 5090 sm_120 bf16 极限 = **41.0 ms** (Backend E, §3.4.2 实测). 追加 V1 论文 §4.2.2 全部 fusion 不再有效 — Inductor 已自动. 触达 30ms 需要跳出 PyTorch 工具链.

##### E. 未来重启 TRT 的 4 条路径

| 选项 | 触发条件 | 工程量 | 风险 |
|---|---|---|---|
| **1** | 等 PyTorch 2.13 stable + torch-tensorrt 2.13 stable + cu128 (估 2026 Q2-Q3) | 重测既有 pipeline | 极低 |
| 2 | 5090 跑 sm_90 PTX 模拟 (PyTorch 2.7 stable + cu126) | 1-2 天 (新 venv + ckpt 兼容验证) | 中 (损失 sm_120 新特性) |
| 3 | 完整 ONNX 流水线 (Python 3.10 + PyTorch 2.7 + cu124 + 拆分 flow loop) | 3-5 天 | 高 (ONNX 导出 flow loop 是技术难点) |
| **4** ✅ | **接受 V1 32ms 当前 best, TRT 留 fallback Y** | 0 (当前路径) | 0 |

**当前选 4**. 重启选 1 需要等 PyTorch stable, 预计 2026 Q2-Q3.

##### F. 相关链接

- `optimize/results/FINAL_30ms_attempt_summary.md` — 完整 30ms 攻关时间线 + 全 8 backend 实测表
- `optimize/results/phase1_F_G_J_findings.md` — F/G/J backend 失败分析
- §3.4.2 实测踩坑 — 前置障碍解决记录 (sm_120 兼容 / dtype mismatch 等)
- §6 V1 Triton 实施日志 — 取代 TRT 的最终路径

### 3.5 阶段 4 — 任务质量 (3-6 周)

#### #4 V2 速度自适应学习 (V2 §4.4)

**目标**: 折叠衣袖 / 抓叠对位等精细阶段自动降速, 其他阶段加速

**步骤**:
1. 改 teleop 节点 `arm_teleop_node.py` 加油门输入 (脚踏板 / 摇杆 axis)
2. 收集 ~200 个 episode, 每帧记 `(observation, target_speed_factor)`
3. 训轻量回归 head (1 层 MLP), 输入: pi05 image encoder 输出 + 关节状态, 输出: `speed_factor ∈ [0.5, 4.0]`
4. head 训练用哪个框架? **遵从 ckpt framework** (PyTorch ckpt 用 PyTorch 训 head, JAX ckpt 用 JAX)
5. 部署: `speed_factor` → 阶段 2 QP 的 `dt_ref` 乘子 (与 QP 联动)
6. 迭代: 每天训练, 次日部署更高基线速度

**实验迭代瓶颈**: 200 ep 油门 teleop 数采 = ~10-20 hr 操作员时间, AI 不能替代。

**多任务环境**: deepdive_kai0 同时跑 Task A / P / PS, 需 task-conditioned 速度模型, 或 per-task 训练。

### 3.6 阶段 5 — 推理极致 (可选)

#### #3 Flash 推测推理 — **降级为研究项**

**降级理由 (v0.11 数据)**: 当前 baseline V1+autotune **P50=32 ms** (§6). Flash 3× 复合 → ~11 ms 端到端, 但:
- inference_rate=3Hz timer 节流 333ms, 提至 10Hz 也只是 100ms 周期 → 11ms vs 32ms 在 RTC 节奏里无差
- CUDA Graph 已把 P99−P50 抖动控制在 ±1 ms 内
- 工程量 4-8 周 + 每 ckpt 训 draft 3-6 hr GPU, 边际收益与成本不匹配

**仅当以下场景成立才重启**:
- 动态目标任务 (传送带分拣等, 推理 < 30ms 才有意义)
- inference_rate 拉到 30+ Hz (周期 < 33ms)

**实施路径 (备查)**: per-ckpt 110M draft 训练 + flow-matching 端点重建并行验证 + 阶段 fallback 信号 (论文用夹爪过零, deepdive_kai0 折叠/抓叠任务需补关节速度突变 / 视觉特征余弦相似度) + δ 阈值 per-task 调.

#### #7 客户端 MPC + 滞后辨识 (3-5 周)

**前置**: 真机测试 2 测出 Piper t_motion > 50ms; 若 < 30ms 直接 skip

**步骤**:
1. chirp 扫频测单关节响应, 辨识 τ
2. 新增 `ros2_ws/src/piper/scripts/mpc_tracker_node.py` 跑 acados/CasADi MPC
3. 与 Piper 自带 PD 兼容性 — 可能需要降 Piper PD 增益让 MPC 主导

---

## 4. 真机测试方案

### 4.1 测试 1: sim01 模型实际推理延迟 ✅ 完成 (2026-05-20)

**目的**: 区分"模型推理时间" vs "timer 节流时间", 确定推理优化上限.

#### 4.1.1 测量方法

利用 `ros2_ws/src/piper/scripts/policy_inference_node.py:2085-2148` 已内置的 inference timer:

```python
t_start = time.monotonic()              # line 2085
# ... obs 构造 + RTC 注入 + ...
result = self.policy.infer(obs)         # line 2145, WebSocket 同步调用
infer_ms = (time.monotonic() - t_start) * 1000   # line 2147
self.get_logger().info(f'infer {infer_ms:.0f}ms | chunk={actions.shape} | ...')
# line 2229: 每次推理都 log 一条
```

**测量范围 (client 视角)**: `t_start` 含 obs 构造 + WebSocket send → server side JAX `sample_actions` JIT 推理 → WebSocket recv → ROS 落 actions. 这是 **端到端 RTT** , **不是裸 GPU 推理时间**.

**测量工具**: `start_scripts/kai/diag/measure_jax_infer_latency.sh` (87 行) 自动从 `~/.ros/log/` 找最新含 `infer XXXms` 的 log, 提取数值, 算分位数 + 阈值决策.

#### 4.1.2 实验配置

| 项 | 值 |
|---|---|
| **Session log** | `~/.ros/log/python_659068_1779190543244.log` (468 KB) |
| **日期** | 2026-05-19 19:36-19:43 (~7.9 min wall clock) |
| **Config name** | `pi05_flatten_fold_a_new_pure_1200` |
| **Checkpoint** | `/data1/DATA_IMP/checkpoints/task_a_pure200_base_pi05_step49999` |
| **Asset id (norm_stats)** | `a_new_pure_200` |
| **Mode** | RTC enabled (`Pi0Config → Pi0RTCConfig`, `pi05=True`) |
| **Execution** | joint, depth_in=False, ee_pose_in=False, EXECUTE 真机 |
| **Hardware** | RTX 5090 sm_120, sim01/ipc01 |
| **Timer** | `inference_rate = 3.0 Hz` (周期 333 ms) |
| **Chunk** | (50, 14) joint action, 14-DoF dual Piper |

#### 4.1.3 原始数据样本 (前 5 条)

```
[1779190576.541] infer  209ms | chunk=(50, 14) | L[0]=[-0.07,+0.06,-0.52,...] R[0]=[+0.27,+0.30,-0.42,...]
[1779190584.858] infer 8190ms | ...                    ← JIT cold-start outlier (excluded)
[1779190585.135] infer  272ms | chunk=(50, 14) | ...
[1779190585.399] infer  202ms | chunk=(50, 14) | ...
[1779190585.715] infer  184ms | chunk=(50, 14) | ...
```

**清洗规则**: 跳过前 5 个 (JIT compile warmup) + 排除 > 500 ms (1 个 outlier @ 8190 ms 估计为 GC / cache cold).

| | Raw | Cleaned |
|---|---:|---:|
| N samples | 1305 | **1299** |
| Dropped | — | 6 (0.46%) |

#### 4.1.4 实测分位数

| 指标 | 值 (ms) |
|---|---:|
| **P50** | **196.0** |
| P95 | 221.0 |
| P99 | 232.0 |
| Mean ± Std | 198.6 ± 13.2 |
| Min — Max | 174 — 276 |
| P95 − P50 (jitter) | 25.0 |
| P99 − P50 (tail) | 36.0 |
| Session 推理率 | 2.93 Hz (timer 333ms 59% utilization) |

#### 4.1.5 分布直方图 (cleaned, N=1299)

```
[  0, 180) ms |   22 (1.7%)
[180, 190) ms |  380 (29.3%) ██████████████   ← P50 / mode
[190, 200) ms |  356 (27.4%) █████████████
[200, 210) ms |  232 (17.9%) ████████
[210, 220) ms |  222 (17.1%) ████████
[220, 230) ms |   68 (5.2%)  ██
[230, 250) ms |   18 (1.4%)
[250, 300) ms |    1 (0.1%)
[300, 500) ms |    0 (0.0%)
```

**形态**: 单峰窄分布 (CV = Std/Mean = 6.6%), 主体集中 180-220 ms. **无长尾**, P99-P50 仅 36 ms — JAX/XLA 在 5090 上的步进时间非常稳定.

#### 4.1.6 与 V1 路径对比

| 指标 | JAX (sim01 实测, 端到端 RTT) | V1 Triton (offline 5090 benchmark, 裸推理) | 提升 |
|---|---:|---:|---:|
| P50 | **196 ms** | **32 ms** | **6.1×** |
| P95 | 221 ms | ~33 ms | ~6.7× |
| Jitter (P95-P50) | 25 ms | < 1 ms | -96% |
| Std/Mean | 6.6% | < 0.5% | -92% |

> 注: V1 数据来自 §6 offline benchmark, 未含 WebSocket overhead. 真 V1 serve 部署后, RTT 将增 WebSocket + obs 构造开销 (~5-10 ms 量级, B1 profile 待量化). 但即便 +10 ms, V1 serve RTT 估 ~42 ms ≪ JAX 196 ms, 仍 4-5× 加速.

#### 4.1.7 决策映射

P50 = 196 ms → 落在 "100-200ms 标准 5090 baseline" 档:

| 决策项 | 结论 |
|---|---|
| **V1 路径价值** | ✅ **确认 6.1× 加速空间, §6 V1 Triton 实施 + §7 B4 serve 包装方向正确** |
| **AOT compile 优先级** | ❌ 抖动 25ms ≪ 100ms 阈值, JAX 端无 AOT 需求 |
| **inference_rate 提升潜力** | ⏳ JAX 196ms 占满 3Hz × 59%, 提至 5Hz 即超时. **V1 落地后 32ms 可拉 20-30 Hz** (C2 子任务) |
| **#6 JAX 浅层优化 (阶段 1)** | △ 可顺手做 (AOT cache / bf16 局部), 估 1.5-2× → 100-130 ms, 但价值低于直接走 V1 |

阈值表对照:

| 指标 | 后续行动 |
|---|---|
| P50 < 80ms | 模型已很快, V1 收益小, 阶段 3 优先级可降低 |
| **P50 100-200ms** ✅ | **标准 5090 baseline, V1 路径价值明显, 6.1× 加速** |
| P50 > 250ms | 可能有 cache miss / fp32 残留, V1 收益最大 |
| P95-P50 > 100ms | 抖动严重, AOT compile 必做 (sim01 实测 25ms, 不需要) |

#### 4.1.8 测量复跑

```bash
# 自动找最新 log
./start_scripts/kai/diag/measure_jax_infer_latency.sh

# 指定 log
./start_scripts/kai/diag/measure_jax_infer_latency.sh <log_file>

# 调过滤
SKIP_WARMUP=10 MAX_MS=300 ./start_scripts/kai/diag/measure_jax_infer_latency.sh
```

### 4.2 测试 2: Piper 关节 t_motion 滞后

**目的**: 量化 motor 响应滞后, 决定阶段 5 #7 是否值得。

**方法**: 单关节阶跃响应测试
```python
# ros2_ws/src/piper/scripts/test_motor_lag.py (待写)
# 发 0.1 rad 阶跃到 left_arm joint 1, 同时记录 joint_state 回传
# t0 = 发送时刻
# t1 = joint_state 跨越 50% 步进时刻
# t_motion = t1 - t0
# 重复 10 次取均值 / 方差
```

**期望读出 → 后续决策**:
| t_motion | 决策 |
|---|---|
| < 30ms | #7 skip; Piper 自带 PD 已足够 |
| 30-80ms | #7 中等价值, 可后置 |
| 80-150ms | #7 价值升至中段 |
| > 150ms | #7 价值高, 紧急做 |

---

