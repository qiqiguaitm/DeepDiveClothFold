# Pi0.5 推理后端对比测试 Plan

> 配套文档: [realtime_vla_optimization_analysis.md](realtime_vla_optimization_analysis.md) §3.4.2 (阶段 3 选项 X 的 PyTorch 推理 serve 搭建)
> 目的: 实测验证 V1 手写 Triton + CUDA Graph vs torch.compile 的实际收益差距, 决定阶段 3.4.2 用策略 A 还是策略 B
> 状态: Plan v0.1, 待用户拍板后实施

---

## 1. 目标

对比 4 种 pi0.5 推理实现的延迟分布, 回答下面 3 个问题:

1. **torch.compile (reduce-overhead) 能拿到多少 V1 手调路线的收益?** 这决定阶段 3.4.2 是否需要写 Triton kernel
2. **CUDA Graph 单独的收益是多少?** 这是 V1 第一阶段的核心机制
3. **pi0.5 PyTorch forward 是否能 fullgraph 编译?** flow matching loop / dynamic prompt_len 可能 graph break

---

## 2. 测试条件 (5 个 backend, v0.2 updated)

| 编号 | 实现 | Inductor fusion | CUDA Graph | autotune |
|:---:|---|:---:|:---:|---|
| **A** | PyTorch eager (手动解 compile) | ❌ | ❌ | — |
| **B** | `torch.compile(mode="default")` | ✅ | ❌ | min |
| **C** | 手动 `torch.cuda.CUDAGraph` | ❌ | ✅ | — |
| **D** | `torch.compile(mode="reduce-overhead")` | ✅ | ✅ (auto) | std |
| **E** | `torch.compile(mode="max-autotune")` (**deepdive_kai0 默认**) | ✅ | ✅ (auto) | **max** |

**关键事实**: `PI0Pytorch.__init__:113` 默认就把 `sample_actions` 用 `torch.compile(max-autotune)` 包装了。E 是 deepdive_kai0 当前实际运行状态, A baseline 需脚本显式解除 compile (`del model.sample_actions` 让其落回 class method)。

注: V1 论文路径 (= ✅ inductor + ✅ graph + ✅ 手写 triton) 不在本测试范围, 因为手写 Triton 需要 5090 重 autotune 4-6 周, 是阶段 3.4.2 策略 A 的内容; 本测试是验证策略 B (compile/reduce-overhead/max-autotune) 是否够用。

---

## 3. 硬件 / 软件环境

### 硬件选择 (待用户拍板)

| 选项 | 优势 | 劣势 |
|---|---|---|
| **sim01 (2× RTX 5090 32GB)** | 实际部署环境, 直接给阶段 3.4.2 决策提供数据 | 5090 是 Blackwell sm_120, PyTorch 2.5+ 与 sm_120 编译兼容性可能有坑 |
| **gf0 (8× A100 80GB)** | A100 PyTorch 生态最稳定, 数据可信度最高 | 训练机, 跟部署环境 (5090) 算力曲线不同 |
| **uc02 (H100)** | H100 sm_90 与 5090 sm_120 相近 | 资源占用与训练任务冲突 |

**推荐**: sim01 5090 (优先) → 若 PyTorch 2.x 与 sm_120 有兼容问题, 退到 A100 / H100 做 baseline

### 软件
- PyTorch: 2.5+ (CUDA Graph API 稳定 + torch.compile 成熟)
- Python: 3.12 (与 deepdive_kai0 venv 一致)
- CUDA / cuDNN: 跟随硬件 (5090 需 CUDA 12.6+)
- triton: 3.0+ (pi05_pytorch 内部已用)

---

## 4. 模型与输入

### Pi0.5 模型
- 来源: `kai0/src/openpi/models_pytorch/pi0_pytorch.py::PI0Pytorch`
- 配置: pi05=True, action_horizon=50, joint_dim=14, num_cameras=3
- 模式: `model.eval()` + `torch.inference_mode()`

### 权重选择 (推荐 (a))

| 选项 | 描述 | 适合本测试? |
|---|---|:---:|
| **(a) 随机初始化** | 架构同 pi05, weights 随机, 数值无意义 | ✅ (只测速度, 数值无关) |
| (b) 从 JAX ckpt 转 | 用 `realtime-vla/convert_from_jax_pi05.py` 转 PyTorch state_dict | △ (额外工作, 但可顺便验证 ckpt 转换路径) |

**默认选 (a)**. 若想顺便验证转换路径, 选 (b)。

### Dummy 输入 (与真实 ROS2 推理 payload 形状一致)
```python
inputs = {
    "observation_images": torch.randn(1, 3, 3, 224, 224, dtype=torch.bfloat16, device="cuda"),
    # (B=1, num_cam=3, channels=3, H=224, W=224)
    "joint_state": torch.randn(1, 14, dtype=torch.float32, device="cuda"),
    "prompt": "Flatten and fold the cloth",  # 或 token ids
    "diffusion_noise": torch.randn(1, 50, 14, dtype=torch.float32, device="cuda"),
}
```

---

## 5. 测速指标

### 每个 backend 采集
- **Warm-up**: 10 次推理 (不计时, 触发 compile / CUDA Graph capture / cuDNN autotune)
- **测量**: 100 次推理 (`torch.cuda.synchronize()` 后取 `time.perf_counter()`)
- **统计**:
  | 指标 | 含义 |
  |---|---|
  | mean ± std | 平均 + 抖动 |
  | P50 / P95 / P99 | 分布尾巴 |
  | min | 极限性能 |
  | GPU memory peak | `torch.cuda.max_memory_allocated()` |
  | first-call time | 首次推理 (含 compile 触发 / graph capture) |

### 二级指标 (可选)
- **NVTX profile**: 用 `nsys profile` 跑一次, 看 CPU 端 launch overhead 占比
- **kernel 数量**: 在每个 backend 下数 CUDA launch 数 (用 `nvprof` 或 nsys)

---

## 6. 实现要点

### A. PyTorch eager
```python
@torch.inference_mode()
def run_eager(model, inputs):
    return model(**inputs)

# 测速:
torch.cuda.synchronize()
t0 = time.perf_counter()
run_eager(model, inputs)
torch.cuda.synchronize()
t1 = time.perf_counter()
```

### B. torch.compile default
```python
model_b = torch.compile(model, mode="default", fullgraph=False)
# fullgraph=False 允许 graph break (pi05 flow matching 可能有)
# 也测一个 fullgraph=True 看是否成功
```

### C. 手动 CUDA Graph
仿 `realtime-vla/pi05_infer.py:746-802`:
```python
# 1. 预分配 buffers (输入/输出/中间所有 tensor)
buffers = {
    "obs_images": torch.empty_like(inputs["observation_images"]),
    "joint_state": torch.empty_like(inputs["joint_state"]),
    "diffusion_noise": torch.empty_like(inputs["diffusion_noise"]),
    "output_action": torch.empty(1, 50, 14, device="cuda"),
}

# 2. Warm-up (3 次让 cuDNN/cuBLAS 选 kernel 变体)
for _ in range(3):
    output = model(**buffers)  # 直接读 buffer

# 3. Capture
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    graph.capture_begin()
    output = model(**buffers)
    graph.capture_end()

# 4. Replay
def run_cuda_graph(new_inputs):
    buffers["obs_images"].copy_(new_inputs["observation_images"])
    buffers["joint_state"].copy_(new_inputs["joint_state"])
    buffers["diffusion_noise"].copy_(new_inputs["diffusion_noise"])
    graph.replay()
    return output  # output 在 capture 时锁定地址, replay 后含新结果
```

**关键约束**:
- pi05 forward 内若有 `.item()`, `.cpu()`, `print()` 等 host sync → 必须移到 capture 外
- prompt 编码必须 capture 前预处理好 (Tokenizer 在 Python 层, 不能进 graph)
- shape / dtype 全程固定

### D. torch.compile reduce-overhead
```python
model_d = torch.compile(model, mode="reduce-overhead", fullgraph=True)
# 自动叠加 CUDA Graph
# 若 fullgraph=True 失败 (graph break), fallback fullgraph=False 测一次
```

---

## 7. 测试脚本

实现(已落地真实文件): **`optimize/benchmark_pi05_inference.py`**(~1161 行, 5 backend A–E, 测 `PI0Pytorch.sample_actions`)。本文档 §2/§4/§6 是其设计依据(脚本头部也回指本文档)。

```bash
.venv/bin/python optimize/benchmark_pi05_inference.py --backends A,B,C,D,E --n-test 100
```

结构: `build_model`(随机权重 pi05)→ `make_dummy_observation`(3 相机 224² bf16 + state + prompt 64 token)→ 每 backend 包成闭包(eager / compile-default / cuda-graph / reduce-overhead / max-autotune)→ `benchmark(fn, name, n_warmup, n_test)` 采 mean/std/P50/P95/P99/min/peak-mem → 打印表格 + 写 markdown 报告。

---

## 8. 输出格式

### 终端表格
```
=== pi0.5 inference benchmark ===
Hardware: RTX 5090 (sm_120)
PyTorch: 2.5.1, CUDA: 12.6
Date: 2026-MM-DD

Backend                       Mean    Std    P50    P95    P99    Min    Speedup
─────────────────────────────────────────────────────────────────────────────────
A. eager                       XX.X   X.X   XX.X   XX.X   XX.X   XX.X   1.00x
B. compile-default             XX.X   X.X   XX.X   XX.X   XX.X   XX.X   X.XXx
C. cuda-graph                  XX.X   X.X   XX.X   XX.X   XX.X   XX.X   X.XXx
D. compile+graph               XX.X   X.X   XX.X   XX.X   XX.X   XX.X   X.XXx

GPU peak memory: A=X.XGB, B=X.XGB, C=X.XGB, D=X.XGB

First-call time (含 compile / graph capture):
A: XX ms, B: XX,XXX ms, C: XXX ms, D: XX,XXX ms
```

### Markdown 报告 (落到 `docs/deployment/benchmark_results/pi05_inference_<hostname>_<date>.md`)
- 同上表格
- 加 "已知 graph break / 编译失败" 字段
- 加 "5090 vs A100 vs H100" 对比 (若多硬件测试)

---

## 9. 风险点 + 预案

| 风险 | 预案 |
|---|---|
| pi05 PyTorch forward 含 Python control flow (50 步 denoising loop), torch.compile `fullgraph=True` 可能失败 | fallback `fullgraph=False`, 都测; 在结果里标 "graph break" |
| CUDA Graph 不允许 host-device sync; pi05 forward 内 prompt 处理 `tokenizer(...)` 是 CPU 操作 | 像 V1 那样把 tokenizer + prompt embedding 放 capture 外, 只把 transformer + denoising 放 graph 内 |
| 5090 (sm_120) 上 PyTorch 2.5 + Triton 3 兼容性 | 先在 A100 (gf0) 跑通验证 baseline 正确, 再迁 5090 |
| pi05 forward 不接受 dict 输入, 接口可能要 unpack | 看实际 `PI0Pytorch.forward` 签名调整 |
| bfloat16 + Linux 5090 driver 兼容性 | 先测 fp32, 再切 bf16 |

---

## 10. 执行步骤

1. **用户确认** §3 硬件选择 + §4 ckpt 选择 + §9 是否多硬件对比
2. ✅ 脚本已落地: `optimize/benchmark_pi05_inference.py`(5 backend A–E)
3. 在选定机器上跑 (单次完整测试 ~5-10 min, 含 4 backend × warm-up + 100 测量)
4. 输出落到 `docs/deployment/benchmark_results/pi05_inference_<hostname>_<date>.md`
5. 结果反馈主分析文档:
   - 若 D ≥ 0.8 × C → 阶段 3.4.2 直接用策略 B (torch.compile reduce-overhead), 不写 Triton kernel
   - 若 D < 0.5 × C → 阶段 3.4.2 需走策略 A (V1 手写 Triton)
   - 若 D 不能 fullgraph 编译 → 评估 graph break 影响, 可能需走策略 A

---

## 11. 待用户确认问题

### Q1: 硬件选择
- (a) **sim01 5090** (推荐, 直接对应阶段 3.4.2 决策)
- (b) gf0 A100 (PyTorch 最稳)
- (c) 多硬件对比 (sim01 + gf0, 工时 ×2)

### Q2: ckpt 来源
- (a) **随机初始化** (推荐, 只测速度)
- (b) 从现有 JAX ckpt 转 PyTorch (顺便验证转换路径)

### Q3: 完整推理 vs 只测 forward?
- (a) **完整推理** (含 50 步 denoising loop, 与生产一致)
- (b) 只测单 forward (更短, 速度对比更纯)

### Q4: 是否测 inference_rate 上限?
- 这个测试只测单次推理时间, **不直接测 inference_rate 上限**
- 但单次推理 P95 < 100ms → 可以推断 10Hz 跟得上
- 如需直接测 rate, 应另起任务跑 `policy_inference_node` + `ros2 param set inference_rate 10.0`

### Q5: 是否同时测推理 batch=1 / batch=4 / batch=8?
- 默认 batch=1 (与真机部署一致)
- 大 batch 测试可看 throughput, 但与单次延迟优化关系不大

---

## 12. 后续

测试结果出来后:
- 反馈到 `realtime_vla_optimization_analysis.md` §3.4.2 策略 A vs B 决策
- 若 D 满足需求 → 把策略 B 列为阶段 3.4.2 默认实施路径
- 若 D 不够 → 阶段 3.4.2 走策略 A, 工程量重估

---

## 修订历史

| 版本 | 时间 | 内容 |
|:---:|---|---|
| v0.1 | 2026-05-19 | 初版 plan, 4 backend 对比 (eager / compile / cuda-graph / compile+graph), 6 项待用户确认问题 |
| v0.2 | 2026-05-19 | 用户决策: sim01 5090 + 随机权重 + 完整推理 + batch=1; 加 backend E max-autotune 变 5-backend (PI0Pytorch.__init__:113 默认就是 max-autotune); JAX→PI0Pytorch 转换工具推迟到阶段 3.4.1 实现; 脚本落地 `optimize/` |
