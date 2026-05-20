# Plan: Pi0.5 30ms 推理 — TRT 路线攻关

> 起点: backend E `torch.compile(max-autotune)` 在 5090 上 P50 = 41.0 ms (bf16, 100 iter)
> **目标: 30 ms (-27%)**, 模型架构不变, 不重训, 不量化
> 主要路径: 跳出 PyTorch Inductor, 用 TensorRT 推理引擎

---

## 0. 当前栈障碍 (为什么必须跳出 PyTorch)

| 已试 | 结果 | 上限 |
|---|---|---|
| Inductor max-autotune (E) | ✅ | **41.0 ms** |
| + coord_descent_tuning (F) | ❌ 退化 6× (与 fullgraph=False 冲突) | — |
| + fp16 (G) | ❌ 退化 6× (Inductor fp16 codegen 不成熟) | — |
| + 手动 QKV 融合 (K, V1 §4.2.2) | ✅ 但 mean 不变 (-0.1ms) | Inductor 已自动 |
| AOTInductor (H) | ❌ compile OK 但 load fail (PyTorch 2.12 nightly + sm_120 AOTI bug) | — |
| Inductor + CUTLASS (J) | ❌ OOM 未测完 | — |

**结论**: PyTorch 2.12 nightly + Inductor 极限 = 41ms, 11ms 缺口必须 TRT。

---

## 1. 关键发现 — phantom env 是现成宝藏

经过环境探查, `/data1/miniconda3/envs/phantom/` 已经具备**几乎所有必需依赖**:

| 组件 | 版本 | 状态 |
|---|---|---|
| Python | 3.10 | ✅ |
| **torch** | **2.7.1+cu128** ← **stable, 支持 sm_120** | ✅ |
| torchvision | 0.22.1+cu128 | ✅ |
| **tensorrt** | **10.14.1.48.post1** | ✅ |
| transformers | (已装, 版本待确认与 4.53 兼容性) | ⚠️ 需对齐 |
| flax | — (未装) | ⚠️ 需装 (PI0Pytorch 内部依赖) |
| `torch_tensorrt` | — | ⚠️ 需装 (但可绕过, 用 ONNX 路径) |
| `trtexec` | `/data1/shock/workspace/gaohan/docker/TensorRT-10.10.0.31/targets/x86_64-linux-gnu/bin/trtexec` | ✅ (10.10, 与 phantom env 的 10.14 不同, 需 fallback 到 Python API) |
| nvcc | 12.4 (phantom env) | △ (5090 sm_120 可用) |

**优势**: 不需要新建 conda env, 不需要重装 torch + cu128 + trt (这些 5090 兼容包从 NVIDIA index 装非常慢)。

---

## 2. 实施路线 — 3 个 sub-option

### 选项 A (强烈推荐): phantom env + ONNX → TRT 三段式

```
[phantom env Python 3.10]
   │
   ├─ Step 1: 装 flax + 对齐 transformers (deepdive_kai0 PI0Pytorch 依赖)
   │
   ├─ Step 2: 构建 PI0Pytorch (复用现有 patches: sample_noise/dt/time/embed_prefix/...)
   │
   ├─ Step 3: torch.export.export(wrapper_module, example_inputs)
   │           → ExportedProgram
   │
   ├─ Step 4: torch.onnx.export(ep, "pi05.onnx", ...)
   │           → ONNX graph file
   │
   ├─ Step 5: trtexec 或 tensorrt Python API
   │           --onnx=pi05.onnx
   │           --saveEngine=pi05.engine
   │           --fp16  (bf16 也可)
   │           --tacticSources=+CUBLAS,+CUDNN,+CUTLASS
   │           --builderOptimizationLevel=5  (最深优化)
   │           → pi05.engine
   │
   └─ Step 6: tensorrt Python API runtime
              context.execute_v2(bindings)
              benchmark 100 iter
```

**优势**:
- 不动 deepdive_kai0 现有 .venv_5090 / 主 venv
- 不重装 torch + cu128 + trt (这些用 NVIDIA index 装很慢)
- ONNX export 是工业标准, 兼容性最好
- TRT Python API runtime 推理时**完全 bypass PyTorch**, 期望 -20-30%

**风险**:
- torch.export trace pi05 sample_actions (含 flow matching 10 步循环) 可能 fail
- ONNX 不支持某些 PyTorch op (例如自定义 RoPE), 需要 fallback
- TRT engine 编译要 5-15 分钟 (一次性)

### 选项 B: torch_tensorrt (一步到位但风险高)

```
[phantom env]
   ├─ uv pip install torch_tensorrt (匹配 torch 2.7.1+cu128)
   ├─ model = torch_tensorrt.compile(model, ...)
   └─ benchmark
```

**优势**: 一行代码 (类似 torch.compile)

**风险**: torch_tensorrt 2.7 stable 是否兼容 cu128? PyPI 上没看到, NVIDIA index 之前 hang。这条路在 venv_5090 nightly 上已失败 (要 CUDA 13). phantom env 用 stable 2.7 可能可行, 也可能同样问题。

### 选项 C: 新建 conda env (备用 fallback)

仅在 A/B 都失败时执行: 全新 Python 3.10 + 最新兼容 PyTorch + TRT stack. 3-5 天工程。

**默认选项 A**.

---

## 3. 选项 A 详细步骤

### Step 1: phantom env 依赖对齐 (30 分钟)

```bash
PHANTOM=/data1/miniconda3/envs/phantom

# 1.1 检查 transformers 版本 (PI0Pytorch 要 4.53.2)
$PHANTOM/bin/python -c "import transformers; print(transformers.__version__)"

# 1.2 如果不是 4.53.2, 装 (但要保留 phantom 现有项目兼容性, 谨慎!)
# 推荐: 不动 transformers, 把 transformers_replace patch 重新装一遍到 phantom env
cp -rv /home/tim/workspace/deepdive_kai0/kai0/src/openpi/models_pytorch/transformers_replace/* \
       $PHANTOM/lib/python3.10/site-packages/transformers/

# 1.3 装 flax (PI0Pytorch __init__ 间接依赖? 需测)
$PHANTOM/bin/pip install flax

# 1.4 装 numpy / safetensors (已有但确认)
$PHANTOM/bin/python -c "import numpy, safetensors, einops, jax"
```

**验证标准**:
- `from openpi.models_pytorch.pi0_pytorch import PI0Pytorch` 不报错
- `PI0Pytorch(config)` 实例化成功 (与 .venv_5090 输出一致)

**风险**: phantom env 是 gaohan/dexmal 项目共用, 改 transformers patch 可能破坏其他项目。**先确认 phantom 没在被使用** (`who`, `ps`)。

---

### Step 2: ONNX export pi05 (1-2 天)

这是技术难点。

#### 2.1 包装 nn.Module (复用已有 `_Pi05InferenceWrapper`)

直接复用 `optimize/benchmark_pi05_inference.py` 的 `bench_H_aoti` 中的 `_Pi05InferenceWrapper`. 已知能通过 torch.export。

#### 2.2 torch.export.export

```python
import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
# ... build model + apply patches + wrap ...
ep = torch.export.export(wrapper, example_args, strict=False)
```

H AOTI 验证此步骤可行 (export 用了 59 秒, 成功生成 ExportedProgram)。

#### 2.3 ONNX export 三种方案

**方案 2.3.a (推荐): torch.onnx.export with dynamo=False, 老 tracer**
```python
torch.onnx.export(
    wrapper, example_args, "pi05.onnx",
    dynamo=False,  # 用老 tracer, 兼容性最好
    opset_version=20,  # 5090 + transformer ops
    input_names=[...], output_names=["actions"],
    dynamic_axes=None,  # 静态 shape 推理
    do_constant_folding=True,
)
```

**方案 2.3.b: torch.onnx.export with dynamo=True (新)**
- 更现代, 处理 control flow 更好, 但 nightly 上不稳定
- phantom env torch 2.7.1 stable, 应该 OK

**方案 2.3.c: onnx-export-pt2 (从 .pt2 转 ONNX)**
- 如果 2.3.a/b 失败, 可以从 H AOTI 已生成的 `pi05_aoti.pt2` 反向转 ONNX
- 但 .pt2 包含 cubin + .so, 不一定能转 ONNX

**已知风险点**:
- pi05 sample_actions 内有 `while time >= -dt/2` (我们 patch 为 for-loop), `for` 循环 ONNX 支持需 `opset_version>=15`
- PaliGemma attention 用 SDPA: ONNX 有原生 SDPA op (`Attention-23` since opset 23) 但兼容性挑剔
- `_attn_implementation="eager"` 路径走标准 matmul, ONNX 友好

#### 2.4 ONNX 验证

```bash
# 用 onnxruntime CPU 验证 ONNX 正确性
$PHANTOM/bin/python -c "
import onnxruntime as ort
sess = ort.InferenceSession('pi05.onnx', providers=['CPUExecutionProvider'])
out = sess.run(None, dummy_inputs)
# compare with original torch output, maxabs diff < 1e-3
"
```

---

### Step 3: TRT engine 构建 (4-6 小时, 含编译时间)

#### 3.1 trtexec 命令行 (推荐)

```bash
$PHANTOM/bin/python -m tensorrt_bin.trtexec \
    --onnx=pi05.onnx \
    --saveEngine=pi05.engine \
    --bf16 \  # 与 model dtype 一致
    --tacticSources=+CUBLAS,+CUDNN,+CUTLASS,+EDGE_MASK_CONVOLUTIONS \
    --builderOptimizationLevel=5 \  # 最深优化, 编译时间 +30-60%
    --useCudaGraph \  # 用 CUDA Graph (类似 torch.compile reduce-overhead)
    --memPoolSize=workspace:8192M \
    --shapes=image_base:1x3x224x224,image_left:1x3x224x224,image_right:1x3x224x224,... \
    --verbose
```

**关键 TRT 旗标**:
- `--bf16`: bf16 推理 (与我们 model 一致, 保持数值)
- `--tacticSources=...`: 让 TRT 同时 sweep cuBLAS + cuDNN + CUTLASS + Triton kernel
- `--builderOptimizationLevel=5`: 最深优化 (level 1-5, 5 = 最久最优)
- `--useCudaGraph`: TRT 自带 CUDA Graph
- `--memPoolSize`: workspace 大 (5090 32GB) 让 TRT 找更激进 kernel

#### 3.2 Python API alternative (如果 trtexec 不兼容)

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("pi05.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.BF16)
config.builder_optimization_level = 5
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

engine = builder.build_serialized_network(network, config)
with open("pi05.engine", "wb") as f:
    f.write(engine)
```

---

### Step 4: TRT runtime 推理 + benchmark (半天)

```python
import tensorrt as trt
import torch

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("pi05.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Pre-allocate inputs/outputs in GPU buffers
inputs_gpu = {...}  # bf16 tensors
outputs_gpu = {...}

# Run benchmark 100 iter
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) * 1000 / 100
print(f"TRT mean: {elapsed:.1f} ms")
```

---

### Step 5: 数值对比验证 (1-2 hr)

```python
# Run original (E max-autotune) and TRT on same input, compare outputs
diff = (out_e - out_trt).abs().max().item()
assert diff < 1e-2, f"TRT output diverges: maxabs={diff}"
```

---

## 4. 工程时间表

| 阶段 | 任务 | 工时估计 | 依赖 | 输出 |
|---|---|---|---|---|
| **Step 1** | phantom env 依赖对齐 | 30 min - 1 hr | 现有环境 | PI0Pytorch 可在 phantom 实例化 |
| **Step 2** | ONNX export | **1-2 天** ⚠️ 难点 | Step 1 | `pi05.onnx` 文件, onnxruntime CPU 验证通过 |
| **Step 3** | TRT engine 构建 | 4-6 hr (含 trtexec 编译) | Step 2 | `pi05.engine` |
| **Step 4** | TRT runtime benchmark | 半天 | Step 3 | benchmark 数据 (mean/p50/p99/std) |
| **Step 5** | 数值对比验证 | 1-2 hr | Step 4 | 数值差异报告 |
| **合计** | — | **2-3 天** | — | TRT 30ms 实测结果 |

---

## 5. 预期收益与风险

### 收益预测

| 优化 | 预期 ms | 累计 ms |
|---|---|---|
| 当前 baseline (E) | 41.0 | 41.0 |
| TRT engine (bf16) | -8-13 | **28-33 ms** ← 触达 30ms |
| TRT engine (fp16) | -2-3 (额外) | 26-30 ms (若 fp16 数值过关) |

### 主要风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **R1: ONNX export pi05 失败** | 中 | 致命 (Step 2 不通就整个 plan 死) | 三方案备选 (dynamo=False/True/from-pt2); 若全失败, 退到选项 C (新建 env) |
| **R2: ONNX 算子兼容性差** | 中 | 大 (要手动写 custom op) | opset 20+, 跳过 SDPA 用 eager attention; 必要时改 model code |
| **R3: TRT engine 5090 sm_120 支持** | 低 | 大 | TRT 10.14 官方支持 sm_120, 如失败试 TRT 10.15+ |
| **R4: trtexec 版本与 tensorrt Python 不匹配** | 中 | 小 | 系统 trtexec 是 10.10, phantom python 是 10.14。**fallback 全用 Python API** |
| **R5: phantom env 与其他用户/项目冲突** | 中 | 中 | 实施前 `who` 检查 + 备份 site-packages |
| **R6: 数值差异 > 1e-2** | 低 | 中 | 退到 fp32 TRT (慢但稳); 或保留 PyTorch fallback |
| **R7: TRT 推理仍 > 30ms** | 中-高 | 中 | 接受 33-35ms 作为最终 best, 评估是否真需 30ms |

### 回退路径

如果 plan 失败:
1. Step 2 失败 → 选项 C (新建 conda env, 3-5 天)
2. TRT 实测 ≥ 35ms → 接受 PyTorch 工具链 41ms 上限
3. 数值差异 > 1e-2 → 排查 ONNX dtype 路径, 或退 fp32

---

## 6. 验证标准 (final acceptance)

TRT 路线认为成功的条件:

| 标准 | 目标 |
|---|---|
| Mean 推理时间 | ≤ 35 ms (理想 ≤ 30 ms) |
| P99 / P50 抖动 | < 5 ms |
| 数值等价 | maxabs(out_trt - out_pytorch_E) < 1e-2 |
| Cold-start | < 30 秒 (engine 加载 < 10 秒) |
| 完整推理路径 | 100 次连续推理无 fault |

---

## 7. 实施步骤产物清单

落地 `optimize/` 下:

```
optimize/
├── TRT_30ms_PLAN.md               (本文件)
├── trt/                            (新建)
│   ├── export_pi05_onnx.py        Step 2 脚本
│   ├── build_trt_engine.sh        Step 3 trtexec 命令
│   ├── benchmark_pi05_trt.py      Step 4 推理 benchmark
│   ├── verify_numerical.py        Step 5 数值对比
│   └── README.md                   使用文档
└── results/
    ├── pi05.onnx                   ONNX 中间文件
    ├── pi05.engine                 TRT engine
    ├── trt_benchmark_<date>.md     TRT 实测报告
    └── numerical_diff_<date>.md    数值对比报告
```

---

## 8. 决策点 (待用户拍板)

| 决策点 | 选项 |
|---|---|
| **D1**: 走 phantom env (选项 A) 还是新建 env (选项 C)? | 推荐 A — phantom env 已是 stable PyTorch + sm_120 + TRT |
| **D2**: phantom env 修改 transformers patch 风险接受? | 需先 `who` 确认无人在用; 备份 site-packages |
| **D3**: 数值容差? bf16 default; 若不够再试 fp16/fp32 | bf16 推荐 (与现 E 一致) |
| **D4**: 失败 fallback 走哪? | C 新建 env, 或接受 41ms |

---

## 9. 修订历史

| 版本 | 时间 | 内容 |
|---|---|---|
| v0.1 | 2026-05-20 | 初版 plan, 利用 phantom env 现成 PyTorch 2.7.1+cu128 stable + TRT 10.14 |
