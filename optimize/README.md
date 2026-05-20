# pi0.5 Inference Optimization Benchmarks

> 配套文档:
> - [realtime_vla_optimization_analysis.md](../docs/deployment/realtime_vla_optimization_analysis.md) — 主分析文档
> - [pi05_inference_backend_benchmark_plan.md](../docs/deployment/pi05_inference_backend_benchmark_plan.md) — 测试 plan

本目录用于 deepdive_kai0 推理优化路线 (主分析文档 §3.4.2) 的实测脚本与结果。当前包含 **5 backend** 推理速度对比。

---

## 文件结构

```
optimize/
├── README.md                       # 本文件
├── benchmark_pi05_inference.py     # 主脚本: 5 backend 推理速度对比
├── cuda_graph_wrapper.py           # 手动 CUDA Graph 工具类 (backend C 的实现)
└── results/                        # 输出报告 (gitignored)
    └── pi05_inference_<host>_<date>.md
```

---

## 5 个 Backend 对比设计

| 编号 | 实现 | Inductor fusion | CUDA Graph | autotune |
|:---:|---|:---:|:---:|---|
| **A** | PyTorch eager (手动解 compile) | ❌ | ❌ | — |
| **B** | `torch.compile(mode="default")` | ✅ | ❌ | min |
| **C** | 手动 `torch.cuda.CUDAGraph` (eager kernel) | ❌ | ✅ | — |
| **D** | `torch.compile(mode="reduce-overhead")` | ✅ | ✅ (auto) | std |
| **E** | `torch.compile(mode="max-autotune")` | ✅ | ✅ (auto) | **max** (deepdive_kai0 默认) |

**B vs A** = Inductor fusion 单独的收益
**C vs A** = CUDA Graph 单独的收益
**D vs A** = fusion + CUDA Graph 复合收益
**E vs D** = max-autotune 相对 reduce-overhead 的边际收益
**E vs A** = deepdive_kai0 当前架构相对纯 eager 的总收益

---

## 重要约定 / 决策记录

### 1. 权重: 随机初始化 (非真实 ckpt)

JAX→PI0Pytorch state_dict 转换工具**尚不存在** (deepdive_kai0 训练侧用 PyTorch safetensors 加载, 没有从 JAX orbax 转换的路径; V1 的 `convert_from_jax_pi05.py` 转的是 V1 自己的 flat weights 字典, 不兼容 PI0Pytorch)。

**完整 JAX→PI0Pytorch 转换工具** (≥ 1-2 天工程 + 数值对齐) 推迟到主分析文档 §3.4.1 (PyTorch 训练等效性 POC) 一并实现。

本 benchmark 只关心**推理速度** (架构决定, 与权重数值无关), 所以用随机初始化即可。

### 2. Backend E (max-autotune) 是 deepdive_kai0 当前实际默认

看 `kai0/src/openpi/models_pytorch/pi0_pytorch.py:113`:

```python
self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
```

即 `PI0Pytorch.__init__` 默认就把 `sample_actions` 用 `torch.compile(max-autotune)` 包了。这意味着:

- 测试 A (eager) 必须**显式解除** compile (本脚本 `restore_eager_sample_actions(model)` 做这件事)
- E 是 deepdive_kai0 现状, 提供"用户实际看到的速度"基线
- 与 plan v0.1 的 4 backend 不同, **本实现是 5 backend**

### 3. 硬件 = sim01 (2× RTX 5090 32GB)

直接对应主分析文档 §3.4.2 决策 (策略 A 手写 Triton 全栈 vs 策略 B torch.compile)。

### 4. batch_size = 1, 完整推理 (含 10 步 denoising)

与真机部署 (ROS2 policy_inference_node, 单帧推理) 一致。

---

## 运行方式

### 准备环境

需在 `kai0/.venv` (Python 3.12 + JAX 0.5.3 + torch 2.7.1+cu126 + transformers 4.53.2) 下跑。

**特别注意**: `pi0_pytorch.py` 顶部有这个 import check:

```python
from transformers.models.siglip import check
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise ValueError(msg)
```

这要求 `kai0/src/openpi/models_pytorch/transformers_replace/*` 已经覆盖到 `.venv/lib/python3.12/site-packages/transformers/`。若没装, 第一次跑会失败。

修复:
```bash
cd /home/tim/workspace/deepdive_kai0/kai0
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.12/site-packages/transformers/
```

### Dry-run (smoke test, 每 backend 跑 1 次推理)

```bash
cd /home/tim/workspace/deepdive_kai0
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py --dry-run
```

这会:
- 构造 PI0Pytorch with random weights
- 对每个 backend 跑 2 次推理 (warm-up 1 + test 1)
- 报告每个 backend 是否能跑通, 不做精细计时

用于先验证脚本可以跑, 各 backend 没有崩 (例如 CUDA Graph 是否能 capture, compile 是否能 fullgraph)。

### 完整 benchmark

```bash
cd /home/tim/workspace/deepdive_kai0
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py
# 默认: 5 backend × (10 warm-up + 100 timed iterations)
```

预计耗时:
- 首次模型构造: 30-60 s
- A eager: 100 × ~200ms = 20s
- B compile-default: 首次 compile ~60-120s, 之后 100 × ~150ms = 15s
- C cuda-graph: capture ~10s, 之后 100 × ~50ms = 5s
- D reduce-overhead: 首次 compile ~120-180s, 之后 100 × ~50ms = 5s
- E max-autotune: 首次 compile ~5-15 min (max-autotune 搜索 Triton 配置, 慢), 之后 100 × ~50ms = 5s

**总耗时: 约 10-25 min** (max-autotune 编译占大头)

### 只跑部分 backend

```bash
# 只跑 A B C (跳过 max-autotune, 节省 ~10 min)
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py --backends A,B,C

# 单测 E (deepdive_kai0 当前默认), 看绝对值
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py --backends E --n-test 200
```

### 自定义参数

```bash
# 多测点
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py --n-warmup 20 --n-test 200

# 不同 denoising step
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py --num-steps 4   # 快速模式
kai0/.venv/bin/python optimize/benchmark_pi05_inference.py --num-steps 20  # 高精度模式
```

---

## 结果解读

输出在终端 + markdown 报告 (`results/pi05_inference_<hostname>_<date>.md`)。

### 决策映射 (主分析文档 §3.4.2)

| E backend P50 推理时间 | 推断 |
|---|---|
| < 60 ms | **策略 B 够用** — torch.compile(max-autotune) 已饱和, 不需要手写 Triton |
| 60-100 ms | 策略 B 先上, 实战不够再升级策略 A |
| > 150 ms | **必须策略 A** — V1 手写 Triton 全栈端口 |
| E 编译失败 / graph break | 走策略 A (因为 B 无法生效) |

### 分量诊断

- **B - A 显著正 (例如 -20%)** → Inductor fusion 在 pi05 上有效, kernel fusion 是收益来源之一
- **C - A 显著正 (例如 -40%)** → CUDA Graph 在 pi05 上是大头, V1 的 4.2.1 策略对 pi05 成立
- **D ≈ B + C 复合** → 两者正交叠加成功
- **E ≤ D** → max-autotune 没拿到额外收益 (autotune 搜索空间已被 reduce-overhead 覆盖)
- **E > D** → max-autotune 的额外 autotune 时间值得

---

## 已知风险 + 调试建议

| 症状 | 原因 | 修复 |
|---|---|---|
| `from transformers.models.siglip import check` 失败 | transformers_replace 未安装 | 见上方"准备环境"步骤 |
| backend D/E `fullgraph=True` 失败 | sample_actions 有 Python control flow (`while time >= -dt / 2:`) | 脚本自动 fallback `fullgraph=False`, 记入报告 |
| backend C CUDA Graph capture 失败 | sample_actions 内部有 host sync (`.item()` / `.cpu()`) | 检查 `denoise_step` / `embed_prefix` 是否纯 device-side; 5090 + PyTorch 2.7 应该 OK |
| OOM | 第一次模型构造 + 5 backend 串跑 | 加 `--backends X` 单测 |
| max-autotune 编译过慢 | 正常 | 等; 或 `--backends A,B,C,D` 跳过 E |
| 推理速度比预期慢很多 | XLA cache 未命中 / dtype 错乱 | 看 `dry-run` 输出验证模型 dtype + GPU memory |

---

## 下一步

完成 benchmark 后:
1. 把 `results/*.md` 反馈到 [`realtime_vla_optimization_analysis.md`](../docs/deployment/realtime_vla_optimization_analysis.md) §3.4.2
2. 决定阶段 3.4.2 走策略 A 还是策略 B
3. 后续可加 NVTX profile (用 `nsys profile`) 看 kernel 数量分布, 进一步验证 fusion 效果
