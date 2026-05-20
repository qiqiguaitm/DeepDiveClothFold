# Pi0.5 推理优化 — 目标 30ms

> 起点: backend E (`torch.compile(max-autotune)`) 在 5090 上 P50 = 43.6 ms (bf16, 100 iter)
> **目标: 30 ms (-32%), 模型架构不变, 不重训, 不量化**

---

## 约束

| 约束 | 说明 |
|---|---|
| 模型架构不变 | PI0Pytorch / PaliGemma / Action Expert 结构原样 |
| 推理结果不变 | 输出 action chunk 在数值容差内一致 (<1e-3 absdiff) |
| 不允许重训练 | 包括蒸馏、LoRA-fine-tune、prune-retrain 全部排除 |
| 不允许量化训练 | 排除 QAT (Quantization-Aware Training) |
| 不做 int8 / int4 量化 | post-training int8 也排除 |
| **可以做** | fp16/bf16 切换、kernel fusion、CUDA Graph、TRT、AOTI、kernel autotune |

---

## 当前 baseline (5 backend)

| Backend | Mean (ms) | P50 | Speedup |
|:---:|---:|---:|---:|
| A eager | 256.5 | 256.5 | 1.00× |
| B compile-default | 92.4 | 89.2 | 2.78× |
| C cuda-graph (manual) | 60.7 | 60.1 | 4.22× |
| D compile-reduce-overhead | 48.3 | 48.1 | 5.31× |
| **E** compile-max-autotune | **43.6** | **43.5** | **5.88×** |

---

## 优化候选清单 (按 ROI 排序)

### F. coordinate_descent_tuning (E 之上)

**原理**: Inductor `coordinate_descent_tuning=True` 在 max-autotune 选定的 kernel 上做坐标下降 fine-tune (单变量轮转优化 BLOCK_M/N/K/num_stages/num_warps)。比 max-autotune 的"21 个固定配置 sweep"再深一层。

**改动**:
```python
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
# 同时保留 max-autotune
```

**预期**: -5-10% (43.6 → 39-41 ms)
**风险**: 编译时间增加 2-3× (autotune sweep 时间 ×N)
**实施成本**: 1 行 config + 重测

### G. fp16 替代 bf16

**原理**: 5090 (Blackwell) 上 fp16 在 cuBLAS / Triton 上比 bf16 略快 (~10-15%), 因为:
- fp16 Tensor Core 历史更久, 优化更深
- bf16 在某些 GEMM 上仍走通用路径
- fp16 vs bf16 同样是 16-bit, 精度方面 fp16 范围小但 mantissa 多 (适合 inference; bf16 范围大适合 training)

**数值风险**: fp16 max = 65504. pi05 中间激活值通常 < 100, 但 RMSNorm 内部如果用 `sum(x^2)` 可能溢出 (实际 transformers 实现里会 upcast)。需要测试。

**改动**:
```python
model = PI0Pytorch(config).to(device).to(torch.float16).eval()
# patches 改用 fp16
```

**预期**: -10-20% (43.6 → 35-39 ms)
**风险**: 数值不变约束需要测 — 与 bf16 ckpt 对比 maxabs diff
**实施成本**: 改 dtype + 全 patch 路径过一遍

### H. AOTInductor (export + AOT 编译 .so)

**原理**: `torch._inductor.aoti_compile_and_package` 把模型导出为独立 .so + .pt2 包, 推理时直接加载, 完全 bypass:
- TorchDynamo trace (运行时无)
- Python 字节码解释
- 仍保留 Inductor fusion + CUDA Graph (编译时静态生成)

类似 TorchScript 的"AOT 编译"模式, 但用 Inductor 生成的 Triton kernel 而不是旧的 TorchScript runtime。

**预期**: -5-10% (43.6 → 39-41 ms) — 主要来自消除 dispatch overhead
**风险**: 中等 — pi05 含 flow matching 循环, export 时要展开成静态图 (10 步 × ... )
**实施成本**: 中 (需要 torch.export trace + 包装)

### I. TensorRT 集成

**原理**: PyTorch → ONNX/torch.export → TensorRT engine。TRT 优势:
- 比 Inductor 更激进的 kernel fusion (FP16 + Tensor Core 路径优化)
- 自动 layer fusion (Conv-BN-ReLU, MultiHeadAttention 整体融合)
- 比 Inductor 更深的 GEMM autotune (cuBLAS, cuDNN, custom kernels 同时 sweep)

**实施路径**:
- (a) `torch_tensorrt.compile(model, ...)` — PyTorch model 直接走 TRT
- (b) `torch.export.export(...)` → ONNX → `trtexec` 转 engine → 用 polygraphy / tensorrt Python runtime

**预期**: -20-30% (43.6 → 30-35 ms) ✅ **触达 30ms 目标的最可能路径**
**风险**: 中 — TRT-PyTorch nightly 兼容性 (TRT 通常 lag 主流 PyTorch 1-3 月)
**实施成本**: 中-高

### J. CUTLASS GEMM backend (Inductor max_autotune_gemm_backends)

**原理**: 默认 Inductor max-autotune 用 Triton 模板; 加 CUTLASS GEMM backend 让它同时 sweep CUTLASS 的高度优化 kernel:
```python
torch._inductor.config.max_autotune_gemm_backends = "TRITON,CUTLASS"
```

**预期**: -5-10% (43.6 → 39-41 ms)
**风险**: 低 (失败时 fallback Triton)
**实施成本**: 1 行 config

---

## 实施顺序 (建议)

```
阶段 1 (低成本探底, 1-2 hr):
  F + J + G (config 调优 + fp16) → 看能否到 35-38ms
       │
       ↓
阶段 2 (中等成本, 半天):
  H AOTInductor → 看能否到 35ms 以下
       │
       ↓
阶段 3 (硬骨头, 1-2 天):
  I TensorRT → 期望触达 30ms
```

---

## 测试 / 验证标准

每个新 backend 必须满足:
1. **速度**: P50 < 当前 baseline (E=43.5ms)
2. **数值等价**: max-abs(output_new - output_E) < 1e-3 (相对 1e-2)
3. **稳定性**: P99-P50 < 5ms (CUDA Graph 量级)
4. **可重现**: 100 iter 跑 3 次, Mean 变动 < 5%

每个新 backend 加入 `benchmark_pi05_inference_v2.py`, 与 ABCDE 一起对比, 输出统一报告。

---

## 风险与回退

| 假设破裂 | 回退路径 |
|---|---|
| fp16 数值偏差 > 1e-2 | 退回 bf16, 跳过 G |
| AOTI export pi05 flow loop 不能 trace | 用 torch.compile + 手动 CUDA Graph 二合一替代 H |
| TRT-PyTorch nightly 不兼容 | 走 ONNX → trtexec 两阶段路径; 或退到 30ms 不到的 best result |
| coordinate_descent 编译时间 >10 min | 接受 once-only, 缓存 Inductor cache, 部署时复用 |

---

## 修订历史

| 版本 | 时间 | 内容 |
|---|---|---|
| v0.1 | 2026-05-19 | 初版 plan, 5 个优化候选 (F-J), 目标 30ms |
