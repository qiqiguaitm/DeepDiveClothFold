"""
Minimal smoke test for phantom env's ONNX → TRT → 5090 workflow.

Goal: in < 10 minutes verify whether this combination CAN run:
  - PyTorch 2.7.1+cu128 (phantom env's torch)
  - tensorrt 10.14 (phantom env's trt)
  - 5090 sm_120 hardware

Uses a tiny dummy model (no pi05 complexity) to isolate the env validation
from model-specific issues like transformers version mismatch.

If this smoke test PASSES, the TRT path is viable — we'd then build our own
kai0-owned env with same package recipe.
If it FAILS, we know upfront not to invest in phantom-style setup.

Usage:
    /data1/miniconda3/envs/phantom/bin/python optimize/trt_smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

print(f"Python {sys.version_info[:2]}")
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"sm_120 supported: {'sm_120' in torch.cuda.get_arch_list()}")
print(f"device: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────────────────
# Dummy transformer-flavor model (a tiny stand-in for pi05)
# 768-dim, 1 attention block, but uses same ops as pi05:
#   - Linear projections (Q/K/V/O)
#   - SDPA attention
#   - SiLU activation (Gated MLP)
#   - RMSNorm-style norm (using LayerNorm as a stand-in to avoid pi05 complexity)
# ──────────────────────────────────────────────────────────

class TinyTransformerBlock(nn.Module):
    """Single decoder-style block: norm → SDPA → norm → Gated MLP → residual"""
    def __init__(self, dim=768, n_heads=12, mlp_hidden=2048):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(dim, mlp_hidden, bias=False)
        self.down_proj = nn.Linear(mlp_hidden, dim, bias=False)

    def forward(self, x):
        # Attention sub-block
        h = self.norm1(x)
        b, n, d = h.shape
        q = self.q_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = x + self.o_proj(attn)

        # Gated MLP sub-block
        h = self.norm2(x)
        x = x + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class TinyModel(nn.Module):
    """Stack of 4 transformer blocks — small enough to compile fast."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyTransformerBlock() for _ in range(4)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    model = TinyModel().to(device).to(dtype).eval()
    batch, seq, dim = 1, 256, 768
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # ─────────────────────────────────────────
    # Phase 1: torch.export
    # ─────────────────────────────────────────
    print("\n=== Phase 1: torch.export ===")
    t0 = time.perf_counter()
    try:
        ep = torch.export.export(model, (x,), strict=False)
        print(f"  ✓ torch.export OK ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ torch.export FAILED: {type(e).__name__}: {e}")
        return 1

    # ─────────────────────────────────────────
    # Phase 2: ONNX export
    # ─────────────────────────────────────────
    print("\n=== Phase 2: ONNX export ===")
    onnx_path = "/tmp/trt_smoke_test_model.onnx"
    t0 = time.perf_counter()
    try:
        # phantom env's torch 2.7.1 supports both dynamo and legacy paths
        torch.onnx.export(
            model, (x,), onnx_path,
            opset_version=17,
            input_names=["x"], output_names=["out"],
            do_constant_folding=True,
            dynamo=False,  # legacy tracer, more permissive
        )
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  ✓ ONNX export OK ({time.perf_counter() - t0:.1f}s, {size_mb:.1f}MB)")
    except Exception as e:
        print(f"  ✗ ONNX export FAILED: {type(e).__name__}: {e}")
        return 1

    # ─────────────────────────────────────────
    # Phase 3: TRT engine build
    # ─────────────────────────────────────────
    print("\n=== Phase 3: TRT engine build ===")
    engine_path = "/tmp/trt_smoke_test_model.engine"
    t0 = time.perf_counter()
    try:
        import tensorrt as trt
        print(f"  tensorrt {trt.__version__}")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  parse error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.BF16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = 3  # 1-5; use 3 for fast smoke test

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build returned None")
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized))
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"  ✓ TRT engine OK ({time.perf_counter() - t0:.1f}s, {size_mb:.1f}MB)")
    except Exception as e:
        print(f"  ✗ TRT engine FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    # ─────────────────────────────────────────
    # Phase 4: TRT runtime inference
    # ─────────────────────────────────────────
    print("\n=== Phase 4: TRT runtime inference ===")
    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        if engine is None:
            raise RuntimeError("Deserialize returned None")
        context = engine.create_execution_context()

        # Allocate I/O buffers on GPU
        # In TRT 10.x, we use IO tensor names instead of indices
        n_io = engine.num_io_tensors
        io_names = [engine.get_tensor_name(i) for i in range(n_io)]
        io_shapes = {name: engine.get_tensor_shape(name) for name in io_names}
        io_modes = {name: engine.get_tensor_mode(name) for name in io_names}
        print(f"  I/O tensors: {io_names}")
        print(f"  shapes: {io_shapes}")

        # Allocate — map TRT dtype directly to torch dtype (numpy can't represent bf16)
        TRT_TO_TORCH = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        bindings = {}
        for name in io_names:
            shape = io_shapes[name]
            trt_dtype = engine.get_tensor_dtype(name)
            torch_dtype = TRT_TO_TORCH.get(trt_dtype, torch.float32)
            t = torch.empty(tuple(shape), dtype=torch_dtype, device="cuda")
            bindings[name] = t
            context.set_tensor_address(name, t.data_ptr())

        # Fill input
        bindings["x"].copy_(x.to(bindings["x"].dtype))

        # Warm-up
        for _ in range(5):
            context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Benchmark 100 iters
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f"  ✓ TRT inference OK: mean={arr.mean():.3f}ms p50={np.percentile(arr, 50):.3f}ms p99={np.percentile(arr, 99):.3f}ms")

        # Numerical comparison
        out_trt = bindings["out"].clone()
        out_torch = model(x)
        diff = (out_trt.float() - out_torch.float()).abs().max().item()
        rel = diff / out_torch.abs().mean().item()
        print(f"  TRT vs PyTorch maxabs diff: {diff:.4e}  (rel {rel:.4e})")
    except Exception as e:
        print(f"  ✗ TRT runtime FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("✓ phantom env ONNX → TRT → 5090 workflow VERIFIED")
    print("=" * 60)
    print("\nNext: build a kai0-owned env with this package recipe:")
    print("  - torch 2.7.1+cu128")
    print("  - tensorrt 10.14")
    print("  - transformers 4.53.2 (+ transformers_replace patch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
