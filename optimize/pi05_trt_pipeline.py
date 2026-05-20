"""
pi0.5 推理 TRT 全流程: PI0Pytorch → ONNX → TRT engine → benchmark

Env: kai0/.venv_5090_trt (Python 3.10 + torch 2.7.1+cu128 + tensorrt 10.14)

5 个阶段:
  1) Build PI0Pytorch with bf16 patches (复用 benchmark_pi05_inference.py 的 patches)
  2) torch.onnx.export → pi05.onnx
  3) tensorrt Python API → pi05.engine (bf16, opt-level 5)
  4) TRT runtime 100-iter benchmark
  5) 数值对比 (TRT vs PyTorch eager, maxabs < 1e-2)

Usage:
  cd /home/tim/workspace/deepdive_kai0
  kai0/.venv_5090_trt/bin/python optimize/pi05_trt_pipeline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_KAI0_SRC = _REPO_ROOT / "kai0" / "src"
sys.path.insert(0, str(_KAI0_SRC))
sys.path.insert(0, str(_HERE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger(__name__)

# Reuse model build + patches from benchmark script
from benchmark_pi05_inference import (
    build_model,
    make_dummy_observation,
    ACTION_HORIZON,
    ACTION_DIM,
    NUM_STEPS,
    MAX_TOKEN_LEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper: tensor-only IO module for ONNX export
# ─────────────────────────────────────────────────────────────────────────────

class Pi05Wrapper(nn.Module):
    """Wraps PI0Pytorch.sample_actions in tensor-only IO."""
    INPUT_NAMES = [
        "image_base", "image_left", "image_right",
        "tokenized_prompt", "tokenized_prompt_mask",
        "token_ar_mask", "token_loss_mask", "state", "noise",
    ]
    OUTPUT_NAMES = ["actions"]

    def __init__(self, base_model, num_steps=NUM_STEPS):
        super().__init__()
        # Avoid registering as submodule (keep .state_dict() small for ONNX)
        object.__setattr__(self, "_base", base_model)
        self.num_steps = num_steps

    def forward(self, image_base, image_left, image_right,
                tokenized_prompt, tokenized_prompt_mask,
                token_ar_mask, token_loss_mask, state, noise):
        class _Obs: pass
        o = _Obs()
        o.images = {
            "base_0_rgb": image_base,
            "left_wrist_0_rgb": image_left,
            "right_wrist_0_rgb": image_right,
        }
        o.image_masks = {
            "base_0_rgb": torch.ones(image_base.shape[0], dtype=torch.bool, device=image_base.device),
            "left_wrist_0_rgb": torch.ones(image_left.shape[0], dtype=torch.bool, device=image_left.device),
            "right_wrist_0_rgb": torch.ones(image_right.shape[0], dtype=torch.bool, device=image_right.device),
        }
        o.tokenized_prompt = tokenized_prompt
        o.tokenized_prompt_mask = tokenized_prompt_mask
        o.token_ar_mask = token_ar_mask
        o.token_loss_mask = token_loss_mask
        o.state = state
        return self._base.sample_actions(image_base.device, o, noise=noise, num_steps=self.num_steps)


def make_args_tuple(obs, device, dtype):
    noise = torch.zeros(1, ACTION_HORIZON, ACTION_DIM, dtype=dtype, device=device)
    return (
        obs.images["base_0_rgb"],
        obs.images["left_wrist_0_rgb"],
        obs.images["right_wrist_0_rgb"],
        obs.tokenized_prompt,
        obs.tokenized_prompt_mask,
        obs.token_ar_mask,
        obs.token_loss_mask,
        obs.state,
        noise,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build model
# ─────────────────────────────────────────────────────────────────────────────

def stage1_build_model(device):
    log.info("=" * 60)
    log.info("Stage 1: Build PI0Pytorch + patches")
    log.info("=" * 60)
    t0 = time.perf_counter()
    model = build_model(device)
    obs = make_dummy_observation(device, batch=1)
    wrapper = Pi05Wrapper(model).to(device).eval()
    # ONNX export needs requires_grad=False on all params (otherwise tracer
    # refuses to fold weights into constants)
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"  Build OK in {time.perf_counter()-t0:.1f}s")
    log.info(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M (all requires_grad=False)")
    return wrapper, obs, model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def stage2_onnx_export(wrapper, obs, device, dtype, onnx_path: Path):
    log.info("=" * 60)
    log.info("Stage 2: ONNX export")
    log.info("=" * 60)
    args = make_args_tuple(obs, device, dtype)

    log.info(f"  Input shapes:")
    for n, t in zip(Pi05Wrapper.INPUT_NAMES, args):
        log.info(f"    {n}: {tuple(t.shape)} {t.dtype}")

    t0 = time.perf_counter()
    log.info(f"  Calling torch.onnx.export (dynamo=True, more memory-efficient for large models)...")
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, args, str(onnx_path),
                input_names=Pi05Wrapper.INPUT_NAMES,
                output_names=Pi05Wrapper.OUTPUT_NAMES,
                opset_version=17,
                dynamo=True,
                external_data=True,  # weights stored separately in *.onnx_data
                optimize=True,
            )
    except Exception as e:
        log.exception(f"ONNX export FAILED: {e}")
        raise

    size_mb = onnx_path.stat().st_size / 1e6
    log.info(f"  ONNX export OK in {time.perf_counter()-t0:.1f}s ({size_mb:.1f} MB)")
    return onnx_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: TRT engine build
# ─────────────────────────────────────────────────────────────────────────────

def stage3_trt_build(onnx_path: Path, engine_path: Path, opt_level=5):
    log.info("=" * 60)
    log.info(f"Stage 3: TRT engine build (bf16, opt-level={opt_level})")
    log.info("=" * 60)
    import tensorrt as trt
    log.info(f"  tensorrt {trt.__version__}")

    t0 = time.perf_counter()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(f"  parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = opt_level

    log.info("  Building serialized network (this can take 5-30 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build_serialized_network returned None")

    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    size_mb = engine_path.stat().st_size / 1e6
    log.info(f"  TRT engine OK in {time.perf_counter()-t0:.1f}s ({size_mb:.1f} MB)")
    return engine_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: TRT runtime benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _trt_to_torch_dtype(trt_dtype):
    import tensorrt as trt
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }.get(trt_dtype, torch.float32)


def stage4_trt_benchmark(engine_path: Path, obs, device, dtype, n_warmup=10, n_test=100):
    log.info("=" * 60)
    log.info(f"Stage 4: TRT runtime benchmark (n_test={n_test})")
    log.info("=" * 60)
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    n_io = engine.num_io_tensors
    io_names = [engine.get_tensor_name(i) for i in range(n_io)]
    log.info(f"  TRT engine I/O tensors: {io_names}")

    # Allocate buffers on device, matching input shapes
    bindings = {}
    for name in io_names:
        shape = engine.get_tensor_shape(name)
        td = _trt_to_torch_dtype(engine.get_tensor_dtype(name))
        t = torch.empty(tuple(shape), dtype=td, device=device)
        bindings[name] = t
        context.set_tensor_address(name, t.data_ptr())

    # Fill inputs from obs
    args = make_args_tuple(obs, device, dtype)
    for name, val in zip(Pi05Wrapper.INPUT_NAMES, args):
        if name in bindings:
            bindings[name].copy_(val.to(bindings[name].dtype))
        else:
            log.warning(f"  Input '{name}' not in TRT engine I/O (got {io_names})")

    # Warm-up
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(n_warmup):
        context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    # First-call (timing including any first-fault)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000
    log.info(f"  First post-warmup call: {first_ms:.2f} ms")

    # 100-iter benchmark
    times = []
    for _ in range(n_test):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    stats = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "first_call_ms": first_ms,
    }
    log.info(f"  TRT benchmark: mean={stats['mean']:.2f}ms p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms p99={stats['p99']:.2f}ms")
    return stats, bindings


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Numerical comparison
# ─────────────────────────────────────────────────────────────────────────────

def stage5_numerical_compare(wrapper, obs, device, dtype, trt_bindings):
    log.info("=" * 60)
    log.info("Stage 5: Numerical comparison TRT vs PyTorch eager")
    log.info("=" * 60)
    args = make_args_tuple(obs, device, dtype)
    with torch.inference_mode():
        out_torch = wrapper(*args)

    # TRT output is in trt_bindings["actions"]
    if "actions" not in trt_bindings:
        log.error(f"  'actions' not in TRT bindings: {list(trt_bindings)}")
        return
    out_trt = trt_bindings["actions"]

    diff = (out_trt.float() - out_torch.float()).abs()
    log.info(f"  PyTorch output: shape={tuple(out_torch.shape)} dtype={out_torch.dtype}")
    log.info(f"  TRT output:     shape={tuple(out_trt.shape)} dtype={out_trt.dtype}")
    log.info(f"  maxabs: {diff.max().item():.4e}")
    log.info(f"  mean abs: {diff.mean().item():.4e}")
    log.info(f"  out_torch mean abs: {out_torch.abs().mean().item():.4e}")
    rel = diff.max().item() / out_torch.abs().mean().item()
    log.info(f"  relative maxabs: {rel:.4%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(_HERE / "results" / "pi05.onnx"))
    parser.add_argument("--engine", default=str(_HERE / "results" / "pi05.engine"))
    parser.add_argument("--opt-level", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--skip-onnx", action="store_true", help="Skip Stage 2 (reuse existing ONNX)")
    parser.add_argument("--skip-build", action="store_true", help="Skip Stage 3 (reuse existing engine)")
    args = parser.parse_args()

    onnx_path = Path(args.onnx); onnx_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path = Path(args.engine); engine_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    log.info(f"Hardware: {torch.cuda.get_device_name(device)}")
    log.info(f"torch {torch.__version__}, cuda {torch.version.cuda}")

    wrapper, obs, model = stage1_build_model(device)

    if not args.skip_onnx:
        stage2_onnx_export(wrapper, obs, device, dtype, onnx_path)
    else:
        log.info(f"Skipping Stage 2, reusing {onnx_path}")

    if not args.skip_build:
        stage3_trt_build(onnx_path, engine_path, opt_level=args.opt_level)
    else:
        log.info(f"Skipping Stage 3, reusing {engine_path}")

    stats, bindings = stage4_trt_benchmark(engine_path, obs, device, dtype,
                                            n_warmup=args.n_warmup, n_test=args.n_test)
    stage5_numerical_compare(wrapper, obs, device, dtype, bindings)

    log.info("=" * 60)
    log.info("FINAL: pi0.5 TRT benchmark complete")
    log.info(f"  Mean: {stats['mean']:.2f}ms  P50: {stats['p50']:.2f}ms  P99: {stats['p99']:.2f}ms")
    log.info(f"  vs E max-autotune baseline (41.0 ms)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
