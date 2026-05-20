"""
Benchmark V1 Pi05Inference with a REAL deepdive_kai0 ckpt (loaded from converted .pkl).

Compares against V1 README's report:
  RTX 5090 3-view pi05 Triton: 34.2 ms (synthetic random weights)

Expected: real ckpt weights ≈ synthetic, since speed is shape-determined.

Usage:
    kai0/.venv_5090_trt/bin/python optimize/v1_triton/benchmark_kai0_v1.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \\
        --num-views 3 --chunk-size 50 --n-test 100
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_infer import Pi05Inference
from pi05_infer_tuned import Pi05InferenceTuned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="V1-format pkl from convert_kai0_to_v1.py")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--tokenizer-model", default=None,
                        help="If provided, run with discrete_state_input=True (deepdive_kai0 production path)")
    parser.add_argument("--tuned", action="store_true",
                        help="Use Pi05InferenceTuned (5090-tuned BLOCK_SIZE, -8.8% vs default)")
    args = parser.parse_args()

    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
    print(f"Loading checkpoint from {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        ckpt = pickle.load(f)
    print(f"  loaded {sum(v.numel()*v.element_size() for v in ckpt.values())/1e9:.2f} GB tensors")

    cls = Pi05InferenceTuned if args.tuned else Pi05Inference
    print(f"Building {cls.__name__}(num_views={args.num_views}, chunk_size={args.chunk_size}) ...")
    t0 = time.perf_counter()
    infer = cls(
        ckpt,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        discrete_state_input=False,
    )
    print(f"  build (incl CUDA Graph capture) in {time.perf_counter()-t0:.1f}s")

    # Dummy input — note: speed doesn't depend on input values, only shapes
    input_image = torch.randn(args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    input_noise = torch.randn(args.chunk_size, 32, dtype=torch.bfloat16, device="cuda")

    # Warm up
    print(f"Warm-up {args.n_warmup} iter ...")
    for _ in range(args.n_warmup):
        _ = infer.forward(input_image, input_noise)
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmark {args.n_test} iter ...")
    times = []
    for _ in range(args.n_test):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = infer.forward(input_image, input_noise)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print()
    print("=" * 60)
    print(f"[Pi05 Triton]: views={args.num_views} chunk_size={args.chunk_size}")
    print(f"  Mean: {arr.mean():.3f} ms")
    print(f"  Std:  {arr.std():.3f} ms")
    print(f"  P50:  {np.percentile(arr, 50):.3f} ms")
    print(f"  P95:  {np.percentile(arr, 95):.3f} ms")
    print(f"  P99:  {np.percentile(arr, 99):.3f} ms")
    print(f"  Min:  {arr.min():.3f} ms")
    print(f"  Max:  {arr.max():.3f} ms")
    print("=" * 60)
    print(f"  V1 README report (5090, 3 views): 34.2 ms (P50)")
    print(f"  PyTorch E max-autotune baseline:  41.0 ms (P50)")


if __name__ == "__main__":
    main()
