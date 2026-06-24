#!/usr/bin/env python3
"""R6 — measure v1 (Triton, 5090-tuned) per-decode-step latency → FLASH 组合加速经济模型.

R6 的全部价值押在一个数上: **v1 在 5090 上每个 flow-matching 去噪步的墙钟成本**。
- v1 整条前向 = 单次 CUDA-graph replay, 内含 encoder(prefill) + 固定 10 步 Euler 解码 (P50≈32ms)。
- FLASH 组合加速 = 用 draft 一次性给出整条 chunk, 再用 **K 步** (而非 10 步) verify; 接受则省下
  (10-K) 步。要算这个省了多少, 必须把 32ms **拆成** "encoder 截距 a + 每步斜率 b"。
- v2 (FLASH-on-eager) 之所以**没快过 v1**: eager 每步 ~50ms ≫ v1 每步 b; 步数省了也被每步成本吃掉。
  R6 = FLASH 步数省 × v1 每步成本 → 才是真组合加速。

做法 (纯加性, **不改** pi05_infer_tuned.py): 用现成 `Pi05InferenceTuned` 装好权重/buffer, 然后对
num_steps ∈ steps 各自**另起一张 CUDA graph** 调 `pi05_model_tuned(..., num_steps=K)` (style/time
buffer 在 load 时已为 10 步预算好, K≤10 直接复用), 计时 replay。T(K)=a+b·K 最小二乘 → a=encoder/
prefill, b=每解码步。再用 a,b + draft 成本 (R1-b 实测 eager draft≈2.5ms; R6 真实 draft 复用 v1
encoder prefix 只多一层 Gemma block, 量级相同) 组装 R6 延迟模型 + 不同接受率下的期望加速。

注意: 这里只测**延迟** (数值是否正确无关, 故输入填随机有限值即可); verify 的正确性/接受率走
spec_pi0_pytorch + R1-d draft 那条线, 不在本脚本。

Run:
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090_trt/bin/python optimize/v1_triton/bench_v1_per_step_5090.py \
    --pkl /data1/DATA_IMP/checkpoints/ckpt_v1/pytorch_pure200_step50000/v1_p200.pkl \
    --steps 1,2,4,6,8,10 --iters 100 --warmup 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pi05_infer import transformer_encoder  # noqa: E402  (18-layer VLM prefill → KV; FLASH 在 draft 轮跳过)
from pi05_infer import vision_encoder  # noqa: E402  (SigLIP; draft 轮仍需, 喂 draft 的 prefix_embs)
from pi05_infer_tuned import Pi05InferenceTuned  # noqa: E402
from pi05_infer_tuned import pi05_model_tuned  # noqa: E402


def _prime_input_buffers(infer) -> None:
    """填好 encoder/decoder 输入 buffer (prebaked language_embeds 路径, discrete_state_input=False),
    避免未初始化/NaN 在 matmul 里传播影响计时稳定性。复刻 forward() 的 buffer-copy 头部。"""
    b, w = infer.buffers, infer.weights
    prompt_embeds = w["language_embeds"]
    prompt_len = int(prompt_embeds.shape[0])
    start = infer.num_views * 256
    b["encoder_x"][start : start + prompt_len].copy_(prompt_embeds)
    b["valid_encoder_len"].fill_(start + prompt_len)
    b["decoder_rope_weights"].copy_(infer.get_decoder_rope_weights(prompt_len))
    b["observation_images_normalized"].copy_(
        torch.randn_like(b["observation_images_normalized"])
    )
    b["diffusion_noise"].copy_(torch.randn_like(b["diffusion_noise"]))


def _capture_graph(infer, num_steps: int) -> torch.cuda.CUDAGraph:
    """对给定 num_steps 另起一张 graph (不动 infer.infer_graph)。warmup 3 次再 capture。"""
    for _ in range(3):
        pi05_model_tuned(infer.weights, infer.buffers, infer.num_views, infer.encoder_seq_len, num_steps)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g.capture_begin()
        pi05_model_tuned(infer.weights, infer.buffers, infer.num_views, infer.encoder_seq_len, num_steps)
        g.capture_end()
    torch.cuda.synchronize()
    return g


def _capture_stage_graph(infer, fn) -> torch.cuda.CUDAGraph:
    """对任意 stage-subset 回调 fn(infer) 另起一张 graph 计时 (vision-only / vision+encoder)。"""
    for _ in range(3):
        fn(infer)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g.capture_begin()
        fn(infer)
        g.capture_end()
    torch.cuda.synchronize()
    return g


def _time_graph(g: torch.cuda.CUDAGraph, iters: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts[i] = s.elapsed_time(e)  # ms
    return ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="v1 pickle (convert_kai0_to_v1.py 产物)")
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--steps", default="1,2,4,6,8,10", help="逗号分隔的 num_steps 采样点")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--draft-ms", type=float, default=2.5,
                    help="draft 一次前向墙钟 (R1-b 实测 eager≈2.5ms; R6 真实 draft 复用 v1 prefix 量级相同)")
    ap.add_argument("--accept-overhead-ms", type=float, default=0.5,
                    help="radius accept + stitch + 夹爪门控 (grid=1 kernel, 量级 0.x ms)")
    ap.add_argument("--verify-k", default="2,3", help="逗号分隔的 verify 步数候选")
    args = ap.parse_args()

    steps = [int(x) for x in args.steps.split(",")]
    assert max(steps) <= 10, "v1 style/time buffer 仅为 10 步预算, num_steps 须 ≤10"

    print(f"[load] {args.pkl}")
    t0 = time.perf_counter()
    with open(args.pkl, "rb") as f:
        ckpt = pickle.load(f)
    infer = Pi05InferenceTuned(ckpt, num_views=args.num_views, chunk_size=args.chunk_size,
                               discrete_state_input=False)
    print(f"[load] build+capture(10) in {time.perf_counter()-t0:.1f}s  encoder_seq_len={infer.encoder_seq_len}")
    _prime_input_buffers(infer)

    # ── 逐 K 计时 ──
    rows = []  # (K, p50, mean, std)
    for K in steps:
        g = _capture_graph(infer, K)
        ts = _time_graph(g, args.iters, args.warmup)
        rows.append((K, float(np.percentile(ts, 50)), float(ts.mean()), float(ts.std())))
        print(f"  num_steps={K:2d}  P50={rows[-1][1]:7.3f}ms  mean={rows[-1][2]:7.3f}±{rows[-1][3]:.3f}ms")
        del g
        torch.cuda.empty_cache()

    # ── 线性拟合 T(K) = a + b·K (用 P50, 抗尾) ──
    Ks = np.array([r[0] for r in rows], dtype=np.float64)
    P50 = np.array([r[1] for r in rows], dtype=np.float64)
    A = np.vstack([np.ones_like(Ks), Ks]).T
    (a, b), *_ = np.linalg.lstsq(A, P50, rcond=None)
    resid = P50 - (a + b * Ks)
    full10 = a + b * 10.0

    # ── prefill 拆分: vision (draft 轮仍付) vs VLM-prefill (FLASH 在 draft 轮跳过, 复用 KV) ──
    # a = T_vis + T_vlm; FLASH draft 轮只跳 T_vlm (transformer_encoder 18 层 → KV cache 复用),
    # 但 vision_encoder 仍跑 (产 draft 的 prefix_embs)。分别捕图计时来拆 a。
    g_vis = _capture_stage_graph(infer, lambda inf: vision_encoder(inf.weights, inf.buffers, inf.num_views))
    t_vis = float(np.percentile(_time_graph(g_vis, args.iters, args.warmup), 50))
    del g_vis
    torch.cuda.empty_cache()
    g_ve = _capture_stage_graph(
        infer,
        lambda inf: (vision_encoder(inf.weights, inf.buffers, inf.num_views),
                     transformer_encoder(inf.weights, inf.buffers, inf.encoder_seq_len)),
    )
    t_vis_enc = float(np.percentile(_time_graph(g_ve, args.iters, args.warmup), 50))
    del g_ve
    torch.cuda.empty_cache()
    t_vlm = max(t_vis_enc - t_vis, 0.0)  # VLM-prefill (18 层 encoder transformer)

    print("\n========== R6: v1 5090 latency 拆解 ==========")
    print("  线性拟合 T(K) = a + b·K   (K=去噪步数)")
    print(f"    a (prefill 截距 = vision+VLM) = {a:7.3f} ms")
    print(f"      ├ T_vision (SigLIP)         = {t_vis:7.3f} ms   <- draft 轮仍付 (喂 draft prefix_embs)")
    print(f"      └ T_vlm-prefill (18 层 enc) = {t_vlm:7.3f} ms   <- **FLASH draft 轮跳过** (复用 KV cache)")
    print(f"        (vision+enc 实测 = {t_vis_enc:.3f} ms, 对照拟合截距 a={a:.3f} ms)")
    print(f"    b (每解码步斜率)              = {b:7.3f} ms/step")
    print(f"    T(10) = full 全量             = {full10:7.3f} ms   (拟合残差 max {np.abs(resid).max():.3f}ms)")

    d, ov = args.draft_ms, args.accept_overhead_ms
    vks = [int(x) for x in args.verify_k.split(",")]

    # ── 模型 A: 无 prefill-skip (每轮重算整条 prefill) —— 旧 (错) 假设, 留作对照 ──
    print("\n  -- 模型 A: 无 prefill 复用 (每轮重算 vision+VLM) [上一版错误假设] --")
    for K in (2,):
        t_acc = a + d + K * b + ov
        print(f"     K={K} 接受 = a+draft+{K}b+ov = {t_acc:.2f}ms → {full10/t_acc:.2f}× (天花板, prefill 全付)")

    # ── 模型 B: prefill-skip (FLASH 真实路径: draft 轮复用 VLM KV, 只重算 vision) ──
    # 稳态 draft 轮 = T_vis + draft + K·b + ov  (T_vlm 经周期性 full 轮摊薄, N→∞ 趋零)
    print("\n  -- 模型 B: FLASH 真实 prefill-skip (draft 轮跳 VLM-prefill, 复用上次 full 轮 KV) --")
    print(f"     full 轮      = T_vis + T_vlm + 10b = {t_vis:.1f}+{t_vlm:.1f}+{10*b:.1f} = {full10:.2f} ms")
    for K in vks:
        t_acc = t_vis + d + K * b + ov            # 接受: vision(新帧) + draft + K 步 verify(用缓存 KV) + 缝合
        sp = full10 / t_acc
        print(f"     K={K} draft-接受 = T_vis+draft+{K}b+ov = "
              f"{t_vis:.1f}+{d:.1f}+{K}×{b:.2f}+{ov:.1f} = {t_acc:6.2f} ms → **{sp:.2f}×** vs full")
    # 周期性 full 摊薄: 每 N 轮 1 full + (N-1) draft(接受) 的平均
    print("     周期性 full 刷新 (每 N draft 轮 1 full 轮) 的均摊加速 (K=2, 全接受):")
    K0 = vks[0]
    t_draft0 = t_vis + d + K0 * b + ov
    for N in (2, 4, 8, 16):
        avg = (full10 + (N - 1) * t_draft0) / N
        print(f"        N={N:2d}: 均摊 {avg:6.2f} ms → {full10/avg:.2f}×")

    print("\n  -- 读数 (修正: 用户指出 FLASH 确实跳 VLM-prefill, 上一版漏了) --")
    print(f"     • prefill {a:.1f}ms 里, 可跳的 VLM-prefill = {t_vlm:.1f}ms ({100*t_vlm/full10:.0f}% of full), "
          f"不可跳的 vision = {t_vis:.1f}ms。")
    print(f"     • R6 真实天花板 (K=2, 稳态全接受) ≈ {full10/(t_vis+d+2*b+ov):.2f}× vs v1 "
          f"(模型 B), 远高于模型 A 的 {full10/(a+d+2*b+ov):.2f}×。")
    print("     • 但 prefill-skip = verify 用**上次帧**的 VLM KV → draft 轮对视觉**部分开环** "
          "(只有 vision SigLIP 是新的, 18 层上下文是旧的); full 刷新越稀 (N 越大) 越开环。")
    print("     • 这正是本仓库反复诊断的开环失效模式: R6 的加速是用'可控视觉陈旧'换的, 非免费。"
          " N 的安全上限须真机看任务对视觉时效的敏感度 (抓取/接触段须高频 full)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
