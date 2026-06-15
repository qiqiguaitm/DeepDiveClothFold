"""FastWAM infer_action 推理优化:首帧 KV 缓存(已内置)+ context_emb 预计算 +
torch.compile CUDA-graph + FP8 ActionDiT 块。

参考 giga_world_policy/scripts/opt_ans.py;针对 FastWAM MoT 架构重写。
ActionDiT:1024-dim × 30 层;per-step hot path = 30 块 × flash-attn(q_act=48,
kv=[video_kv+act_kv])。首帧 video KV 由 mot.prefill_video_cache() 预计算一次(内置)。
本文件新增:context_emb 预计算(text_embedding 跨步恒定)→ 减少每步 MLP 重算;
编译步函数进 CUDA-graph;可选 FP8 swapping ActionDiT Linear 层。

用法(bench+parity,支持真实权重):
  cd /data1/tim/workspace/deepdive_kai0/fastwam   # or PFS path
  PYTHONPATH=src python scripts/opt_infer_action.py \\
      [--ckpt runs/.../step_025510.pt] \\
      [--tier exact|fp8] [--nfe 5 10 20] [--parity] [--bench 30]

eval 集成:eval_offline_fold.py --engine opt --opt_tier exact|fp8 --nfe 5
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d

import torch.nn as nn

# ---------------------------------------------------------------------------
# FP8 — tensorwise fallback (sm_120/jpsz only supports per-tensor scale_a/scale_b)
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0  # e4m3 max

class _FP8LinearTensorwise(nn.Module):
    """Per-tensor W8A8 FP8 Linear via torch._scaled_mm with scalar scales.

    Rowwise (per-token activation) scaling is not supported on all sm_120 builds;
    tensorwise is universally available and gives ~same bandwidth win at bs=1.
    Weight scale is fixed at init (offline); activation scale is computed per forward.
    """
    def __init__(self, lin: nn.Linear):
        super().__init__()
        N, K = lin.weight.shape
        w = lin.weight.data.float()
        amax_w = w.abs().max().clamp(min=1e-8)
        scale_w = amax_w / _FP8_MAX
        wq = (w / scale_w).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("wq", wq)
        self.register_buffer("scale_w", scale_w.to(torch.float32).reshape(1))
        self.bias = nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None
        self.N, self.K = N, K

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, self.K).float()
        amax_x = x2.abs().max().clamp(min=1e-8)
        scale_x = (amax_x / _FP8_MAX).reshape(1)
        xq = (x2 / scale_x).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(xq, self.wq.t(), scale_a=scale_x, scale_b=self.scale_w,
                               bias=self.bias.to(x.dtype) if self.bias is not None else None,
                               out_dtype=x.dtype)
        return out.reshape(*shape[:-1], self.N)


def _swap_fp8(module, min_k=256):
    """Try rowwise FP8 (gwp); fall back to tensorwise if not supported on this GPU."""
    # Try rowwise first (highest accuracy)
    _rowwise_ok = None
    def _can_rowwise():
        nonlocal _rowwise_ok
        if _rowwise_ok is None:
            try:
                a = torch.ones(16, 16, dtype=torch.float8_e4m3fn, device="cuda")
                torch._scaled_mm(a, a.t(),
                                 scale_a=torch.ones(16, 1, device="cuda"),
                                 scale_b=torch.ones(1, 16, device="cuda"),
                                 out_dtype=torch.bfloat16)
                _rowwise_ok = True
            except Exception:
                _rowwise_ok = False
        return _rowwise_ok

    if _can_rowwise():
        gwp = REPO.parent / "giga_world_policy" / "scripts"
        sys.path.insert(0, str(gwp))
        from fp8_linear import swap_linears_to_fp8
        return swap_linears_to_fp8(module, min_k=min_k), "rowwise"

    # Tensorwise fallback
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            N, K = child.weight.shape
            if K >= min_k and K % 16 == 0 and N % 16 == 0:
                setattr(module, name, _FP8LinearTensorwise(child).to(child.weight.device))
                n += 1
        else:
            sub_n, _ = _swap_fp8(child, min_k)
            n += sub_n
    return n, "tensorwise"


# ---------------------------------------------------------------------------
# Pure step function (compilable, no self-mutation issues)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _step_core(action_expert, mot,
               latents_action, timestep,
               ctx_emb, ctx_attn_mask,
               video_kv_cache, attention_mask, video_seq_len, action_freqs):
    """Hot-path step function with pre-cached constants.

    Replaces action_expert.pre_dit + mot.forward_action_with_video_cache + post_dit.
    Skips re-computing text_embedding(context) — pre-cached in ctx_emb.
    This function's inputs are all tensors or static ints → safe for torch.compile CUDA-graph.
    """
    ae = action_expert
    t = ae.time_embedding(sinusoidal_embedding_1d(ae.freq_dim, timestep))
    t_mod = ae.time_projection(t).unflatten(1, (6, ae.hidden_dim))
    tokens = ae.action_encoder(latents_action)
    tokens = mot.forward_action_with_video_cache(
        action_tokens=tokens,
        action_freqs=action_freqs,
        action_t_mod=t_mod,
        action_context_payload={"context": ctx_emb, "mask": ctx_attn_mask},
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    return ae.head(tokens)


# ---------------------------------------------------------------------------
# ActionStepRunner
# ---------------------------------------------------------------------------

class ActionStepRunner:
    """infer_action 加速 runner:预缓存 video-KV 和 context_emb,编译步函数。

    用法:
        runner = ActionStepRunner(model)
        runner.compile_step()        # 只需一次
        # 每次新 obs:
        runner.prepare(ctx, ctx_mask, image, proprio)
        for t, dt in schedule:
            pred = runner.step(latents_action, t)
            latents_action = sched.step(pred, dt, latents_action)
    """

    def __init__(self, model):
        self.model = model
        self._step_fn = None     # compiled or None
        # constants filled by prepare()
        self.video_kv_cache = None
        self.attention_mask = None
        self.video_seq_len = None
        self._ctx_emb = None
        self._ctx_attn_mask = None
        self._action_freqs = None

    def compile_step(self, mode="reduce-overhead"):
        """Wrap _step_core with torch.compile — call once after FP8/other patches."""
        import functools
        # Bind model components so compile sees them as constants, not arguments.
        # All per-step tensors (latents_action, timestep, ctx_emb, video_kv_cache,
        # attention_mask, action_freqs) are arguments → CUDA graph replays them correctly
        # even when prepare() recomputes them between observations.
        fn = functools.partial(_step_core, self.model.action_expert, self.model.mot)
        self._step_fn = torch.compile(fn, mode=mode, fullgraph=False)
        print(f"[opt] compiled step with mode={mode}", flush=True)

    @torch.no_grad()
    def prepare(self, context, context_mask, image, proprio=None, action_horizon=48):
        """预计算所有跨步常量(每次新 obs 调一次)。"""
        m = self.model
        ae = m.action_expert
        dev = m.device
        dt = m.torch_dtype
        fuse_flag = bool(getattr(m.video_expert, "fuse_vae_embedding_in_latents", False))

        ctx = context.to(dev, dt, non_blocking=True)
        cmask = context_mask.to(dev, non_blocking=True)
        if proprio is not None:
            p = proprio.to(dev, dt)
            if p.ndim == 1:
                p = p.unsqueeze(0)  # (D,) → (1, D)
            ctx, cmask = m._append_proprio_to_context(ctx, cmask, p)

        # context_emb: ActionDiT.text_embedding(ctx) — 跨步恒定
        self._ctx_emb = ae.text_embedding(ctx).contiguous()  # [1, L', hidden_dim]
        # context_attn_mask: [1, action_horizon, L']
        self._ctx_attn_mask = cmask.unsqueeze(1).expand(-1, action_horizon, -1).contiguous()

        # video KV prefill (首帧 latent)
        img = image.to(dev, dt, non_blocking=True)
        first_frame_latents = m._encode_input_image_latents_tensor(img, tiled=False)
        t_video = torch.zeros((first_frame_latents.shape[0],), dtype=dt, device=dev)
        video_pre = m.video_expert.pre_dit(
            x=first_frame_latents, timestep=t_video,
            context=ctx, context_mask=cmask, action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        v_len = int(video_pre["tokens"].shape[1])
        a_mask = m._build_mot_attention_mask(
            video_seq_len=v_len, action_seq_len=action_horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=dev,
        )
        kv_raw = m.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=a_mask[:v_len, :v_len],
        )
        # clone out of inductor pool so CUDA-graph reads stable addresses
        self.video_kv_cache = [{k2: v2.clone().contiguous() for k2, v2 in d.items()} for d in kv_raw]
        self.attention_mask = a_mask
        self.video_seq_len = v_len
        self._action_freqs = ae.freqs[:action_horizon].view(action_horizon, 1, -1).to(dev).contiguous()

    @torch.no_grad()
    def step(self, latents_action, timestep):
        """单步去噪;prepare() 必须先调。"""
        fn = self._step_fn or (lambda *a: _step_core(self.model.action_expert, self.model.mot, *a))
        if self._step_fn is not None:
            torch.compiler.cudagraph_mark_step_begin()
            out = fn(latents_action, timestep,
                     self._ctx_emb, self._ctx_attn_mask,
                     self.video_kv_cache, self.attention_mask,
                     self.video_seq_len, self._action_freqs)
            return out.clone()
        return fn(latents_action, timestep,
                  self._ctx_emb, self._ctx_attn_mask,
                  self.video_kv_cache, self.attention_mask,
                  self.video_seq_len, self._action_freqs)


# ---------------------------------------------------------------------------
# opt_infer_action:端到端入口,契约同 model.infer_action
# ---------------------------------------------------------------------------

@torch.no_grad()
def opt_infer_action(model, runner, *,
                     context, context_mask, image, proprio=None,
                     action_horizon=48, num_inference_steps=20, seed=0):
    """与 model.infer_action 相同的输入/输出约定,热路径换成 runner。

    Returns dict with keys 'action' [48,14] float32 on cpu, 'action_loop_ms'.
    """
    dev, dt = model.device, model.torch_dtype
    runner.prepare(context, context_mask, image, proprio=proprio, action_horizon=action_horizon)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents_action = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        generator=generator, device="cpu", dtype=torch.float32,
    ).to(dev, dt)

    ts, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps, device=dev, dtype=dt, shift_override=None)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step_t, step_dt in zip(ts, deltas):
        tsa = step_t.unsqueeze(0).to(dtype=dt, device=dev)
        pred = runner.step(latents_action, tsa)
        latents_action = model.infer_action_scheduler.step(pred, step_dt, latents_action)
    torch.cuda.synchronize()
    loop_ms = (time.perf_counter() - t0) * 1000

    return {
        "action": latents_action[0].detach().to("cpu", torch.float32),
        "action_loop_ms": loop_ms,
    }


# ---------------------------------------------------------------------------
# Model builder (random weights for timing, optional real ckpt for parity)
# ---------------------------------------------------------------------------

def _build_model(args, device, dtype):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs")):
        cfg = compose(config_name="train", overrides=[
            "data=visrobot01_fold",
            "task=visrobot01_fold_uncond_1e-4",
            "model.load_text_encoder=false",
            "model.skip_dit_load_from_pretrain=true",
            "model.action_dit_pretrained_path=null",
        ])

    vdc = OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)
    adc = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    vsched = OmegaConf.to_container(cfg.model.video_scheduler, resolve=True)
    asched = OmegaConf.to_container(cfg.model.action_scheduler, resolve=True)

    video_expert = WanVideoDiT(**vdc).to(device=device, dtype=dtype)
    action_expert = ActionDiT(**adc).to(device=device, dtype=dtype)  # action_dim already in adc (from data config)
    vae = WanVideoVAE38().to(device=device, dtype=dtype)
    mot = MoT(mixtures={"video": video_expert, "action": action_expert},
              mot_checkpoint_mixed_attn=False)
    model = FastWAM(
        video_expert=video_expert, action_expert=action_expert, mot=mot, vae=vae,
        text_encoder=None, tokenizer=None, text_dim=int(vdc["text_dim"]),
        device=device, torch_dtype=dtype,
        video_train_shift=float(vsched.get("train_shift", 5.0)),
        video_infer_shift=float(vsched.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(vsched.get("num_train_timesteps", 1000)),
        action_train_shift=float(asched["train_shift"]),
        action_infer_shift=float(asched["infer_shift"]),
        action_num_train_timesteps=int(asched["num_train_timesteps"]),
    )

    if getattr(args, "ckpt", None):
        ckpt_path = Path(args.ckpt)
        if ckpt_path.exists():
            sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
            model.mot.load_state_dict(sd["mot"], strict=True)
            print(f"[load] real ckpt {ckpt_path.name}  step={sd.get('step')}", flush=True)
        else:
            print(f"[warn] ckpt not found: {ckpt_path}  → random weights (timing identical)", flush=True)
    else:
        print("[build] random weights (latency is shape-determined)", flush=True)

    # move non-registered-buffer tensors to GPU so CUDA graphs don't trip on CPU ops
    def _to_dev(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x
    model.action_expert.freqs = _to_dev(model.action_expert.freqs)
    model.video_expert.freqs = tuple(_to_dev(f) for f in model.video_expert.freqs)
    for obj in (model.vae, getattr(model.vae, "model", None)):
        if obj is None: continue
        for attr in ("mean", "std"):
            if hasattr(obj, attr): setattr(obj, attr, _to_dev(getattr(obj, attr)))
        if hasattr(obj, "scale") and isinstance(obj.scale, (list, tuple)):
            obj.scale = [_to_dev(s) for s in obj.scale]

    return model.eval()


# ---------------------------------------------------------------------------
# CLI: bench + parity
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Optional real .pt checkpoint (parity meaningful; timing is weight-independent)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tier", default="exact", choices=["eager", "exact", "fp8"],
                    help="eager=no compile  exact=compile CUDA-graph  fp8=compile+FP8 ActionDiT")
    ap.add_argument("--nfe", type=int, nargs="+", default=[20, 10, 5])
    ap.add_argument("--bench", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--parity", action="store_true",
                    help="Compare opt vs stock for nfe[0]; meaningful with real ckpt")
    ap.add_argument("--action_horizon", type=int, default=48)
    a = ap.parse_args()

    import os; os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dev, dtype = "cuda:0", torch.bfloat16
    print(f"[device] GPU={a.gpu} -> {torch.cuda.get_device_name(0)} | tier={a.tier}")

    model = _build_model(a, dev, dtype)

    if a.tier == "fp8":
        n, fp8_mode = _swap_fp8(model.action_expert.blocks)
        n2, _ = _swap_fp8(model.action_expert.text_embedding)
        n3, _ = _swap_fp8(model.action_expert.time_embedding)
        n4, _ = _swap_fp8(model.action_expert.time_projection)
        print(f"[fp8/{fp8_mode}] blocks={n} text_emb={n2} time_emb={n3} time_proj={n4}")

    runner = ActionStepRunner(model)
    if a.tier in ("exact", "fp8"):
        runner.compile_step("reduce-overhead")

    # synthetic inputs (shape-identical to real deployment; values don't affect timing)
    ctx_len = 128
    ctx = torch.randn(1, ctx_len, model.text_dim, device=dev, dtype=dtype)
    ctx_mask = torch.ones(1, ctx_len, dtype=torch.bool, device=dev)
    img = torch.randn(1, 3, 384, 320, device=dev, dtype=dtype)

    def run_opt(nfe):
        return opt_infer_action(model, runner, context=ctx, context_mask=ctx_mask,
                                image=img, action_horizon=a.action_horizon,
                                num_inference_steps=nfe, seed=0)

    def run_stock(nfe):
        with torch.no_grad():
            return model.infer_action(prompt=None, input_image=img,
                                      action_horizon=a.action_horizon,
                                      context=ctx, context_mask=ctx_mask,
                                      num_inference_steps=nfe, seed=0)

    # parity (only meaningful with real ckpt)
    if a.parity:
        print(f"\n[parity] warming up opt ({a.warmup} calls)...")
        for _ in range(a.warmup): run_opt(a.nfe[0])
        ref = run_stock(a.nfe[0])["action"]
        opt_out = run_opt(a.nfe[0])["action"]
        diff = (opt_out.float() - ref.float()).abs()
        rel = diff.max().item() / (ref.float().abs().max().item() + 1e-6)
        print(f"[parity] nfe={a.nfe[0]} max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} "
              f"rel={rel:.3e} -> {'OK' if rel < 2e-2 else 'WARN'}")

    # benchmark
    print(f"\n{'nfe':>5}  {'stock ms':>10}  {'opt ms':>10}  {'speedup':>9}  {'chunks/s':>10}")
    for nfe in a.nfe:
        # stock (skip for fp8; random weights make parity meaningless)
        if a.tier != "fp8":
            for _ in range(a.warmup): run_stock(nfe)
            torch.cuda.synchronize(); t_st = []
            for _ in range(a.bench):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                run_stock(nfe)
                torch.cuda.synchronize(); t_st.append((time.perf_counter()-t0)*1000)
            ms_st = sum(t_st)/len(t_st)
        else:
            ms_st = float("nan")

        # opt (extra warmup for compile on first nfe)
        extra = 3 if a.tier in ("exact", "fp8") else 0
        for _ in range(a.warmup + extra): run_opt(nfe)
        torch.cuda.synchronize(); t_op = []
        for _ in range(a.bench):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run_opt(nfe)
            torch.cuda.synchronize(); t_op.append((time.perf_counter()-t0)*1000)
        ms_op = sum(t_op)/len(t_op)
        spd = ms_st/ms_op if ms_st == ms_st else float("nan")
        print(f"{nfe:>5}  {ms_st:>10.1f}  {ms_op:>10.1f}  {spd:>8.2f}x  {1000/ms_op:>10.1f}")

    peak = torch.cuda.max_memory_allocated()/2**30
    print(f"\n[mem] peak CUDA: {peak:.2f} GiB")
    print(f"[summary] tier={a.tier}  nfe={a.nfe}")


if __name__ == "__main__":
    main()
