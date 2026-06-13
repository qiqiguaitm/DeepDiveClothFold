"""gwp_ans / gwp_ori 优化推理服务 (部署用)。

与 scripts/inference_server.py 的区别:去噪循环走 scripts/opt_ans.opt_call + AnsPrefixRunner
(fp8 + T_a 短步),复现 docs/wam_mae_root_cause_and_optimization.md 的部署王牌档
(gwp_ans @ fp8+T_a3 = 87ms 全栈)。协议沿用 giga_models.sockets.RobotInferenceServer
(ZeroMQ REP, endpoint='inference', torch.save/load 线格式),与离线 server 一致。

观测契约 (ROS2 桥发来的 dict):
  observation.state                       : [14] 原始关节角+夹爪 (未归一化)
  observation.images.cam_high             : CHW float[0,1]  (= top_head)
  observation.images.cam_left_wrist       : CHW float[0,1]  (= hand_left)
  observation.images.cam_right_wrist      : CHW float[0,1]  (= hand_right)
返回:
  action [action_chunk, 14] 反归一化后的关节目标 (abs ckpt: 绝对关节; delta ckpt: 已加 state)

用法 (gwp venv):
  cd giga_world_policy && source /home/tim/gwp_eval_env/venv/bin/activate  # 或直接用其 python
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/serve_gwp_opt.py \
    --transformer_path /data2/gwp_eval/checkpoints/gwp_ans/transformer \
    --model_id        /data2/gwp_eval/checkpoints/Wan2.2-TI2V-5B-Diffusers \
    --stats_path      assets_visrobot01/norm_stats_vis_abs.json \
    --t5_embedding_pkl /data2/gwp_eval/data/visrobot01_val/t5_embedding/episode_000000.pt \
    --opt_tier fp8 --steps_act 3 --port 8093
"""
import argparse
import time
import types

import torch

from giga_models.sockets import RobotInferenceServer
from diffusers.models import AutoencoderKLWan

from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.pipeline.wa_pipeline import WAPipeline
from world_action_model.pipeline.utils import (
    add_state_to_action,
    build_ref_image,
    denormalize_action,
    extract_normalization_tensors,
    load_stats,
    load_t5_embedding_from_pkl,
    normalize_state,
    resolve_delta_mask,
    resolve_view,
)


def build_policy(args):
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    dm = torch.tensor(resolve_delta_mask(stats, 14).tolist(), device=dev, dtype=torch.bool)
    t5 = load_t5_embedding_from_pkl(args.t5_embedding_pkl, target_len=int(args.t5_len)).to(dev, torch.float32)

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    tf = CasualWorldActionTransformer.from_pretrained(args.transformer_path).to(dt)
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev)

    LOOKAHEAD = bool(getattr(tf.config, "action_attends_video", False))
    ANS = bool(getattr(tf.config, "async_noise", False))
    # ANS ckpt: 动作 T_a 步先出 (默认 5); --steps_act 覆盖 (部署默认 3)
    steps_act = (args.steps_act or None) if not ANS else (args.steps_act or 5)
    print(f"[serve_gwp_opt] action_attends_video={LOOKAHEAD} async_noise={ANS} -> T_a={steps_act}/T_O={args.steps_inf}",
          flush=True)

    # ---- 优化引擎 (复现 episode_report --engine opt 的 setup) ----
    from scripts.opt_ans import AnsPrefixRunner, opt_call
    from scripts.prefix_cache import PrefixCachedRunner
    if args.opt_tier == "fp8":
        from scripts.fp8_linear import swap_linears_to_fp8
        print(f"[serve_gwp_opt] fp8 swapped {swap_linears_to_fp8(tf.blocks)} linears", flush=True)
    if args.opt_tier in ("exact", "fp8"):
        for _mod in tf.modules():
            if hasattr(_mod, "fuse_projections") and hasattr(_mod, "set_processor"):
                try:
                    _mod.fuse_projections()
                except Exception:
                    pass
    runner = AnsPrefixRunner(tf) if LOOKAHEAD else PrefixCachedRunner(tf)
    if args.opt_tier in ("exact", "fp8"):
        runner.compile_prepare("reduce-overhead")
        (runner.compile_step_ans if LOOKAHEAD else runner.compile_step)("reduce-overhead")
    print(f"[serve_gwp_opt] engine=opt tier={args.opt_tier} runner={type(runner).__name__}", flush=True)

    @torch.no_grad()
    def inference(self, observation):
        t0 = time.perf_counter()
        state = observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        state = state.float().reshape(-1)[:14]
        images = {
            "observation.images.cam_high": resolve_view(observation, "observation.images.cam_high"),
            "observation.images.cam_left_wrist": resolve_view(observation, "observation.images.cam_left_wrist"),
            "observation.images.cam_right_wrist": resolve_view(observation, "observation.images.cam_right_wrist"),
        }
        for k, v in list(images.items()):
            images[k] = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        ref = build_ref_image(images=images, dst_size=(args.dst_width, args.dst_height), crop_mode="center")

        st = state.unsqueeze(0).to(dev)
        ns = normalize_state(st, norm, mode=args.norm_mode).to(dev, dt)
        act = opt_call(
            pipe, runner, image=ref, state=ns,
            prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32),
            height=args.dst_height, width=args.dst_width, num_frames=args.num_frames,
            action_chunk=args.action_chunk, num_inference_steps=args.steps_inf,
            action_num_inference_steps=steps_act, is_ans=LOOKAHEAD, bac_skip=args.opt_bac,
        )
        pa = add_state_to_action(
            denormalize_action(act[0].float(), norm, mode=args.norm_mode),
            st[0].float(), action_chunk=args.action_chunk, mask=dm,
        )
        ms = (time.perf_counter() - t0) * 1e3
        return {"actions": pa.cpu(), "server_timing": {"infer_ms": ms}}

    pipe.inference = types.MethodType(inference, pipe)

    # warmup/compile: 先跑一次假观测把 fp8/CUDA-graph 编译开销摊在启动期
    if args.warmup:
        dummy = {
            "observation.state": torch.zeros(14),
            "observation.images.cam_high": torch.zeros(3, args.dst_height, args.dst_width // 3),
            "observation.images.cam_left_wrist": torch.zeros(3, args.dst_height, args.dst_width // 3),
            "observation.images.cam_right_wrist": torch.zeros(3, args.dst_height, args.dst_width // 3),
        }
        for i in range(2):
            r = pipe.inference(dummy)
            print(f"[serve_gwp_opt] warmup {i}: {r['actions'].shape} {r['server_timing']['infer_ms']:.1f}ms", flush=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--transformer_path", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--t5_embedding_pkl", required=True)
    ap.add_argument("--t5_len", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dst_width", type=int, default=768)
    ap.add_argument("--dst_height", type=int, default=192)
    ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--steps_inf", type=int, default=10)         # T_O 视频去噪步
    ap.add_argument("--steps_act", type=int, default=3)          # T_a 动作去噪步 (部署默认 3)
    ap.add_argument("--opt_tier", default="fp8", choices=["eager", "exact", "fp8"])
    ap.add_argument("--opt_bac", type=int, default=0)
    ap.add_argument("--norm_mode", default="zscore", choices=["minmax", "zscore"])
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    policy = build_policy(args)
    print(f"[serve_gwp_opt] ready, listening tcp://{args.host}:{args.port}", flush=True)
    RobotInferenceServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
