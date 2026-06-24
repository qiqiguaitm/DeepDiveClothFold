"""gwp_ans / gwp_ori 推理服务 —— openpi WebSocket 协议兼容版 (部署用,取代 ZeroMQ 桥)。

把 gwp 世界-动作模型包成 openpi 的 policy.infer(obs)->{"actions":[48,14]},用 openpi
WebsocketPolicyServer 起服务。于是**现有 kai0 policy_inference_node + start_autonomy 链
原封不动**就能连上 (--mode websocket --ws-port <此端口>),gwp 自动继承 kai0 全套控制参数
(inference_rate / latency_k / rtc_execute_horizon / publish_rate / StreamActionBuffer /
min-jerk / proprio 反馈 / jump-protect / rerun / recorder) —— 与 kai0 同栈同参,在线对比 apples-to-apples。

观测契约 (node 发来的 openpi obs):
  state                                [14] 原始关节角+夹爪
  images.{top_head,hand_left,hand_right} CHW (uint8 或 float[0,1]); 经 build_ref_image -> 768x192
  prompt                               (用预算的 T5 embedding, 忽略文本)
返回: {"actions": [48,14] 绝对关节目标} (abs ckpt)

依赖 (gwp venv): websockets + openpi_client (已装)。推理走 scripts/opt_ans.opt_call (fp8+T_a3 ~87ms)。

用法:
  cd giga_world_policy && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/tim/gwp_eval_env/venv/bin/python scripts/serve_gwp_ws.py \
      --transformer_path /data2/gwp_eval/checkpoints/gwp_ans/transformer \
      --model_id        /data2/gwp_eval/checkpoints/Wan2.2-TI2V-5B-Diffusers \
      --stats_path      assets_visrobot01/norm_stats_vis_abs.json \
      --t5_embedding_pkl /data2/gwp_eval/data/visrobot01_val/t5_embedding/episode_000000.pt \
      --opt_tier fp8 --steps_act 3 --port 8000
"""
import argparse
import asyncio
import http
import time
import traceback

import numpy as np
import torch

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from diffusers.models import AutoencoderKLWan
from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.pipeline.wa_pipeline import WAPipeline
from world_action_model.pipeline.utils import (
    add_state_to_action, build_ref_image, denormalize_action, extract_normalization_tensors,
    load_stats, load_t5_embedding_from_pkl, normalize_state, resolve_delta_mask)


def _chw01(v):
    """node 发来的单相机图 -> CHW float[0,1] tensor (兼容 uint8/float、CHW/HWC)。"""
    t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
    if t.ndim != 3:
        raise ValueError(f"image must be 3D, got {tuple(t.shape)}")
    if t.shape[0] not in (1, 3):           # HWC -> CHW
        t = t.permute(2, 0, 1)
    t = t.float()
    if t.max() > 1.5:                       # uint8 -> [0,1]
        t = t / 255.0
    return t


class GwpPolicy:
    """openpi BasePolicy 风格:infer(obs)->{"actions": np.ndarray[48,14]}。"""

    def __init__(self, args):
        self.args = args
        self.dev = torch.device(args.device)
        self.dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        stats = load_stats(args.stats_path)
        self.norm = extract_normalization_tensors(stats, device=self.dev, state_dim=14, action_dim=14)
        self.dm = torch.tensor(resolve_delta_mask(stats, 14).tolist(), device=self.dev, dtype=torch.bool)
        self.t5 = load_t5_embedding_from_pkl(args.t5_embedding_pkl, target_len=int(args.t5_len)).to(self.dev, torch.float32)

        vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=self.dt)
        tf = CasualWorldActionTransformer.from_pretrained(args.transformer_path).to(self.dt)
        self.pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=self.dt).to(self.dev)
        self.LOOK = bool(getattr(tf.config, "action_attends_video", False))
        ANS = bool(getattr(tf.config, "async_noise", False))
        self.steps_act = (args.steps_act or None) if not ANS else (args.steps_act or 5)

        from scripts.opt_ans import AnsPrefixRunner, opt_call
        from scripts.prefix_cache import PrefixCachedRunner
        self._opt_call = opt_call
        if args.opt_tier == "fp8":
            from scripts.fp8_linear import swap_linears_to_fp8
            print(f"[serve_gwp_ws] fp8 swapped {swap_linears_to_fp8(tf.blocks)} linears", flush=True)
        if args.opt_tier in ("exact", "fp8"):
            for m in tf.modules():
                if hasattr(m, "fuse_projections") and hasattr(m, "set_processor"):
                    try: m.fuse_projections()
                    except Exception: pass
        self.runner = AnsPrefixRunner(tf) if self.LOOK else PrefixCachedRunner(tf)
        if args.opt_tier in ("exact", "fp8"):
            self.runner.compile_prepare("reduce-overhead")
            (self.runner.compile_step_ans if self.LOOK else self.runner.compile_step)("reduce-overhead")
        print(f"[serve_gwp_ws] tier={args.opt_tier} T_a={self.steps_act} lookahead={self.LOOK}", flush=True)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        a = self.args
        state = torch.as_tensor(np.asarray(obs["state"], dtype=np.float32)).reshape(-1)[:14]
        imgs_in = obs["images"]
        # node 发 bare 键 top_head/hand_left/hand_right; build_ref_image 的 resolve_view 要 canonical 键。
        KMAP = {"top_head": "observation.images.cam_high",
                "hand_left": "observation.images.cam_left_wrist",
                "hand_right": "observation.images.cam_right_wrist"}
        images = {KMAP.get(k, k): _chw01(v) for k, v in imgs_in.items()}
        ref = build_ref_image(images=images, dst_size=(a.dst_width, a.dst_height), crop_mode="center")
        st = state.unsqueeze(0).to(self.dev)
        ns = normalize_state(st, self.norm, mode=a.norm_mode).to(self.dev, self.dt)
        act = self._opt_call(
            self.pipe, self.runner, image=ref, state=ns,
            prompt_embeds=self.t5.unsqueeze(0).to(self.dev, torch.float32),
            height=a.dst_height, width=a.dst_width, num_frames=a.num_frames,
            action_chunk=a.action_chunk, num_inference_steps=a.steps_inf,
            action_num_inference_steps=self.steps_act, is_ans=self.LOOK, bac_skip=a.opt_bac)
        pa = add_state_to_action(denormalize_action(act[0].float(), self.norm, mode=a.norm_mode),
                                 st[0].float(), action_chunk=a.action_chunk, mask=self.dm)
        pa = pa.cpu().numpy().astype(np.float32)

        # --- 在线诊断 ---
        # 每次都打印动作"运动量"(相邻步均绝对差) + state 是否落在 norm 范围内 -> 看是否塌缩/OOD。
        self._n = getattr(self, "_n", 0) + 1
        motion = float(np.abs(np.diff(pa, axis=0)).mean())          # ~0 = 静止塌缩 -> "停顿"
        s_np = state.cpu().numpy()
        z = (s_np - self.norm.state_mean.cpu().numpy()) / (self.norm.state_std.cpu().numpy() + 1e-8)
        ood = float(np.abs(z).max())                                 # state 偏离训练分布的最大 z 分数
        if self._n <= 5 or self._n % 20 == 0:
            print(f"[infer #{self._n}] action_motion={motion:.4f} state_max|z|={ood:.2f} "
                  f"act[0,:7]={pa[0,:7].round(3).tolist()}", flush=True)
        if a.debug_dump_dir and self._n <= a.debug_dump_n:
            import os
            os.makedirs(a.debug_dump_dir, exist_ok=True)
            ref.save(os.path.join(a.debug_dump_dir, f"ref_{self._n:03d}.png"))   # 看颜色/左右/构图/场景
            np.savez(os.path.join(a.debug_dump_dir, f"io_{self._n:03d}.npz"), state=s_np, action=pa)
            print(f"[infer #{self._n}] dumped ref+io -> {a.debug_dump_dir}", flush=True)
        return {"actions": pa}


# --- openpi WebsocketPolicyServer 的最小复制 (只依赖 openpi_client + websockets, 无 JAX) ---
class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        self._policy, self._host, self._port, self._metadata = policy, host, port, (metadata or {})

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                  ping_timeout=300, close_timeout=300, process_request=_health) as s:
            print(f"[serve_gwp_ws] ready, listening ws://{self._host}:{self._port}", flush=True)
            await s.serve_forever()

    async def _handler(self, ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        prev = None
        while True:
            try:
                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(await ws.recv())
                ti = time.monotonic()
                action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": (time.monotonic() - ti) * 1000}
                if prev is not None:
                    action["server_timing"]["prev_total_ms"] = prev * 1000
                await ws.send(packer.pack(action))
                prev = time.monotonic() - t0
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="server error")
                raise


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
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
    ap.add_argument("--steps_inf", type=int, default=10)
    ap.add_argument("--steps_act", type=int, default=3)
    ap.add_argument("--opt_tier", default="fp8", choices=["eager", "exact", "fp8"])
    ap.add_argument("--opt_bac", type=int, default=0)
    ap.add_argument("--norm_mode", default="zscore", choices=["minmax", "zscore"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--debug_dump_dir", default="")   # 落盘头 N 次的 ref图+state+action 做诊断
    ap.add_argument("--debug_dump_n", type=int, default=10)
    args = ap.parse_args()

    policy = GwpPolicy(args)
    if args.warmup:
        dummy = {"state": np.zeros(14, np.float32),
                 "images": {k: np.zeros((3, 240, 320), np.uint8) for k in ("top_head", "hand_left", "hand_right")},
                 "prompt": "Flatten and fold the cloth."}
        for i in range(int(args.warmup)):
            t = time.monotonic(); r = policy.infer(dummy)
            print(f"[serve_gwp_ws] warmup {i}: {r['actions'].shape} {(time.monotonic()-t)*1e3:.0f}ms", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata={"model": "gwp", "action_dim": 14, "action_horizon": args.action_chunk}).serve_forever()


if __name__ == "__main__":
    main()
