"""FastWAM 推理服务 —— openpi WebSocket 协议兼容版 (部署用)。

把 FastWAM 包成 openpi policy.infer(obs)->{"actions":[48,14]},用 openpi WebsocketPolicyServer 起服务。
于是**现有 kai0 policy_inference_node + start_autonomy 链原封不动**就能连(--mode websocket --ws-port),
与 gwp 同栈同参 —— 在线对比 apples-to-apples。结构镜像 giga_world_policy/scripts/serve_gwp_ws.py。

FastWAM 特性(与 gwp 不同):
  - **无 test-time 视频想象**:action expert 只读首帧 KV + 文本 + 因果自注意(infer_action),
    天然回避 gwp_ans 的闭环视频塌缩。
  - 归一化:**z-score**(dataset_stats.json 的 global_mean/std),非 gwp 的 q01/q99。
  - 文本:**预算 T5 context 缓存**(data/text_embeds_cache/visrobot01_fold/*.pt)。
  - 图像:3 相机拼 **[3,384,320]**(top 256x320 + 双腕 128x160),非 gwp 的 768x192。
  - 优化:opt_infer_action(ActionStepRunner,torch.compile+CUDA-graph,fp8 可选)~75ms@nfe4。

依赖:gwp_eval_env(torch 2.11 + hydra/modelscope/boto3 + openpi_client + websockets==15.0.1)。

用法(gwp_eval_env):
  cd fastwam && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:scripts \
    /home/tim/gwp_eval_env/venv/bin/python scripts/serve_fastwam_ws.py \
      --weights runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v3/checkpoints/weights/step_025510.pt \
      --stats data/visrobot01_fold/dataset_stats.json --nfe 4 --opt_tier exact --port 8004
"""
import argparse, asyncio, http, glob, time, traceback
import numpy as np
import torch
import torchvision.transforms.functional as TF

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from eval_offline_fold import build_model, prep_image   # 复用同一套 load + 像素链
from opt_infer_action import ActionStepRunner, opt_infer_action

# node 发 bare 键 -> FastWAM 相机名
KMAP = {"top_head": "cam_high", "hand_left": "cam_left_wrist", "hand_right": "cam_right_wrist"}


def _to_hwc_u8(v):
    a = np.asarray(v)
    if a.ndim == 3 and a.shape[0] in (1, 3):     # CHW -> HWC (node 发 CHW)
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255.0).astype(np.uint8) if a.max() <= 1.5 else a.astype(np.uint8)
    return a[:, :, :3]


class FastwamPolicy:
    def __init__(self, args):
        self.args = args
        self.model = build_model(args.weights)
        dev, dt = self.model.device, self.model.torch_dtype
        # z-score stats
        import json
        st = json.load(open(args.stats))
        self.a_mean = np.array(st["action"]["default"]["global_mean"], np.float32)
        self.a_std = np.array(st["action"]["default"]["global_std"], np.float32)
        self.s_mean = np.array(st["state"]["default"]["global_mean"], np.float32)
        self.s_std = np.array(st["state"]["default"]["global_std"], np.float32)
        # cached T5 context (single fold prompt)
        t5 = torch.load(glob.glob(args.t5_cache)[0], map_location="cpu", weights_only=False)
        ctx = t5["context"]; cmask = t5["mask"].bool()
        ctx = ctx.clone(); ctx[~cmask] = 0.0; cmask = torch.ones_like(cmask)
        if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
        if cmask.ndim == 1: cmask = cmask.unsqueeze(0)
        self.ctx = ctx.to(dev, dt); self.cmask = cmask.to(dev)
        # opt engine
        if args.opt_tier == "fp8":
            from opt_infer_action import _swap_fp8
            n, mode = _swap_fp8(self.model.action_expert.blocks)
            _swap_fp8(self.model.action_expert.text_embedding); _swap_fp8(self.model.action_expert.time_embedding)
            _swap_fp8(self.model.action_expert.time_projection)
            print(f"[serve_fastwam] fp8/{mode} blocks={n}", flush=True)
        self.runner = ActionStepRunner(self.model)
        if args.opt_tier in ("exact", "fp8"):
            self.runner.compile_step("reduce-overhead")
        self.dev, self.dt = dev, dt
        print(f"[serve_fastwam] tier={args.opt_tier} nfe={args.nfe} (infer_action, no video imagination)", flush=True)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        a = self.args
        state = np.asarray(obs["state"], np.float32).reshape(-1)[:14]
        frames = {KMAP.get(k, k): _to_hwc_u8(v) for k, v in obs["images"].items()}
        img = prep_image(frames)                              # [3,384,320] in [-1,1]
        prop = torch.from_numpy((state - self.s_mean) / (self.s_std + 1e-8)).float()
        out = opt_infer_action(self.model, self.runner, context=self.ctx, context_mask=self.cmask,
                               image=img, proprio=prop, action_horizon=48,
                               num_inference_steps=a.nfe, seed=0)
        pa = out["action"].float().cpu().numpy() * (self.a_std + 1e-8) + self.a_mean   # [48,14] abs joints
        pa = pa.astype(np.float32)
        self._n = getattr(self, "_n", 0) + 1
        motion = float(np.abs(np.diff(pa, axis=0)).mean())
        if self._n <= 5 or self._n % 20 == 0:
            print(f"[infer #{self._n}] motion={motion:.4f} act[0,:7]={pa[0,:7].round(3).tolist()}", flush=True)
        if a.debug_dump_dir and self._n <= a.debug_dump_n:
            import os
            from PIL import Image
            os.makedirs(a.debug_dump_dir, exist_ok=True)
            ref = ((img.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()  # [384,320,3]
            Image.fromarray(ref).save(os.path.join(a.debug_dump_dir, f"ref_{self._n:03d}.png"))
            np.savez(os.path.join(a.debug_dump_dir, f"io_{self._n:03d}.npz"), state=state, action=pa)
        return {"actions": pa}


# --- openpi WebsocketPolicyServer 最小复制 (无 JAX) ---
class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        self._policy, self._host, self._port, self._metadata = policy, host, port, (metadata or {})

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                  ping_timeout=300, close_timeout=300, process_request=_health) as s:
            print(f"[serve_fastwam] ready, listening ws://{self._host}:{self._port}", flush=True)
            await s.serve_forever()

    async def _handler(self, ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        prev = None
        while True:
            try:
                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(await ws.recv())
                ti = time.monotonic(); action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": (time.monotonic() - ti) * 1000}
                if prev is not None: action["server_timing"]["prev_total_ms"] = prev * 1000
                await ws.send(packer.pack(action)); prev = time.monotonic() - t0
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="server error"); raise


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stats", default="data/visrobot01_fold/dataset_stats.json")
    ap.add_argument("--t5_cache", default="data/text_embeds_cache/visrobot01_fold/*.pt")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--opt_tier", default="exact", choices=["eager", "exact", "fp8"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--debug_dump_dir", default="")
    ap.add_argument("--debug_dump_n", type=int, default=15)
    args = ap.parse_args()
    policy = FastwamPolicy(args)
    if args.warmup:
        dummy = {"state": np.zeros(14, np.float32),
                 "images": {k: np.zeros((3, 240, 320), np.uint8) for k in ("top_head", "hand_left", "hand_right")}}
        for i in range(int(args.warmup)):
            t = time.monotonic(); r = policy.infer(dummy)
            print(f"[serve_fastwam] warmup {i}: {r['actions'].shape} {(time.monotonic()-t)*1e3:.0f}ms", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata={"model": "fastwam", "action_dim": 14, "action_horizon": 48}).serve_forever()


if __name__ == "__main__":
    main()
