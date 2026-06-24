"""FLASH speculative policy server for kai0 pi05 (PyTorch) — server-only, additive.

部署与训练**位字一致**: 加载标准 PyTorch pi05 ckpt (与 `create_trained_policy` 完全
同款的 transforms / norm_stats / model), 然后**只替换** `Policy._sample_actions` 这一个
seam, 把原来的 10-step flow-matching denoise 换成 FLASH 投机一轮:

    prefill VLM KV 一次  ->  DraftChunkHead 一次前向给出整条 chunk x0_draft  ->  K-way
    "verify-from-draft" (x_t=t·noise+(1-t)·x0_draft, 每个 t 一次 denoise_step,
    x0_hat=x_t-t·v_t)  ->  radius 前缀接受  ->  双臂夹爪相位门  ->  拼接 draft 前缀 +
    verified 尾巴。接受太少 → full_fallback 退回**与标准模型逐字相同**的多步 denoise。

对外仍 emit 标准 `action_kind="joint"` 14D (经标准 output_transforms: Unnormalize +
AgilexOutputs 切片), **复用现有 `policy_inference_node --mode websocket` 客户端, 无任何
ROS2 侧改动** (与 serve_policy_xvla.py / start_serve_v1.sh 同款 server-only 模式)。

安全性:
  • full_fallback 默认开 → 低接受帧退回标准 denoise, 输出≈baseline (FLASH 不会让动作变坏)。
  • spec 路径任何异常 → 自动降级到 eager `full_denoise_from_observation` (5090-safe), 永不 brick 真机。
  • draft head 是**逐 ckpt** 自蒸馏产物 (见 train_scripts/kai/eval/spec_draft_r1d.py),
    必须与所部署 ckpt 配对; 不配对会被 radius 拒绝 → 持续 fallback (慢但不错)。

⚠️ FLASH 是 LOSSY 投机 (radius 启发式作用在连续动作上, 不像 LLM spec-decode 无损)。
离线 R1-d 在 pure200 上 50/50 接受 (radius 0.018), 但**离线≠闭环**: 闭环漂移会拉低接受率,
退回 fallback。首次上机务必先 --execute 前观察接受率日志 + 与 v0 对比。

Run (用 .venv_5090, PyTorch GPU):
    CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python kai0/scripts/serve_policy_flash.py \
        --config pi05_pytorch_a_new_pure_200 \
        --dir /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
        --asset-id a_new_pure_200 \
        --draft /tmp/draft_r1d_pure200.pt \
        --port 8001

通常不直接跑本脚本, 而是经 start_scripts/kai/start_autonomy_from_ckpt_v2.sh 一键拉起。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import pathlib
import socket
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "packages" / "openpi-client" / "src"))

import numpy as np  # noqa: E402
from openpi_client import base_policy as _base_policy  # noqa: E402
import torch  # noqa: E402

from openpi.models_pytorch.draft import DraftChunkHead  # noqa: E402
from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs  # noqa: E402
from openpi.models_pytorch.spec_pi0_pytorch import SpeculativeSampler  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402

logger = logging.getLogger("flash_server")


def _build_draft(blob: dict, model, device, dtype) -> DraftChunkHead:
    """Reconstruct the DraftChunkHead from a saved blob (spec_draft_r1d.py format).

    Blob carries {"state_dict", "img_dim", "chunk_m", "out_dim"} — fully self-describing.
    img_dim must equal the VLM hidden size; we cross-check against the model so a draft
    distilled for a *different* backbone fails loudly here instead of silently degrading.
    """
    vlm_lm = model.paligemma_with_expert.paligemma.language_model
    img_dim = int(blob.get("img_dim", vlm_lm.config.hidden_size))
    if img_dim != int(vlm_lm.config.hidden_size):
        raise ValueError(
            f"draft img_dim={img_dim} != model hidden_size={vlm_lm.config.hidden_size} "
            "— this draft was distilled for a different VLM backbone; re-distill for this ckpt."
        )
    num_layers = int(blob.get("num_layers", 1))
    draft = DraftChunkHead(
        img_dim=img_dim,
        chunk_m=int(blob.get("chunk_m", model.config.action_horizon)),
        out_dim=int(blob.get("out_dim", model.config.action_dim)),
        use_state_token=False,
        num_layers=num_layers,
        gemma_config=vlm_lm.config,
    )
    missing, unexpected = draft.load_state_dict(blob["state_dict"], strict=False)
    if missing:
        logger.warning("draft missing keys[:3]=%s (total %d)", list(missing)[:3], len(missing))
    if unexpected:
        logger.warning("draft unexpected keys[:3]=%s (total %d)", list(unexpected)[:3], len(unexpected))
    if missing or unexpected:
        # A num_layers mismatch silently drops tail layers under strict=False → broken draft.
        raise SystemExit(
            f"[flash] draft state_dict mismatch (missing={len(missing)}, unexpected={len(unexpected)}); "
            f"blob num_layers={num_layers}. Likely a serve/draft num_layers mismatch — refusing to serve a "
            "partially-loaded draft (would emit a near-static / garbage chunk)."
        )
    logger.info("draft built: num_layers=%d", num_layers)
    return draft.to(device=device, dtype=dtype).eval()


def _chunk_signals(actions, arm_dims, gripper_dims) -> dict:
    """Per-chunk motion descriptors in the model's normalized action space.

    `arm_motion` (sum over arm dims of max-min over H) is the SAME metric the offline
    spec_draft_magnitude_probe.py reports — recording it here lets a real-machine trace
    quantify under-actuation that the in-distribution offline probe cannot see on OOD
    frames. gripper net/range feed flash_trace_phase_report.py's grasp/smooth split.
    """
    a = actions[0].detach().float().cpu().numpy()  # (H, action_dim)
    arm = a[:, arm_dims]
    sig = {
        "arm_motion": float((arm.max(0) - arm.min(0)).sum()),
        "arm_d_last": float(np.abs(arm[-1] - arm[0]).mean()),
    }
    gl, gr = (int(gripper_dims[0]), int(gripper_dims[1])) if len(gripper_dims) >= 2 else (None, None)
    if gl is not None:
        sig["gl_net"] = float(a[-1, gl] - a[0, gl])
        sig["gl_rng"] = float(a[:, gl].max() - a[:, gl].min())
    if gr is not None:
        sig["gr_net"] = float(a[-1, gr] - a[0, gr])
        sig["gr_rng"] = float(a[:, gr].max() - a[:, gr].min())
    return sig


def _install_flash_sampler(policy, spec: SpeculativeSampler, device, *,
                           trace_path: str | None = None, no_spec: bool = False) -> dict:
    """Replace Policy._sample_actions with the speculative round. Additive monkey-patch.

    The seam contract (policy.py:119) is `_sample_actions(device, observation, **kw) ->
    [B,H,action_dim]` in the model's normalized space — exactly what spec.sample() returns
    under "actions". On ANY spec exception we degrade to an EAGER full denoise so a FLASH
    bug can never brick the arm. NOTE: we must NOT fall back to the model's *compiled*
    sample_actions — it CUDA-graph-crashes on the 5090 (flash_impl_log.md §2.2); the
    sampler's `full_denoise_from_observation` is the eager, 5090-safe equivalent.

    no_spec=True  → baseline A/B: skip draft+verify entirely, serve pure eager 10-step
                    denoise (= what the --no-spec robot run sees). Still records arm motion
                    to the trace so spec vs no-spec chunk magnitude is directly comparable.
    trace_path    → append one JSONL row per infer (accept/fallback/radius + arm/gripper
                    motion). Off by default → bit-identical to plain v2.
    """
    counters = {"n": 0, "acc_sum": 0.0, "fallback": 0, "err": 0,
                "draft_ms": 0.0, "verify_ms": 0.0, "last_accept": float("nan")}
    ah = int(spec.action_horizon)
    grip = list(spec.args.gripper_dims)
    gset = {int(g) for g in grip}
    arm_dims = [i for i in range(min(14, int(spec.action_dim))) if i not in gset]
    trace_fh = open(trace_path, "a", buffering=1) if trace_path else None  # noqa: SIM115 (lives for server lifetime)
    if trace_fh:
        logger.info("[flash] trace → %s (no_spec=%s)", trace_path, no_spec)

    def _write_trace(row: dict) -> None:
        if trace_fh is None:
            return
        with contextlib.suppress(Exception):  # tracing must never break inference
            trace_fh.write(json.dumps(row) + "\n")

    def _nospec_sample_actions(dev, observation, **kw):
        counters["n"] += 1
        t0 = time.time()
        actions = spec.full_denoise_from_observation(observation, noise=kw.get("noise"))
        if trace_fh is not None:
            row = {"t": t0, "n": counters["n"], "H": ah, "mode": "no_spec",
                   "accept": ah, "fb": 1.0, **_chunk_signals(actions, arm_dims, grip)}
            _write_trace(row)
        return actions

    def _flash_sample_actions(dev, observation, **kw):
        counters["n"] += 1
        t0 = time.time()
        try:
            out = spec.sample(observation, noise=kw.get("noise"))
        except Exception as e:  # safety: degrade to eager full denoise, never crash the robot
            counters["err"] += 1
            logger.warning("[flash] spec.sample raised (%s) → eager full-denoise fallback", e)
            actions = spec.full_denoise_from_observation(observation, noise=kw.get("noise"))
            if trace_fh is not None:
                _write_trace({"t": t0, "n": counters["n"], "H": ah, "mode": "spec_err",
                              "accept": float("nan"), "fb": 1.0, **_chunk_signals(actions, arm_dims, grip)})
            return actions

        try:
            acc = float(out["accepted_prefix_len"].float().mean().item())
            counters["last_accept"] = acc
            counters["acc_sum"] += acc
            counters["fallback"] += int(bool(out.get("used_full_fallback")))
            counters["draft_ms"] += float(out.get("draft_ms", 0.0))
            counters["verify_ms"] += float(out.get("verify_ms", 0.0))
            if trace_fh is not None:
                rad = out.get("radius_dist")
                rad_mean = rad_max = float("nan")
                if rad is not None:
                    rmin = rad.min(dim=-1).values[0].float().cpu().numpy()
                    rad_mean, rad_max = float(rmin.mean()), float(rmin.max())
                _write_trace({"t": t0, "n": counters["n"], "H": ah, "mode": "spec",
                              "accept": acc, "fb": float(bool(out.get("used_full_fallback"))),
                              "rad_mean": rad_mean, "rad_max": rad_max,
                              "draft_ms": float(out.get("draft_ms", 0.0)),
                              "verify_ms": float(out.get("verify_ms", 0.0)),
                              **_chunk_signals(out["actions"], arm_dims, grip)})
            if counters["n"] % 20 == 0:
                n = counters["n"]
                logger.info(
                    "[flash] %d infers | mean_accept=%.1f/%d | fallback=%d (%.0f%%) | "
                    "draft=%.1fms verify=%.1fms | spec_err=%d",
                    n, counters["acc_sum"] / n, ah,
                    counters["fallback"], 100.0 * counters["fallback"] / n,
                    counters["draft_ms"] / n, counters["verify_ms"] / n, counters["err"],
                )
        except Exception:  # logging must never break inference
            pass
        return out["actions"]

    policy._sample_actions = _nospec_sample_actions if no_spec else _flash_sample_actions  # noqa: SLF001
    if no_spec:
        logger.info("[flash] --no-spec: serving EAGER full-denoise baseline (no draft/verify)")
    return counters


class _FlashHealthPolicy(_base_policy.BasePolicy):
    """Opt-in wrapper: joins per-frame FLASH acceptance with a PERIODIC vision-SNR probe.

    R5 (flash_impl_log.md §8) proved FLASH acceptance is a *self-consistency* metric —
    structurally BLIND to whether the input is informative (black the cameras and the draft
    still agrees with the model on that blacked input → acceptance unchanged). So an open-loop
    health signal needs acceptance x an INDEPENDENT vision-SNR. This wrapper supplies that SNR
    by periodically (every `probe_every` infers) re-running the model on a RAW-image-blacked
    copy of the *same* observation and measuring the output shift over arm dims:

        floor  = mean|teacher_real(n1) - teacher_real(n2)|   (denoise noise floor)
        Δblack = mean|teacher_real(n1) - teacher_black(n1)|
        SNR    = Δblack / floor        (~1 ignores vision, ≫1 uses vision)

    Faithful ablation must black BEFORE the transforms, so this wraps `Policy.infer` (which
    sees the raw obs dict) rather than the `_sample_actions` seam. Off by default → when
    `probe_every<=0` the server serves the inner Policy directly (bit-identical to plain v2).
    The probe runs INLINE on probe-frames (~3 eager denoises, adds latency that frame); pick
    `probe_every` large (e.g. 60) so the node's chunk buffer absorbs the occasional slow frame.
    """

    def __init__(self, inner, spec, counters, *, probe_every: int, arm_dims, device, model):
        self._inner = inner
        self._spec = spec
        self._counters = counters
        self._every = int(probe_every)
        self._arm = list(arm_dims)
        self._n = 0
        self._last_snr = float("nan")
        ah, ad = int(model.config.action_horizon), int(model.config.action_dim)
        g = torch.Generator(device=device).manual_seed(0)
        self._n1 = torch.randn((1, ah, ad), generator=g, device=device, dtype=torch.float32)
        self._n2 = torch.randn((1, ah, ad), generator=g, device=device, dtype=torch.float32)

    @property
    def metadata(self):
        return self._inner.metadata

    def reset(self):
        if hasattr(self._inner, "reset"):
            self._inner.reset()

    def _vision_snr(self, obs: dict) -> float:
        """Faithful raw-image-blacked vision-SNR over arm dims (mirrors spec_r5_probe)."""
        import jax

        from openpi.models import model as _model

        def _obs_to_observation(raw):
            inputs = self._inner._input_transform(raw)  # noqa: SLF001
            dev = self._inner._pytorch_device  # noqa: SLF001
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(dev)[None, ...], inputs)
            return _model.Observation.from_dict(inputs)

        imgs = obs.get("images") or {}
        black = {**obs, "images": {k: np.zeros_like(np.asarray(v)) for k, v in imgs.items()}}
        tr1 = self._spec.full_denoise_from_observation(_obs_to_observation(obs), noise=self._n1).float()
        tr2 = self._spec.full_denoise_from_observation(_obs_to_observation(obs), noise=self._n2).float()
        tb1 = self._spec.full_denoise_from_observation(_obs_to_observation(black), noise=self._n1).float()
        floor = float(torch.mean(torch.abs((tr1 - tr2)[0][:, self._arm])).item())
        dblack = float(torch.mean(torch.abs((tr1 - tb1)[0][:, self._arm])).item())
        return dblack / max(floor, 1e-9)

    def infer(self, obs: dict, **kw) -> dict:
        result = self._inner.infer(obs, **kw)
        self._n += 1
        if self._every > 0 and (self._n % self._every == 0):
            try:
                self._last_snr = self._vision_snr(obs)
                acc = self._counters.get("last_accept", float("nan"))
                ah = self._spec.action_horizon
                logger.info("[flash-health] frame %d | accept=%.1f/%d | vision-SNR=%.2fx | "
                            "joint(acceptxSNR)=%.1f  (SNR~1 ⇒ 开环嫌疑: 接受率高但模型不看画面)",
                            self._n, acc, ah, self._last_snr, (acc / ah) * self._last_snr)
            except Exception as e:  # probe must never break serving
                logger.warning("[flash-health] probe failed (%s) — disabling", e)
                self._every = 0
        result["flash_health"] = {"vision_snr": self._last_snr,
                                  "last_accept": self._counters.get("last_accept", float("nan"))}
        return result


def main() -> None:
    ap = argparse.ArgumentParser(__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True, help="base TrainConfig name (config.py)")
    ap.add_argument("--dir", required=True, type=pathlib.Path, help="checkpoint dir (has model.safetensors)")
    ap.add_argument("--asset-id", default=None, help="override_asset_id for norm_stats (assets/<id>/norm_stats.json)")
    ap.add_argument("--draft", required=True, type=pathlib.Path, help="trained DraftChunkHead blob (spec_draft_r1d.py)")
    ap.add_argument("--port", type=int, default=8001, help="websocket port (8000=JAX,8002=V1,8003=XVLA; flash=8001)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--default-prompt", default=None)
    # ---- speculative knobs (mirror SpecArgs; safe defaults) ----
    ap.add_argument("--tau", type=float, default=0.3, help="radius acceptance threshold (smaller=stricter)")
    ap.add_argument("--max-exec-steps", type=int, default=None,
                    help="radius eval window (default=full action_horizon for deployment)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="DISABLE full_fallback (unsafe: low-accept frames won't recover to baseline). Default keeps it ON.")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix flow-matching noise seed per infer (chunk-to-chunk consistency). Default None=standard random.")
    ap.add_argument("--health-probe-every", type=int, default=0,
                    help="R5: every N infers, run a raw-image-blacked vision-SNR probe + log acceptxSNR health. "
                         "0=off (default, bit-identical to plain v2). Adds ~3 eager denoises on probe-frames; "
                         "use large N (e.g. 60) so the node chunk buffer absorbs the slow frame.")
    ap.add_argument("--trace-out", default=None,
                    help="append one JSONL row per infer (accept/fallback/radius + arm/gripper chunk motion). "
                         "Off by default. Works under --no-spec too (arm motion only) for spec-vs-baseline A/B.")
    ap.add_argument("--no-spec", action="store_true",
                    help="A/B baseline: skip draft+verify, serve pure EAGER full-denoise (= --no-spec robot run). "
                         "Combine with --trace-out to record baseline arm motion for comparison.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")

    ckpt = args.dir.resolve()
    if not (ckpt / "model.safetensors").is_file():
        raise SystemExit(f"[flash] {ckpt}/model.safetensors not found — FLASH server只支持 PyTorch ckpt")
    if not args.draft.is_file():
        raise SystemExit(
            f"[flash] draft head not found: {args.draft}\n"
            "       先用 train_scripts/kai/eval/spec_draft_r1d.py 为该 ckpt 自蒸馏一个 draft head。"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[flash] CUDA requested but unavailable")

    train_config = _config.get_config(args.config)
    norm_stats = None
    if args.asset_id:
        norm_stats = _checkpoints.load_norm_stats(ckpt / "assets", args.asset_id)
        if norm_stats is None:
            raise SystemExit(f"[flash] norm_stats missing: {ckpt}/assets/{args.asset_id}/norm_stats.json")

    logger.info("loading standard pi05 policy (config=%s, ckpt=%s, asset_id=%s)…",
                args.config, ckpt.name, args.asset_id)
    policy = _policy_config.create_trained_policy(
        train_config, ckpt, default_prompt=args.default_prompt,
        norm_stats=norm_stats, pytorch_device=str(device),
    )
    model = policy._model  # noqa: SLF001
    mdtype = next(model.parameters()).dtype
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        logger.info("[flash] fixed flow-matching seed=%d (chunk-consistent noise)", args.seed)

    blob = torch.load(args.draft, map_location=device)
    draft = _build_draft(blob, model, device, mdtype)
    logger.info("draft head loaded: img_dim=%s chunk_m=%s out_dim=%s (%.1fM params)",
                blob.get("img_dim"), blob.get("chunk_m"), blob.get("out_dim"),
                sum(p.numel() for p in draft.parameters()) / 1e6)

    spec_args = SpecArgs(
        chunk_m=ah,
        max_exec_steps=(args.max_exec_steps if args.max_exec_steps is not None else ah),
        tau_radius=args.tau,
        full_fallback=not args.no_fallback,
    )
    spec = SpeculativeSampler(model, draft, spec_args)
    counters = _install_flash_sampler(policy, spec, device,
                                      trace_path=args.trace_out, no_spec=args.no_spec)
    logger.info("[flash] sampler installed (tau=%.3f, eval_h=%d, fallback=%s, no_spec=%s); warmup deferred to first infer",
                spec_args.tau_radius, spec_args.max_exec_steps, not args.no_fallback, args.no_spec)

    serve_policy = policy
    if args.health_probe_every > 0:
        grip = {int(g) for g in spec_args.gripper_dims}
        arm_dims = [i for i in range(min(14, ad)) if i not in grip]  # pi05 双臂 0-13 去夹爪 = 12
        serve_policy = _FlashHealthPolicy(policy, spec, counters, probe_every=args.health_probe_every,
                                          arm_dims=arm_dims, device=device, model=model)
        logger.info("[flash-health] R5 acceptxSNR probe ON: every %d infers a raw-blacked vision-SNR probe "
                    "(arm_dims=%s)", args.health_probe_every, arm_dims)

    metadata = dict(policy.metadata or {})
    metadata.update({
        "action_kind": "joint",
        "action_dim": 14,
        "action_horizon": ah,
        "model_action_dim": ad,
        "flash": True,
        "flash_draft": args.draft.name,
        "flash_tau": spec_args.tau_radius,
        "flash_fallback": not args.no_fallback,
        "model_name": f"flash_pi05::{ckpt.name}",
    })

    logger.info("Serving FLASH pi05 on ws://%s:%d (action_kind=joint, dim=14, H=%d, dtype=%s, host=%s)",
                args.host, args.port, ah, mdtype, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(
        policy=serve_policy, host=args.host, port=args.port, metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
