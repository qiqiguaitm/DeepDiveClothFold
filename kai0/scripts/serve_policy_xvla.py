"""X-VLA policy server (lerobot XVLAPolicy, R4 方案②) — multimodal-protocol conformant.

部署与训练**位字一致**: 用训练同款 `lerobot.policies.xvla.XVLAPolicy` 加载 ckpt,
并**精确复刻训练 batch 预处理** (训练绕过 processor, 直接 model.forward(dataset_batch),
见 train_scripts/xvla/{launch/xvla_train.py, data/multi_domain_dataset.py}):

  • 图像: 3 路 (top_head→image / hand_right→image2 / hand_left→image3),
    resize_pad 到 256/256/224, CHW, **float /255 ∈[0,1], 不做 ImageNet 归一**
    (processor 的 ImageNetNormalize 步骤在该训练管线**未被使用**)。
  • proprio: observation.state(20D EE6D) = joint_to_ee6d_row(当前 14D 关节),
    与训练同一函数 (PiperFK link6 + interleaved rot6d + 二值 gripper) → **不需要 ee_pose 输入**。
  • language: facebook/bart-large tokenizer, max_length=50, 固定 deploy prompt。
  • domain_id: force 20 (vis) — 见 curriculum §6。

对外 emit `action_kind="ee"` 16D (per-arm world xyz + quat_wxyz + gripper_m), 复用
现有 policy_inference_node `--execution-mode ee_pose` 客户端 (见
docs/deployment/multimodal_inference_protocol.md + xvla_inference_bringup.md)。

模型 20D EE6D 输出在 arm-base / link6 系 (joint_to_ee6d 用 CalFK link6) → 经
T_world_base{L,R} (config/calibration.yml) 合成到 world 16D。codec (interleaved
Rot6D) 见 xvla_action_codec.py。

⚠️ 旧 x3a/b/c_stage_a ckpt 用 buggy 管线 (block rot6d) 训练, 仅可用本 server 做
**加载/形状联调**, 动作不正确, 勿上真机 (见 xvla_inference_bringup.md §2.1)。

Run (用 .venv_xvla, 见 sim01_deployment.md §3.6):
    kai0/.venv_xvla/bin/python kai0/scripts/serve_policy_xvla.py \
        --ckpt_dir /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_x3c_smooth800_step_final \
        --port 8003
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "train_scripts" / "xvla" / "data"))

import cv2  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from openpi_client import base_policy as _base_policy  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from xvla_action_codec import interleaved_6d_to_rotation_matrix  # noqa: E402
from joint_to_ee6d import joint_to_ee6d_row  # noqa: E402 — 与训练同一 proprio/action 编码

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402

logger = logging.getLogger("xvla_server")

_DEFAULT_BASE_CFG = _REPO_ROOT / "kai0" / "assets" / "xvla" / "lerobot_base"  # 含 lerobot config.json
_DEFAULT_CALIBRATION = _REPO_ROOT / "config" / "calibration.yml"

# 训练 batch 键 (lerobot constants); image 顺序 = dataset preferred [top_head, hand_right, hand_left]
_IMG_KEYS = ["observation.images.image", "observation.images.image2", "observation.images.image3"]
_IMG_SIZES = [256, 256, 224]                  # main/main/wrist, 与 multi_domain_dataset 一致
_OBS_SLOT_FOR_IMG = ["top_head", "hand_right", "hand_left"]
_STATE_KEY = "observation.state"
_LANG_KEY = "observation.language.tokens"


# ──────────────────────────────────────────────────────────────────────────
# 预处理 (镜像 multi_domain_dataset.py)
# ──────────────────────────────────────────────────────────────────────────


def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Resize 保持长宽比 + 0 填充 → (size,size,3) uint8。与 dataset resize_pad 一致。"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    pt, pl = (size - nh) // 2, (size - nw) // 2
    out[pt:pt + nh, pl:pl + nw] = resized
    return out


def _to_hwc_uint8(arr, slot: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))  # CHW → HWC
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"image '{slot}' must be HWC/CHW 3ch; got {a.shape}")
    return a.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# 20D arm-base EE6D → 16D world [xyz, quat_wxyz, grip]
# ──────────────────────────────────────────────────────────────────────────


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _ee6d_to_world8(ee10: np.ndarray, T_world_base: np.ndarray,
                    g_open: float, g_close: float,
                    binarize: bool, thr: float) -> np.ndarray:
    xyz_b = np.asarray(ee10[:3], dtype=np.float64)
    R_b = interleaved_6d_to_rotation_matrix(np.asarray(ee10[3:9], dtype=np.float64))
    T = np.eye(4); T[:3, :3] = R_b; T[:3, 3] = xyz_b
    Tw = T_world_base @ T
    xyz_w = Tw[:3, 3].astype(np.float32)
    quat = _R_to_quat_wxyz(Tw[:3, :3])
    sig = float(ee10[9])
    grip = (g_close if sig > thr else g_open) if binarize else g_open + sig * (g_close - g_open)
    return np.concatenate([xyz_w, quat, np.array([grip], dtype=np.float32)])


def _load_calibration(path: Path):
    cfg = yaml.safe_load(open(path))
    tfs = cfg.get("transforms", {})
    if "T_world_baseL" not in tfs or "T_world_baseR" not in tfs:
        raise KeyError(f"{path}: transforms.T_world_base{{L,R}} missing")
    return (np.array(tfs["T_world_baseL"], dtype=np.float64),
            np.array(tfs["T_world_baseR"], dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────────
# Pipeline trace (opt-in via XVLA_TRACE_DIR) — 与 client 侧同款 _PipeTrace。
# 所有写盘 try/except 包裹: tracing 永不影响 server 推理。trace 关时根本不构造。
# 落盘 schema 见 xvla/analyze_pipeline_trace.py。
# ──────────────────────────────────────────────────────────────────────────


class _PipeTrace:
    def __init__(self, root: str, side: str):
        self.side = side
        self.arr_dir = os.path.join(root, f"{side}_arrays")
        self.img_dir = os.path.join(root, f"{side}_images")
        os.makedirs(self.arr_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        self._f = open(os.path.join(root, f"{side}_trace.jsonl"), "a", buffering=1)

    def event(self, **rec):
        rec.setdefault("t_wall", time.time())
        try:
            self._f.write(json.dumps(rec, default=float, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def save_arrays(self, seq, **arrs):
        try:
            np.savez_compressed(os.path.join(self.arr_dir, f"{int(seq):06d}.npz"), **arrs)
        except Exception:
            pass

    def save_image(self, seq, name, img):
        try:
            a = np.asarray(img)
            if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
                a = np.transpose(a, (1, 2, 0))
            cv2.imwrite(os.path.join(self.img_dir, f"{int(seq):06d}_{name}.jpg"),
                        cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Policy adapter
# ──────────────────────────────────────────────────────────────────────────


class XVLAServerPolicy(_base_policy.BasePolicy):
    def __init__(self, policy: XVLAPolicy, device, dtype, tokenizer,
                 default_prompt, default_domain_id, T_world_baseL, T_world_baseR,
                 g_open, g_close, binarize, seed=42,
                 proprio_feedback=True, proprio_resync=0.15):
        self._p = policy
        self._device = device
        self._dtype = dtype
        self._seed = seed   # 确定性采样种子; None=随机 (每次重采)
        # ① 预测式 proprio (上游 SoftFold-Agilex trick): 用"上一次预测末步"当下次 proprio,
        # 而非实测关节 → 去传感噪声/臂滞后, 连续 chunk 一致 (平滑)。漂移超 resync 则回实测。
        self._proprio_fb = bool(proprio_feedback)
        self._proprio_resync = float(proprio_resync)
        self._pred_proprio = None   # 上一 chunk 末步 20D EE6D
        self._tok = tokenizer
        self._default_prompt = default_prompt
        self._domain_id = int(default_domain_id)
        self._TL, self._TR = T_world_baseL, T_world_baseR
        self._g_open, self._g_close, self._binarize, self._thr = g_open, g_close, binarize, 0.5
        self._chunk = int(policy.config.chunk_size)
        # 固定 prompt 的 token 缓存 (与训练 cached_tokens 同: max_length=50)
        self._tok_cache: Dict[str, torch.Tensor] = {}
        # pipeline trace (opt-in): XVLA_TRACE_DIR 置位才落盘, 否则 self._trace=None → 零开销
        self._infer_n = 0
        self._trace = None
        _tdir = os.environ.get("XVLA_TRACE_DIR")
        if _tdir:
            try:
                self._trace = _PipeTrace(_tdir, "server")
                logger.warning("[trace] pipeline trace ON → %s", _tdir)
            except Exception as e:  # noqa: BLE001
                logger.warning("[trace] init failed: %s", e)

    def _tokens(self, prompt: str) -> torch.Tensor:
        if prompt not in self._tok_cache:
            ids = self._tok([prompt], padding="max_length", max_length=50,
                            truncation=True, return_tensors="pt")["input_ids"]
            self._tok_cache[prompt] = ids.to(self._device)
        return self._tok_cache[prompt]

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.monotonic()
        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError(f"obs['images'] must be dict; got {type(images).__name__}")
        state14 = obs.get("state")
        if state14 is None:
            raise ValueError("obs['state'] (14D joint) required for EE6D proprio")
        state14 = np.asarray(state14, dtype=np.float32).reshape(-1)
        if state14.shape[-1] < 14:
            raise ValueError(f"obs['state'] must be >=14D; got {state14.shape}")

        batch: Dict[str, torch.Tensor] = {}
        _trace_imgs: Dict[str, np.ndarray] = {}   # 模型实际输入图 (resize_pad 后) for trace
        # 图像: top_head/hand_right/hand_left → image/image2/image3, resize_pad, /255 (无 ImageNet 归一)
        for key, size, slot in zip(_IMG_KEYS, _IMG_SIZES, _OBS_SLOT_FOR_IMG):
            arr = images.get(slot)
            if arr is None:
                raise ValueError(f"obs['images'] missing '{slot}' (need top_head/hand_right/hand_left)")
            hwc = _resize_pad(_to_hwc_uint8(arr, slot), size)
            if self._trace is not None:
                _trace_imgs[slot] = hwc
            t = torch.from_numpy(hwc).permute(2, 0, 1).float().div_(255.0)  # (3,H,W) ∈[0,1]
            batch[key] = t.unsqueeze(0).to(self._device, dtype=self._dtype)

        # proprio: 20D EE6D。① 预测式 proprio (上游): 用上次预测末步, 而非实测关节。
        # 实测仍算出来用于首帧初始化 + 漂移保护 (pred 偏离实测 EE 位置过大→resync 回实测)。
        # 注: 上游靠"开环跑完整 chunk"保证末步≈当前; 本管线连续重推时末步有前瞻, 由 resync
        #     兜底, 完整对齐需配合 ② 开环执行 (见 xvla_inference_bringup)。
        state20_sensed = joint_to_ee6d_row(state14[:14]).astype(np.float32)
        if self._proprio_fb and self._pred_proprio is not None:
            dpos = max(float(np.linalg.norm(self._pred_proprio[0:3] - state20_sensed[0:3])),
                       float(np.linalg.norm(self._pred_proprio[10:13] - state20_sensed[10:13])))
            if dpos > self._proprio_resync:
                logger.warning("[proprio] pred 偏离实测 %.0fmm > %.0fmm → resync 回实测",
                               dpos * 1000, self._proprio_resync * 1000)
                state20 = state20_sensed
                self._pred_proprio = None
            else:
                state20 = self._pred_proprio
        else:
            state20 = state20_sensed
        batch[_STATE_KEY] = torch.from_numpy(state20).unsqueeze(0).to(self._device, dtype=self._dtype)

        # domain_id + language
        domain_id = int(obs.get("dataset_id", self._domain_id))
        batch["domain_id"] = torch.tensor([domain_id], dtype=torch.long, device=self._device)
        prompt = obs.get("prompt") or self._default_prompt
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        batch[_LANG_KEY] = self._tokens(prompt)

        # 推理 (flow-matching ODE) → (1, chunk, 20)
        # 确定性采样: generate_actions 用 torch.randn 起 flow-matching 噪声, 默认每次推理
        # 重采 → 同 obs 跨次输出差异大 (实测 ~55mm), 闭环里表现为 chunk 间目标乱跳→真机卡顿。
        # 每次 infer 前固定种子 → 噪声恒定 → 输出成为 obs 的确定函数, 连续 chunk 一致。
        # (X-VLA 走独立 .venv_xvla 纯 PyTorch, 无 V1 Triton 的 determinism/NaN 问题。)
        if self._seed is not None:
            torch.manual_seed(self._seed)
            if self._device.type == "cuda":
                torch.cuda.manual_seed_all(self._seed)
        infer_t0 = time.monotonic()
        with torch.inference_mode():
            chunk = self._p.predict_action_chunk(batch)
        infer_ms = (time.monotonic() - infer_t0) * 1000.0
        acts = chunk.squeeze(0).float().cpu().numpy()  # (chunk, 20)

        # ① 记下本 chunk 末步 (20D EE6D) 作下次 proprio (上游 pred_proprio=action_plan[-1])
        if self._proprio_fb:
            self._pred_proprio = acts[-1].astype(np.float32).copy()

        # 20D arm-base EE6D → 16D world
        H = acts.shape[0]
        out16 = np.empty((H, 16), dtype=np.float32)
        for h in range(H):
            out16[h, 0:8] = _ee6d_to_world8(acts[h, 0:10], self._TL,
                                            self._g_open, self._g_close, self._binarize, self._thr)
            out16[h, 8:16] = _ee6d_to_world8(acts[h, 10:20], self._TR,
                                             self._g_open, self._g_close, self._binarize, self._thr)

        total_ms = float((time.monotonic() - t0) * 1000.0)

        # pipeline trace: 落 server 侧一条 (收到的 state14/算出 state20/原始20D/world16/模型输入图)
        if self._trace is not None:
            try:
                self._infer_n += 1
                cseq = obs.get("trace_seq")
                # 有 client seq 就用它 (跨进程 join); 没有 (如 warmup infer) 用负数, 避免与
                # client seq=1 撞键 (warmup 不走 client 主循环, 不该和真 infer 同 seq)。
                seq = int(cseq) if cseq is not None else -self._infer_n
                self._trace.event(
                    stage="server_infer", seq=seq, client_seq=cseq, infer_n=self._infer_n,
                    t_mono=time.monotonic(), infer_ms=float(infer_ms), total_ms=total_ms,
                    domain_id=int(domain_id), prompt=str(prompt), chunk_h=int(H),
                    proprio_source=("pred" if (self._proprio_fb and self._pred_proprio is not None
                                               and state20 is not state20_sensed) else "sensed"),
                    state14=state14.tolist(), state20=state20.tolist(),
                    out20_min=float(acts.min()), out20_max=float(acts.max()),
                    world16_xyzL_h0=out16[0, 0:3].tolist(), world16_xyzR_h0=out16[0, 8:11].tolist(),
                    quat_norm_h0=[float(np.linalg.norm(out16[0, 3:7])),
                                  float(np.linalg.norm(out16[0, 11:15]))],
                    img_sizes={k: list(v.shape) for k, v in _trace_imgs.items()},
                )
                self._trace.save_arrays(seq, state14=state14, state20=state20, raw20=acts, world16=out16)
                for slot, hwc in _trace_imgs.items():
                    self._trace.save_image(seq, slot, hwc)
            except Exception as e:  # noqa: BLE001
                logger.warning("[trace] server record failed: %s", e)

        return {
            "actions": out16,
            "action_kind": "ee",
            "server_timing": {"infer_ms": float(infer_ms), "total_ms": total_ms},
        }

    def reset(self) -> None:
        self._pred_proprio = None   # ① 新 episode 清掉预测 proprio, 下帧从实测重新初始化
        if hasattr(self._p, "reset"):
            self._p.reset()


# ──────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────


def _load_policy(ckpt_dir: Path, base_cfg_dir: Path, device, dtype) -> XVLAPolicy:
    config = PreTrainedConfig.from_pretrained(str(base_cfg_dir))  # type:xvla → XVLAConfig
    config.device = str(device)
    config.pretrained_path = str(base_cfg_dir)
    # XVLAConfig.dtype drives the model's INTERNAL dtype: generate_actions samples the
    # flow-matching noise with dtype=_get_target_dtype() (=bf16 only if config.dtype=="bfloat16",
    # else fp32) and _apply_dtype() casts params to it. If we leave config.dtype at its "float32"
    # default but .to(bf16) the params below, generate_actions makes fp32 noise → matmul against
    # bf16 DomainAwareLinear weight raises "expected Float but found BFloat16". Keep them in sync.
    # XVLAConfig only supports {"bfloat16","float32"}.
    config.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"
    policy = XVLAPolicy(config)
    sd_path = ckpt_dir / "state_dict.pt"
    if not sd_path.is_file():
        raise FileNotFoundError(sd_path)
    raw = torch.load(sd_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "model_state" not in raw:
        raise ValueError(f"{sd_path}: expected {{'model_state',...}}, got keys {list(raw)[:4]}")
    ms = raw["model_state"]
    # 训练存的是 XVLAPolicy.state_dict() (键含 'model.' 前缀), 直接 load
    res = policy.load_state_dict(ms, strict=False)
    logger.info("state_dict loaded (step=%s, missing=%d, unexpected=%d)",
                raw.get("step"), len(res.missing_keys), len(res.unexpected_keys))
    if res.missing_keys:
        logger.warning("missing[:3]=%s", res.missing_keys[:3])
    if res.unexpected_keys:
        logger.warning("unexpected[:3]=%s", res.unexpected_keys[:3])
    return policy.to(device, dtype=dtype).eval()


def _load_sidecar(ckpt_dir: Path) -> Dict[str, Any]:
    p = ckpt_dir / "sidecar.json"
    return json.load(open(p)) if p.is_file() else {}


def _parse_args():
    p = argparse.ArgumentParser(__doc__.split("\n", 1)[0])
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--base_config", type=Path, default=_DEFAULT_BASE_CFG,
                   help="lerobot XVLAConfig 目录 (含 config.json, 来自 xvla-base)")
    p.add_argument("--calibration_yaml", type=Path, default=_DEFAULT_CALIBRATION)
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--default_prompt", type=str, default=None)
    p.add_argument("--default_dataset_id", type=int, default=None)
    p.add_argument("--gripper_open_value", type=float, default=0.06557999688386917)
    p.add_argument("--gripper_close_value", type=float, default=-0.0054700000174343586)
    p.add_argument("--binarize_gripper", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42,
                   help="flow-matching 确定性采样种子 (固定→连续 chunk 一致, 减卡顿); -1=随机重采")
    p.add_argument("--proprio_feedback", action=argparse.BooleanOptionalAction, default=False,
                   help="server 端预测式 proprio (用上次预测末步当 proprio)。默认【关】: 改由 client node "
                        "走 commanded-proprio (上次下发命令当 proprio, 连续模式正确时序)。开此与 node 版冲突, 勿同开")
    p.add_argument("--proprio_resync", type=float, default=0.15,
                   help="预测 proprio 偏离实测 EE 位置超此 (m) 则 resync 回实测 (漂移保护)")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    if args.dtype == "float16":
        raise SystemExit("--dtype float16 unsupported: XVLAConfig.dtype only accepts bfloat16/float32 "
                         "(see _load_policy dtype-sync note). Use bfloat16 or float32.")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sidecar = _load_sidecar(args.ckpt_dir)
    default_prompt = (args.default_prompt or sidecar.get("deploy_prompt") or "Flatten and fold the cloth.")
    default_domain = (args.default_dataset_id if args.default_dataset_id is not None
                      else int(sidecar.get("deploy_domain_id", 20)))
    logger.info("ckpt=%s | prompt=%r | domain_id=%d | source=%s",
                args.ckpt_dir.name, default_prompt, default_domain, sidecar.get("source", "?"))

    TL, TR = _load_calibration(args.calibration_yaml)
    policy = _load_policy(args.ckpt_dir, args.base_config, device, dtype)
    logger.info("XVLAPolicy loaded (%.1fM params, dtype=%s)",
                sum(p.numel() for p in policy.parameters()) / 1e6, args.dtype)
    tok = AutoTokenizer.from_pretrained("facebook/bart-large")

    srv_policy = XVLAServerPolicy(
        policy=policy, device=device, dtype=dtype, tokenizer=tok,
        default_prompt=default_prompt, default_domain_id=default_domain,
        T_world_baseL=TL, T_world_baseR=TR,
        g_open=args.gripper_open_value, g_close=args.gripper_close_value,
        binarize=args.binarize_gripper, seed=(None if args.seed < 0 else args.seed),
        proprio_feedback=args.proprio_feedback, proprio_resync=args.proprio_resync)

    metadata = {
        "action_kind": "ee",
        "action_dim": 16,
        "action_horizon": int(policy.config.chunk_size),
        "obs_keys": ["images.top_head", "images.hand_left", "images.hand_right", "state", "prompt"],
        "model_name": f"xvla_lerobot::{args.ckpt_dir.name}",
        "xvla_domain_id": default_domain,
        "ckpt_step": int(sidecar.get("step", -1)),
    }
    logger.info("Serving X-VLA (lerobot) on ws://%s:%d (action_kind=ee, dim=16, H=%d, host=%s)",
                args.host, args.port, metadata["action_horizon"], socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(
        policy=srv_policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
