#!/usr/bin/env python3
"""Offline vision-ablation for an X-VLA ckpt, replayed from a --trace dump.

Question: does the model's action actually depend on the camera image, or is it
open-loop (action ≈ f(proprio) alone, vision ignored)? A model that "executes a
fixed grasp regardless of whether the cloth is there" will show ~0 action change
when the image is swapped/blanked while the proprio state is held fixed.

Method (seed fixed + proprio_feedback OFF → each infer is an independent,
deterministic function of (image, state)):
  For sampled frames i, reconstruct obs from server_images/*.jpg + server_arrays/*.npz.
    d_img   = ‖A(img_i, state_i) − A(img_j, state_i)‖   swap IMAGE, hold state
    d_blank = ‖A(img_i, state_i) − A(black, state_i)‖   blank IMAGE, hold state
    d_state = ‖A(img_i, state_i) − A(img_i, state_k)‖   swap STATE, hold image
  Reported per chunk-step, separately for EE xyz (m) and gripper.
  vision/proprio influence ratio = d_img / d_state. ~0 → vision-blind (open-loop).

Usage (must use the X-VLA venv):
  kai0/.venv_xvla/bin/python train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py \
      --trace /tmp/xvla_stack/trace_<ts> \
      --ckpt /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_x3c_smooth800_d5anchor_step_final \
      --n 12
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch

# import the actual serve building blocks so preprocessing is byte-identical to deploy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "kai0", "scripts"))
import serve_policy_xvla as S  # noqa: E402

# E1 (use_proprio=False ckpts): lerobot EE6DActionSpace.preprocess indexes proprio gripper
# channels → IndexError when proprio dim=0. Guard the empty case (no-op for proprio>0).
from lerobot.policies.xvla import action_hub as _ah  # noqa: E402
_orig_ee6d_pre = _ah.EE6DActionSpace.preprocess
def _ee6d_pre_safe(self, proprio, action, mode="train"):
    if proprio.shape[-1] == 0:
        am = action.clone(); am[..., self.gripper_idx] = 0.0
        return proprio, am
    return _orig_ee6d_pre(self, proprio, action, mode)
_ah.EE6DActionSpace.preprocess = _ee6d_pre_safe


def _load_policy_noproprio(ckpt, base_cfg, device, dtype):
    """Build XVLAPolicy with use_proprio=False (proprio_dim=0) so an E1 ckpt (sliced
    action_encoder.fc) loads with matching shapes; serve _load_policy assumes proprio."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    config = PreTrainedConfig.from_pretrained(str(base_cfg))
    config.device = str(device); config.pretrained_path = str(base_cfg)
    config.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"
    config.use_proprio = False
    policy = XVLAPolicy(config)
    raw = torch.load(ckpt / "state_dict.pt", map_location="cpu", weights_only=True)
    res = policy.load_state_dict(raw["model_state"], strict=False)
    print(f"[no-proprio] use_proprio=False load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return policy.to(device, dtype=dtype).eval()


CAMS = ("top_head", "hand_right", "hand_left")


def _load_obs(trace, seq):
    import cv2
    imgs = {}
    for c in CAMS:
        p = os.path.join(trace, "server_images", f"{seq:06d}_{c}.jpg")
        bgr = cv2.imread(p)
        if bgr is None:
            return None
        imgs[c] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = os.path.join(trace, "server_arrays", f"{seq:06d}.npz")
    if not os.path.isfile(arr):
        return None
    state14 = np.load(arr)["state14"].astype(np.float32)
    return {"images": imgs, "state": state14}


def _xyz_grip(out16):
    """out16 (H,16) → (xyz_LR (H,6), grip_LR (H,2))."""
    xyz = np.concatenate([out16[:, 0:3], out16[:, 8:11]], axis=1)
    grip = np.stack([out16[:, 7], out16[:, 15]], axis=1)
    return xyz, grip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12, help="frames to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-proprio", action="store_true",
                    help="ckpt trained with use_proprio=False (E1) — build proprio_dim=0 model")
    ap.add_argument("--imagenet-norm", choices=["auto", "true", "false"], default="auto",
                    help="override sidecar image_norm (E1 ckpt has no sidecar; p0 lineage => true)")
    ap.add_argument("--prompt", default=None, help="override deploy prompt")
    args = ap.parse_args()

    from pathlib import Path
    dtype = torch.float32
    device = torch.device(args.device)
    ckpt = Path(args.ckpt)
    sidecar = S._load_sidecar(ckpt)
    if args.imagenet_norm == "auto":
        imagenet_norm = str(sidecar.get("image_norm", "")).lower() == "imagenet"
    else:
        imagenet_norm = args.imagenet_norm == "true"
    TL, TR = S._load_calibration(S._DEFAULT_CALIBRATION)
    policy = (_load_policy_noproprio(ckpt, S._DEFAULT_BASE_CFG, device, dtype)
              if args.no_proprio else S._load_policy(ckpt, S._DEFAULT_BASE_CFG, device, dtype))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("XVLA_BART_TOK", "facebook/bart-large"))
    # proprio_feedback=False → no cross-call state carryover; seed fixed → deterministic.
    srv = S.XVLAServerPolicy(
        policy=policy, device=device, dtype=dtype, tokenizer=tok,
        default_prompt=args.prompt or sidecar.get("deploy_prompt", "Flatten and fold the cloth."),
        default_domain_id=int(sidecar.get("deploy_domain_id", 20)),
        T_world_baseL=TL, T_world_baseR=TR, g_open=0.08, g_close=0.0,
        binarize=False, seed=args.seed, proprio_feedback=False, imagenet_norm=imagenet_norm)
    print(f"imagenet_norm={imagenet_norm}  seed={args.seed}  proprio_feedback=OFF")

    # collect available seqs
    seqs = sorted(int(os.path.basename(f)[:6]) for f in glob.glob(f"{args.trace}/server_arrays/0*.npz"))
    if len(seqs) < 3:
        raise SystemExit(f"too few frames in {args.trace}")
    pick = seqs[:: max(1, len(seqs) // args.n)][: args.n]
    print(f"frames: {len(seqs)} total, sampling {len(pick)}: {pick}")

    def infer(imgs, state):
        srv.reset()
        out = srv.infer({"images": imgs, "state": state})
        return out["actions"]  # (H,16)

    d_img_xyz, d_blank_xyz, d_state_xyz = [], [], []
    d_img_grip, d_blank_grip, d_state_grip = [], [], []
    obss = {s: _load_obs(args.trace, s) for s in pick}
    obss = {s: o for s, o in obss.items() if o is not None}
    keys = list(obss)
    for n, i in enumerate(keys):
        j = keys[(n + len(keys) // 2) % len(keys)]   # a far-apart different frame
        oi, oj = obss[i], obss[j]
        A_real = infer(oi["images"], oi["state"])
        A_img = infer(oj["images"], oi["state"])               # swap IMAGE, hold state_i
        black = {c: np.zeros_like(v) for c, v in oi["images"].items()}
        A_blank = infer(black, oi["state"])                     # blank IMAGE, hold state_i
        A_state = infer(oi["images"], oj["state"])              # swap STATE, hold image_i

        for (acc, A2) in [(0, A_img), (1, A_blank), (2, A_state)]:
            xz0, g0 = _xyz_grip(A_real)
            xz1, g1 = _xyz_grip(A2)
            dx = float(np.linalg.norm(xz0 - xz1, axis=1).mean())   # mean over chunk steps, m
            dg = float(np.abs(g0 - g1).mean())
            [d_img_xyz, d_blank_xyz, d_state_xyz][acc].append(dx)
            [d_img_grip, d_blank_grip, d_state_grip][acc].append(dg)

    def stat(a):
        a = np.array(a)
        return f"{a.mean()*1000:7.2f}mm" if a.mean() < 1 else f"{a.mean():.3f}"

    print("\n── 动作变化 (mean over chunk, 对 N 帧平均) ──")
    print(f"  swap IMAGE,hold state  d_img   : xyz={np.mean(d_img_xyz)*1000:7.2f}mm  grip={np.mean(d_img_grip):.4f}")
    print(f"  BLANK image,hold state d_blank : xyz={np.mean(d_blank_xyz)*1000:7.2f}mm  grip={np.mean(d_blank_grip):.4f}")
    print(f"  swap STATE,hold image  d_state : xyz={np.mean(d_state_xyz)*1000:7.2f}mm  grip={np.mean(d_state_grip):.4f}")
    r_xyz = np.mean(d_img_xyz) / max(1e-9, np.mean(d_state_xyz))
    r_grip = np.mean(d_img_grip) / max(1e-9, np.mean(d_state_grip))
    print("\n── 视觉/本体影响比 (d_img / d_state) ──")
    print(f"  xyz : {r_xyz:.3f}   grip : {r_grip:.3f}")
    print("  解读: →0 = 换图几乎不改动作, 模型靠 proprio 开环 (vision-blind);")
    print("        ~1 = 图像与本体影响相当 (健康闭环)。")


if __name__ == "__main__":
    main()
