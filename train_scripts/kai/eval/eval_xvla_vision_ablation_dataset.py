#!/usr/bin/env python3
"""Dataset-based vision-ablation for an X-VLA ckpt (no deploy trace needed).

Same question as eval_xvla_vision_ablation_offline.py — does the action depend on
the camera image, or is it open-loop (action ≈ f(proprio), vision ignored)? — but
sources obs frames from the *training* v1 dataset (LeRobotEE6DDataset) instead of a
recorded deploy trace, because the EE6D dataset stores 20D proprio (not raw 14D
joints) so it cannot be replayed through serve_policy_xvla.infer (which expects 14D
+ joint_to_ee6d). Here we build the model batch directly (byte-identical to the
training collate: imagenet-norm images, 20D state, BART tokens, domain_id) and probe
predict_action_chunk with a fixed flow-matching seed → each infer is a deterministic
function of (images, state).

  For sampled frames i (far frame j):
    d_img   = ‖A(img_i,state_i) − A(img_j,state_i)‖   swap IMAGE, hold state
    d_blank = ‖A(img_i,state_i) − A(black,state_i)‖   blank IMAGE, hold state
    d_state = ‖A(img_i,state_i) − A(img_i,state_j)‖   swap STATE, hold image
  vision/proprio influence ratio = d_img / d_state. ~0 → vision-blind (open-loop).

Run (X-VLA venv on gf0):
  xvla/X-VLA-env/.venv/bin/python train_scripts/kai/eval/eval_xvla_vision_ablation_dataset.py \
      --ckpt xvla/ckpts/xvla_e0_v1_official/step_final \
      --data-root xvla/data/self_built/A_v1_noRelabel_ee6d/2026-04-23 --n 12
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "kai0" / "scripts"))
sys.path.insert(0, str(_REPO / "train_scripts" / "xvla" / "data"))
import serve_policy_xvla as S  # noqa: E402
from multi_domain_dataset import LeRobotEE6DDataset, imagenet_normalize_chw  # noqa: E402

IMG_KEYS = ["observation.images.image", "observation.images.image2", "observation.images.image3"]


def _xyz_grip(a):  # a:(H,20) ee6d → xyz_LR (H,6) m, grip_LR (H,2)
    xyz = np.concatenate([a[:, 0:3], a[:, 10:13]], axis=1)
    grip = np.stack([a[:, 9], a[:, 19]], axis=1)
    return xyz, grip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True, help="one date dir of the v1 EE6D dataset")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain-id", type=int, default=20)
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--base-config", default=str(S._DEFAULT_BASE_CFG))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    policy = S._load_policy(Path(args.ckpt), Path(args.base_config), device, dtype)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("XVLA_BART_TOK", "facebook/bart-large"))
    lang = tok([args.prompt], padding="max_length", max_length=50, truncation=True,
               return_tensors="pt")["input_ids"].to(device)  # (1,50), same as training collate
    print(f"prompt={args.prompt!r}  domain_id={args.domain_id}  seed={args.seed}")

    # image_aug=False → no ColorJitter → deterministic preprocessing
    ds = LeRobotEE6DDataset(root=args.data_root, domain_id=args.domain_id, task_prompt=args.prompt,
                            action_qdur=2.0, static_skip=False, image_aug=False)
    N = len(ds)
    pick = list(range(0, N, max(1, N // args.n)))[: args.n]
    print(f"dataset frames: {N}, sampling {len(pick)}: {pick}")

    def gen(imgs, state):
        batch = {k: imgs[k].unsqueeze(0).to(device, dtype) for k in IMG_KEYS}
        batch["observation.state"] = state.unsqueeze(0).to(device, dtype)
        batch["domain_id"] = torch.tensor([args.domain_id], dtype=torch.long, device=device)
        batch["observation.language.tokens"] = lang
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        if hasattr(policy, "reset"):
            policy.reset()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)  # (1,H,20)
        return chunk.squeeze(0).float().cpu().numpy()

    def load(idx):
        d = ds[idx]
        return {k: d[k] for k in IMG_KEYS}, d["observation.state"]

    def black(imgs):  # camera-sees-black in normalized space = normalize(zeros)
        return {k: imagenet_normalize_chw(torch.zeros_like(imgs[k])) for k in IMG_KEYS}

    samples = {i: load(i) for i in pick}
    keys = list(samples)
    di, db, dst, gi, gb, gst = [], [], [], [], [], []
    for n, i in enumerate(keys):
        j = keys[(n + len(keys) // 2) % len(keys)]
        (imi, sti), (imj, stj) = samples[i], samples[j]
        Ar = gen(imi, sti)
        Aimg = gen(imj, sti)
        Ablk = gen(black(imi), sti)
        Ast = gen(imi, stj)
        for A2, dl, gl in [(Aimg, di, gi), (Ablk, db, gb), (Ast, dst, gst)]:
            x0, g0 = _xyz_grip(Ar)
            x1, g1 = _xyz_grip(A2)
            dl.append(float(np.linalg.norm(x0 - x1, axis=1).mean()))
            gl.append(float(np.abs(g0 - g1).mean()))

    print("\n── 动作变化 (mean over chunk, 对 N 帧平均) ──")
    print(f"  swap IMAGE,hold state  d_img   : xyz={np.mean(di)*1000:7.2f}mm  grip={np.mean(gi):.4f}")
    print(f"  BLANK image,hold state d_blank : xyz={np.mean(db)*1000:7.2f}mm  grip={np.mean(gb):.4f}")
    print(f"  swap STATE,hold image  d_state : xyz={np.mean(dst)*1000:7.2f}mm  grip={np.mean(gst):.4f}")
    r_xyz = np.mean(di) / max(1e-9, np.mean(dst))
    r_grip = np.mean(gi) / max(1e-9, np.mean(gst))
    print("\n── 视觉/本体影响比 (d_img / d_state) ──")
    print(f"  xyz : {r_xyz:.3f}   grip : {r_grip:.3f}")
    print("  解读: →0 = 换图几乎不改动作, 靠 proprio 开环 (vision-blind); ~1 = 健康闭环。")


if __name__ == "__main__":
    main()
