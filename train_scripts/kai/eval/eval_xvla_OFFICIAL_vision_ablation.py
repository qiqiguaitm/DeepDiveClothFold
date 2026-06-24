#!/usr/bin/env python3
"""Vision-ablation for the OFFICIAL 2toINF X-VLA-SoftFold raw checkpoint.

Question (positive control for our gate): does the *official* X-VLA model read the
camera image, or is it also open-loop (vision-blind) like our smooth800 ckpts?

This loads the raw 2toINF model via its own code (models.modeling_xvla.XVLA +
XVLAProcessor) and calls the EXACT server inference path (processor -> generate_actions,
domain_id=5 SoftFold, official instruction). We feed REAL frames from a KAI0 trace —
same robot (dual-arm Agilex Piper) + same task (flatten & fold cloth) as official
SoftFold-Agilex, so the obs is near-in-distribution. ee6d proprio (state20, 20-dim,
identical layout) comes straight from the trace npz.

For sampled frames i (seed fixed identically across the 3 calls so the flow-matching
noise prior cancels and only the input differs):
    d_img   = ||A(img_i, state_i) - A(img_j, state_i)||   swap IMAGE, hold state
    d_blank = ||A(img_i, state_i) - A(black, state_i)||    blank IMAGE, hold state
    d_state = ||A(img_i, state_i) - A(img_i, state_k)||    swap STATE, hold image
Reported as EE xyz L2 in mm (over the 30-step chunk, both arms) + gripper.
vision/proprio ratio = d_img / d_state. ~0 -> vision-blind; ~1 -> healthy closed-loop.

Run (X-VLA venv):
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_xvla/bin/python \
    train_scripts/kai/eval/eval_xvla_OFFICIAL_vision_ablation.py \
    --ckpt xvla/ckpts_official/X-VLA-SoftFold \
    --trace /tmp/xvla_stack/trace_20260601_192213 --n 12
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch

UPSTREAM = "/data1/tim/workspace/X-VLA/upstream"

# ee6d 20-dim layout: left xyz[0:3] rot6d[3:9] grip[9], right xyz[10:13] rot6d[13:19] grip[19]
XYZ_IDX = [0, 1, 2, 10, 11, 12]
GRIP_IDX = [9, 19]
# view order the official server expects: image0=cam_high, image1=cam_left_wrist, image2=cam_right_wrist
VIEW_SUFFIX = ["top_head", "hand_left", "hand_right"]


def load_frame(trace, stem):
    from PIL import Image
    imgs = []
    for suf in VIEW_SUFFIX:
        p = f"{trace}/server_images/{stem}_{suf}.jpg"
        imgs.append(np.array(Image.open(p).convert("RGB")))
    d = np.load(f"{trace}/server_arrays/{stem}.npz")
    return imgs, d["state20"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--domain_id", type=int, default=5, help="SoftFold-Agilex domain")
    ap.add_argument("--instruction", default="flatten the cloth and then fold it, then place it to the right side of you")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sys.path.insert(0, UPSTREAM)
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    device = "cuda"
    print(f"[load] {args.ckpt}")
    proc = XVLAProcessor.from_pretrained(args.ckpt)
    model = (XVLA.from_pretrained(args.ckpt, trust_remote_code=True, torch_dtype=torch.float32)
             .to(device).to(torch.float32).eval())
    print(f"[load] done. use_proprio={model.use_proprio} action_mode={model.action_mode} num_actions={model.num_actions}")

    stems = sorted({os.path.basename(p).split("_top_head")[0]
                    for p in glob.glob(f"{args.trace}/server_images/*_top_head.jpg")})
    # keep only stems that have all 3 views + npz
    stems = [s for s in stems
             if all(os.path.exists(f"{args.trace}/server_images/{s}_{suf}.jpg") for suf in VIEW_SUFFIX)
             and os.path.exists(f"{args.trace}/server_arrays/{s}.npz")]
    if len(stems) < 4:
        print(f"[err] only {len(stems)} usable frames"); return
    idx = np.linspace(0, len(stems) - 1, args.n).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    print(f"[data] {len(stems)} frames, sampling {len(idx)}: {idx}")

    frames = {i: load_frame(args.trace, stems[i]) for i in idx}

    @torch.no_grad()
    def infer(images, proprio, seed):
        torch.manual_seed(seed)
        inputs = proc(images, args.instruction)
        input_ids = inputs["input_ids"].to(device)
        image_input = inputs["image_input"].to(device).float()
        image_mask = inputs["image_mask"].to(device)
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0)
        domain_t = torch.tensor([args.domain_id], dtype=torch.long, device=device)
        act = model.generate_actions(input_ids=input_ids, image_input=image_input,
                                     image_mask=image_mask, domain_id=domain_t,
                                     proprio=proprio_t, steps=args.steps)
        return act.squeeze(0).float().cpu().numpy()  # [num_actions, 20]

    def dxyz_mm(a, b):
        return float(np.linalg.norm((a[:, XYZ_IDX] - b[:, XYZ_IDX]), axis=-1).mean() * 1000.0)

    def dgrip(a, b):
        return float(np.abs(a[:, GRIP_IDX] - b[:, GRIP_IDX]).mean())

    rows = []
    for n, i in enumerate(idx):
        j = idx[(n + len(idx) // 2) % len(idx)]   # far image donor
        k = idx[(n + len(idx) // 3) % len(idx)]   # far state donor
        imgs_i, st_i = frames[i]
        imgs_j, _ = frames[j]
        _, st_k = frames[k]
        seed = args.seed + n
        A = infer(imgs_i, st_i, seed)
        A_swapimg = infer(imgs_j, st_i, seed)
        A_blank = infer([np.zeros_like(x) for x in imgs_i], st_i, seed)
        A_swapst = infer(imgs_i, st_k, seed)
        r = dict(frame=stems[i],
                 d_img=dxyz_mm(A, A_swapimg), d_blank=dxyz_mm(A, A_blank), d_state=dxyz_mm(A, A_swapst),
                 g_img=dgrip(A, A_swapimg), g_state=dgrip(A, A_swapst))
        rows.append(r)
        print(f"  {r['frame']:>8}  d_img={r['d_img']:7.2f}mm  d_blank={r['d_blank']:7.2f}mm  "
              f"d_state={r['d_state']:7.2f}mm  | g_img={r['g_img']:.4f} g_state={r['g_state']:.4f}")

    d_img = np.mean([r["d_img"] for r in rows])
    d_blank = np.mean([r["d_blank"] for r in rows])
    d_state = np.mean([r["d_state"] for r in rows])
    ratio = d_img / d_state if d_state > 1e-9 else float("nan")
    blank_ratio = d_blank / d_state if d_state > 1e-9 else float("nan")
    print("\n========== OFFICIAL X-VLA-SoftFold vision-ablation ==========")
    print(f"  swap-IMAGE  d_img   = {d_img:7.2f} mm")
    print(f"  blank-IMAGE d_blank = {d_blank:7.2f} mm")
    print(f"  swap-STATE  d_state = {d_state:7.2f} mm")
    print(f"  vision/proprio ratio (d_img/d_state)   = {ratio:.3f}")
    print(f"  blank /proprio ratio (d_blank/d_state) = {blank_ratio:.3f}")
    print(f"  verdict: {'VISION-BLIND (open-loop)' if ratio < 0.1 else 'reads vision' if ratio > 0.4 else 'weak/ambiguous'}")
    import json
    out = f"/tmp/xvla_official_vision_ablation.json"
    json.dump(dict(ckpt=args.ckpt, trace=args.trace, domain_id=args.domain_id,
                   instruction=args.instruction, n=len(idx), rows=rows,
                   d_img=d_img, d_blank=d_blank, d_state=d_state, ratio=ratio), open(out, "w"), indent=2)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
