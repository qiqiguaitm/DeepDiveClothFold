"""tau0 joint-space fine-tune trainer (flow-matching), P1/P2 freeze schedule.

Adapts GigaWorld's flow-matching forward_step convention to tau0's WanModel.forward:
  sigma ~ U(0,1); sigma = flow_shift*sigma/(1+(flow_shift-1)*sigma); ts = round(sigma*1000)
  noisy = noise*sigma + clean*(1-sigma);  velocity target = noise - clean
  loss = lambda_v * MSE(video) + lambda_a * MSE(action)

The video backbone runs (return_video=True) so the action branch can cross-attend to
its features; set lambda_v small/0 for action-focused FT. Single-sample forward (tau0's
WanModel.forward, like its inference, operates on batch=1; outer loop / grad-accum batches).

`smoke_step()` runs forward+loss+backward on a random-init model with dummy inputs of the
correct shapes — verifies the trainable path + that grads reach the P1 joint projections.
No checkpoint / VAE / data needed for the smoke.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finetune.model_joint import build_joint_wanmodel, set_trainable  # noqa: E402

# tau0 deployment config (configs/deployment/wan_pretrain_rela_eef6d.yaml)
IMG_H, IMG_W = 192, 256
N_VIEWS = 3
# video frames -> T_lat=(CHUNK-1)//4+1. 5->2 (reused GigaWorld cache); 9->3 (tau0 native, P3).
CHUNK = int(os.environ.get("TAU0_CHUNK", "5"))
COND_NOISE = float(os.environ.get("TAU0_COND_NOISE", "0.0"))  # P4: noise-augment conditioning frame
ACTION_CHUNK = 33
ACTION_DIM = 14     # joint
VAE_SP, VAE_T = 16, 4
ZDIM = 48
PATCH = (1, 2, 2)
TEXT_LEN, TEXT_DIM = 512, 4096


def video_latent_shape():
    """latent tensor shape [C, T, h, W*V] that tau0 expects as x (views concat along width)."""
    t_lat = (CHUNK - 1) // VAE_T + 1
    h = IMG_H // VAE_SP
    w = IMG_W // VAE_SP
    return ZDIM, t_lat, h, w * N_VIEWS, h, w


def compute_seq_len():
    _, t_lat, h, Wv, _, _ = video_latent_shape()
    return t_lat * h * Wv // (PATCH[1] * PATCH[2])


class TauFlowTrainer:
    def __init__(self, model, device, flow_shift=5.0, lambda_v=1.0, lambda_a=5.0):
        self.model = model
        self.device = device
        self.flow_shift = flow_shift
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        self.seq_len = compute_seq_len()

    def _sigma(self, n=1):
        s = torch.rand(n, device=self.device)
        s = self.flow_shift * s / (1 + (self.flow_shift - 1) * s)
        ts = torch.round(s * 1000).long()
        s = ts.float() / 1000
        return ts, s

    def forward_step(self, z0, a0, state, context, ref=None):
        """z0:[C,T,h,Wv]  a0:[1,H,14]  state:[1,1,14]  context:[L,TEXT_DIM] (all on device).
        First frame (T=0) is the observed-frame conditioning: held clean, timestep 0, excluded
        from the video loss — mirrors tau0 infer()'s mask2/temp_ts."""
        dev = self.device
        ts, sig = self._sigma(1)
        sig_v = sig.view(1, 1, 1, 1)
        sig_a = sig.view(1, 1, 1)

        # frame-0 conditioning mask: 0 at T=0 (clean), 1 elsewhere (denoised)
        mask = torch.ones_like(z0)
        mask[:, 0] = 0
        if ref is not None:
            z0 = z0.clone()
            z0[:, 0:1] = ref                     # ensure frame 0 == observed conditioning

        nv = torch.randn_like(z0)
        na = torch.randn_like(a0)
        noised_v = (nv * sig_v + z0 * (1 - sig_v))
        zt = ((1 - mask) * z0 + mask * noised_v).to(z0.dtype)   # frame0 clean, rest noised
        if COND_NOISE > 0:   # robustify vs imperfect self-conditioning at rollout (exposure-bias mitigation)
            zt = zt.clone()
            zt[:, 0:1] = zt[:, 0:1] + COND_NOISE * torch.randn_like(zt[:, 0:1])
        at = (na * sig_a + a0 * (1 - sig_a)).to(a0.dtype)
        v_target = nv - z0
        a_target = na - a0

        # per-token video timestep: 0 on conditioned frame, ts elsewhere (patch-downsampled h,w)
        ts_val = float(ts.item())
        temp = (mask[0][:, ::PATCH[1], ::PATCH[2]] * ts_val).flatten()
        if temp.numel() < self.seq_len:
            temp = torch.cat([temp, temp.new_full((self.seq_len - temp.numel(),), ts_val)])
        v_ts = temp[: self.seq_len].unsqueeze(0).to(dev)
        a_ts = ts.view(1, 1).repeat(1, ACTION_CHUNK).float()
        self._vmask = mask

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(
                x=[zt],
                t=v_ts,
                context=[context],
                seq_len=self.seq_len,
                action_states=at,
                action_timestep=a_ts,
                return_video=True,
                return_action=True,
                store_buffer=False,
                history_action_state=state,
            )
        v_pred = out["video"][0]
        a_pred = out["action"]
        m = self._vmask
        v_loss = (((v_pred.float() - v_target.float()) * m) ** 2).sum() / m.sum().clamp(min=1)
        a_loss = ((a_pred.float() - a_target.float()) ** 2).mean()
        total = self.lambda_v * v_loss + self.lambda_a * a_loss
        return total, {"v_loss": v_loss.item(), "a_loss": a_loss.item()}


def smoke_step(phase="p1_warm", device="cuda", dtype=torch.bfloat16):
    print(f"[smoke] building random-init joint model (no weights) ...")
    model, _ = build_joint_wanmodel(action_in_dim=ACTION_DIM, load_pretrained=False,
                                    dtype=dtype, device=device, verbose=False)
    n_tr, n_all = set_trainable(model, phase)
    print(f"[smoke] phase={phase}  trainable={n_tr/1e6:.3f}M / {n_all/1e9:.3f}B")
    model.train()

    tr = TauFlowTrainer(model, torch.device(device))
    print(f"[smoke] seq_len={tr.seq_len}  video_latent={tuple(video_latent_shape()[:4])}")

    C, t_lat, h, Wv, _, _ = video_latent_shape()
    z0 = torch.randn(C, t_lat, h, Wv, device=device, dtype=dtype)
    a0 = torch.randn(1, ACTION_CHUNK, ACTION_DIM, device=device, dtype=dtype)
    state = torch.randn(1, 1, ACTION_DIM, device=device, dtype=dtype)
    context = torch.randn(TEXT_LEN, TEXT_DIM, device=device, dtype=dtype)

    loss, parts = tr.forward_step(z0, a0, state, context)
    loss.backward()

    # verify grads reach the (always-trainable) joint projections
    g = {n: (p.grad is not None and p.grad.abs().sum().item() > 0)
         for n, p in model.named_parameters()
         if n.startswith("action_proj_in") or n.startswith("action_head.head")}
    print(f"[smoke] loss={loss.item():.4f}  v_loss={parts['v_loss']:.4f}  a_loss={parts['a_loss']:.4f}")
    print(f"[smoke] grad reaches joint projections: {g}")
    ok = all(g.values())
    print(f"[smoke] {'PASS ✅ trainable path works' if ok else 'FAIL ❌ no grad to joint heads'}")
    return ok


def real_step(phase="p1_warm", load_pretrained=False, device="cuda", dtype=torch.bfloat16, steps=2):
    """Load a real batch from the reused latent cache; run forward+backward.
    load_pretrained=True requires the tau0 checkpoint to be fully downloaded."""
    from finetune.data_joint import LatentJointDataset
    fdir = os.path.dirname(os.path.abspath(__file__))
    base = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1"
    ds = LatentJointDataset(f"{base}/visrobot01_train", f"{fdir}/assets/statistics_visrobot01.json",
                            action_chunk=ACTION_CHUNK, embed_id=0)
    print(f"[real] dataset episodes={len(ds)}  (window-sampled)")
    model, rep = build_joint_wanmodel(action_in_dim=ACTION_DIM, load_pretrained=load_pretrained,
                                      dtype=dtype, device=device, verbose=True)
    n_tr, n_all = set_trainable(model, phase)
    print(f"[real] phase={phase}  trainable={n_tr/1e6:.3f}M / {n_all/1e9:.3f}B  pretrained={load_pretrained}")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    tr = TauFlowTrainer(model, torch.device(device))
    print(f"[real] seq_len={tr.seq_len}")
    for step in range(steps):
        b = ds[np.random.randint(len(ds))]
        z0 = b["video_latent"].to(device, dtype)
        ref = b["ref"].to(device, dtype)
        a0 = b["action"].unsqueeze(0).to(device, dtype)     # [1,H,14]
        state = b["state"].unsqueeze(0).to(device, dtype)   # [1,1,14]
        ctx = b["t5"].to(device, dtype)
        opt.zero_grad()
        loss, parts = tr.forward_step(z0, a0, state, ctx, ref=ref)
        loss.backward()
        opt.step()
        print(f"[real] step {step}: z0={tuple(z0.shape)} a0={tuple(a0.shape)} t5={tuple(ctx.shape)} "
              f"loss={loss.item():.4f} v={parts['v_loss']:.4f} a={parts['a_loss']:.4f}")
    print("[real] PASS ✅ real-data train step works")


if __name__ == "__main__":
    import numpy as np  # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--phase", default="p1_warm", choices=["p1_warm", "p2_specialize", "all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.smoke:
        smoke_step(phase=args.phase, device=args.device)
    if args.real:
        real_step(phase=args.phase, load_pretrained=args.pretrained, device=args.device)
