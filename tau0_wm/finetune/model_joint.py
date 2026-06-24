"""Build tau0 WanModel in joint-14 space, reusing the pretrained trunk.

`build_joint_wanmodel()` constructs WanModel with action_in_dim=14, loads the
released eef6d-20 checkpoint with strict=False, drops the two action-space-bound
projections (shape 20->14), and freshly initializes them. Everything else
(action_blocks x30, video backbone, embeddings, time/heads) loads from
pretrained. Verified by finetune/p0_probe_load.py.
"""
import json
import os

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diffusion config from configs/deployment/wan_pretrain_rela_eef6d.yaml,
# action_in_dim overridden to the joint dim.
BASE_CFG = dict(
    model_type="ti2v",
    patch_size=[1, 2, 2],
    text_len=512,
    in_dim=48,
    dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    out_dim=48,
    num_heads=24,
    num_layers=30,
    window_size=[-1, 1],
    qk_norm=True,
    cross_attn_norm=True,
    eps=1.0e-06,
    use_ae=True,
    action_dim=1024,
    action_num_heads=16,
    action_ffn_dim=2048,
    action_max_seq_len=60,
)

# parameter-name prefixes whose shape depends on action_in_dim (the only ones
# that must be reinitialized when switching eef6d-20 -> joint-14).
_ACTION_SPACE_BOUND = ("action_proj_in.", "action_head.head.")


def _default_ckpt_dir():
    return os.path.join(_ROOT, "checkpoints", "tau-0-wm")


def build_joint_wanmodel(action_in_dim=14, ckpt_dir=None, load_pretrained=True,
                         dtype=torch.float32, device="cpu", verbose=True):
    """Return (model, report). report = {loaded, reinit:[...], missing:[...], unexpected:[...]}."""
    from models.wan_2_2_models.transformers.model import WanModel  # noqa: E402

    cfg = dict(BASE_CFG, action_in_dim=action_in_dim)
    model = WanModel(**cfg)

    report = {"loaded": 0, "reinit": [], "missing": [], "unexpected": []}
    if load_pretrained:
        ckpt_dir = ckpt_dir or _default_ckpt_dir()
        index = os.path.join(ckpt_dir, "diffusion_pytorch_model.bin.index.json")
        if not os.path.exists(index):
            raise FileNotFoundError(f"checkpoint index not found: {index}")
        from utils.model_utils import load_index_file
        sd = load_index_file(index, mode="bin")
        msd = model.state_dict()
        # drop action-space-bound (shape-mismatched) tensors -> they stay freshly init
        for k in list(sd.keys()):
            if k in msd and sd[k].shape != msd[k].shape:
                report["reinit"].append((k, tuple(sd[k].shape), tuple(msd[k].shape)))
                del sd[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        report["loaded"] = len(sd)
        report["missing"] = list(missing)
        report["unexpected"] = list(unexpected)
        # explicit fresh init of the reinitialized joint projections
        _init_action_projections(model)
        if verbose:
            print(f"[build_joint_wanmodel] loaded {report['loaded']} tensors from pretrained; "
                  f"reinit {len(report['reinit'])} action-space-bound; "
                  f"missing {len(missing)}; unexpected {len(unexpected)}")
            for k, a, b in report["reinit"]:
                print(f"    reinit {k:36s} ckpt{a} -> model{b}")

    is_meta = any(p.is_meta for p in model.parameters())
    if not is_meta:
        model = model.to(device=device, dtype=dtype)
    return model, report


def _init_action_projections(model):
    """Xavier/zero init for the freshly-sized joint projections."""
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in _ACTION_SPACE_BOUND):
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)


def set_trainable(model, mode):
    """Freeze schedule for the two-phase recipe.

    mode='p1_warm'  -> train ONLY the joint projections (action_proj_in/action_head); freeze trunk.
    mode='p2_specialize' -> also train action_blocks (+ time emb); video backbone stays frozen.
    mode='all' -> everything trainable.
    Returns (#trainable_params, #total_params).
    """
    trainable_prefixes = {
        "p1_warm": ("action_proj_in.", "action_head."),
        "p2_specialize": ("action_proj_in.", "action_head.", "action_blocks.",
                          "action_time_embedding.", "action_time_projection."),
    }
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        pre = trainable_prefixes[mode]
        for name, p in model.named_parameters():
            if any(name.startswith(x) for x in pre):
                p.requires_grad_(True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    return n_tr, n_all
