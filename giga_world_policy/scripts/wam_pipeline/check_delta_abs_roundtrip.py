"""delta/abs action 表示的 round-trip 数值自检(无需 checkpoint/数据,只读 norm_stats)。

校验 train 侧(absolute → delta → normalize)与 serve 侧(denormalize → add_state → absolute)
在任意 mask 下严格互逆,以及 resolve_delta_mask 从 norm_stats 内嵌 delta_mask 正确取值。

不变量:  action_hat = add_state( denorm( normalize(action − state·m) ), state, m ) == action
m 全 False(abs)与 m=piper14(delta/mixed)都应 recover,误差 ~0(float)。

用法(repo root,先 source env.sh):
  python -m scripts.wam_pipeline.check_delta_abs_roundtrip \
      --stats ./assets_visrobot01/norm_stats_vis.json ./assets_visrobot01/norm_stats_kai.json
"""

import argparse

import numpy as np
import torch

from world_action_model.pipeline.utils import (
    DEFAULT_PIPER14_DELTA_MASK,
    add_state_to_action,
    denormalize_action,
    extract_normalization_tensors,
    load_stats,
    resolve_delta_mask,
)


def _norm_delta_train_side(action_abs, state, mask, norm, mode):
    """复刻 wa_transforms_lerobot 的 train 侧:absolute → delta(masked)→ normalize。"""
    idx = torch.nonzero(torch.as_tensor(mask, dtype=torch.bool), as_tuple=False).flatten()
    delta = action_abs.clone()
    if idx.numel() > 0:
        delta[:, idx] = action_abs[:, idx] - state[idx]
    eps = 1e-8
    if mode == "zscore":
        return (delta - norm.action_mean) / norm.action_std.clamp_min(eps)
    rng = norm.action_max - norm.action_min + eps
    return ((delta - norm.action_min) / rng) * 2 - 1


def roundtrip(stats_path, mask, action_dim=14, chunk=8, mode="zscore", seed=0):
    stats = load_stats(stats_path)
    norm = extract_normalization_tensors(stats, torch.device("cpu"), state_dim=action_dim, action_dim=action_dim)
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(action_dim, generator=g, dtype=torch.float32)
    action_abs = torch.randn(chunk, action_dim, generator=g, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    nd = _norm_delta_train_side(action_abs, state, mask_t, norm, mode)
    de = denormalize_action(nd, norm, mode)
    recon = add_state_to_action(de, state, chunk, mask_t)
    return float((recon - action_abs).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", nargs="+", required=True)
    ap.add_argument("--action_dim", type=int, default=14)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    piper14 = list(DEFAULT_PIPER14_DELTA_MASK)
    abs_mask = [False] * args.action_dim
    ok = True

    # resolve_delta_mask 回退测试:无 delta_mask 字段的旧 stats → piper14。
    legacy = resolve_delta_mask({"norm_stats": {}}, args.action_dim)
    assert legacy.tolist() == piper14, f"fallback mask wrong: {legacy.tolist()}"
    print(f"[fallback] legacy stats (no delta_mask) -> piper14  OK")

    for sp in args.stats:
        stats = load_stats(sp)
        embedded = resolve_delta_mask(stats, args.action_dim)
        repr_ = stats.get("action_repr", "(none)")
        has = "delta_mask" in stats
        for mode in ("zscore", "minmax"):
            for label, m in (("embedded", embedded.tolist()), ("delta/piper14", piper14), ("abs/all-False", abs_mask)):
                err = roundtrip(sp, m, args.action_dim, mode=mode)
                status = "OK" if err < args.tol else "FAIL"
                ok = ok and err < args.tol
                print(f"[{sp.split('/')[-1]:28s}] embed_field={has} repr={repr_:6s} "
                      f"mode={mode:6s} mask={label:13s} max_err={err:.2e} {status}")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
