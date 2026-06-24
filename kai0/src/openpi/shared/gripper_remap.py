"""Deploy-time gripper frame remap (dependency-light: numpy/os/logging only).

Kept separate from ``normalize.py`` on purpose: ``normalize.py`` imports
``numpydantic`` (for the ``NormStats`` pydantic dataclass), which is NOT
installed in the V1 TensorRT serve venv (``.venv_5090_trt``). ``serve_policy_v1.py``
needs the remap but must not drag in numpydantic, so the remap lives here.

A ckpt trained when the gripper was configured max_range=100 (old, off-spec)
encodes the gripper action/state dims in that old physical frame (full open
~= 0.08-0.10m). After re-zeroing the real grippers to the official 0-70mm frame
(see docs/deployment/data_collection/gripper_calibration.md), deploying such a
ckpt would over/under-command the gripper.

Fix: AFFINE-remap each gripper dim from the ckpt's OWN training range
[q01, q99] (the actual span the gripper covered in training, NOT a fixed ratio)
onto the real robot range [lo, hi] (default 0-0.07m). The same remap is applied
to both the `state` (proprio in) and `actions` (command out) norm_stats, so
proprio normalization and action unnormalization stay consistent. Degenerate
dims (q99~=q01, e.g. an unused 2nd gripper on a single-arm task) are left
untouched. Gated by env so it only affects old ckpts when explicitly enabled.

  KAI0_GRIPPER_DEPLOY_REMAP=1            enable (default off -> no-op)
  KAI0_GRIPPER_REAL_RANGE="0.0,0.07"    real [closed,open] in meters (action units)
  KAI0_GRIPPER_DIMS="6,13"              gripper dims (left,right) in the 14/32-dim vector
"""

import logging
import os

import numpy as np


def gripper_deploy_remap_cfg():
    """Return (dims, lo, hi) from env, or None if disabled/malformed."""
    if os.environ.get("KAI0_GRIPPER_DEPLOY_REMAP", "0") not in ("1", "true", "True", "yes"):
        return None
    rng = os.environ.get("KAI0_GRIPPER_REAL_RANGE", "0.0,0.07")
    try:
        lo, hi = (float(x) for x in rng.split(","))
        dims = [int(x) for x in os.environ.get("KAI0_GRIPPER_DIMS", "6,13").split(",")]
    except Exception as e:
        logging.warning(f"[gripper-remap] bad env (KAI0_GRIPPER_REAL_RANGE/DIMS): {e}; disabled")
        return None
    if hi <= lo:
        logging.warning(f"[gripper-remap] real range hi<=lo ({lo},{hi}); disabled")
        return None
    return dims, lo, hi


def _remap_gripper_arrays(mean, std, q01, q99, dims, lo, hi, *, tag=""):
    """Affine-remap gripper `dims` from training range [q01,q99] (or mean+-2std)
    onto [lo,hi]. Returns new float arrays (mean, std, q01, q99). Degenerate
    dims (range < 1e-6) are skipped. None preserved for q01/q99.
    """
    mean = np.array(mean, dtype=np.float64)
    std = np.array(std, dtype=np.float64)
    q01 = None if q01 is None else np.array(q01, dtype=np.float64)
    q99 = None if q99 is None else np.array(q99, dtype=np.float64)
    for d in dims:
        if d >= mean.shape[-1]:
            continue
        if q01 is not None and q99 is not None:
            lo_t, hi_t = float(q01[d]), float(q99[d])
        else:
            lo_t, hi_t = float(mean[d]) - 2.0 * float(std[d]), float(mean[d]) + 2.0 * float(std[d])
        if hi_t - lo_t < 1e-6:
            logging.info(f"[gripper-remap]{tag} dim {d} degenerate (range~=0), skipped")
            continue
        a = (hi - lo) / (hi_t - lo_t)
        b = lo - a * lo_t
        mean[d] = a * mean[d] + b
        std[d] = a * std[d]
        if q01 is not None:
            q01[d] = a * q01[d] + b
        if q99 is not None:
            q99[d] = a * q99[d] + b
        logging.info(f"[gripper-remap]{tag} dim {d}: train[{lo_t:.4f},{hi_t:.4f}] -> real[{lo:.4f},{hi:.4f}] (a={a:.3f})")
    return mean, std, q01, q99


def remap_gripper_norm_stats(norm_stats):
    """Deploy-time gripper remap for a dict[str, NormStats] (used by
    create_trained_policy). No-op (returns input) when disabled. Rebuilds each
    NormStats (via its own type) with remapped gripper dims; non-gripper dims
    unchanged. Duck-typed on .mean/.std/.q01/.q99 so it needs no numpydantic.
    """
    cfg = gripper_deploy_remap_cfg()
    if cfg is None or not norm_stats:
        return norm_stats
    dims, lo, hi = cfg
    out = {}
    for key, s in norm_stats.items():
        m, sd, q1, q9 = _remap_gripper_arrays(s.mean, s.std, s.q01, s.q99, dims, lo, hi, tag=f" {key}")
        out[key] = type(s)(mean=m, std=sd, q01=q1, q99=q9)
    logging.info(f"[gripper-remap] applied to {list(out)} dims={dims} real=[{lo},{hi}]m")
    return out


def remap_gripper_raw(norm):
    """Deploy-time gripper remap for a raw nested dict
    {"state": {mean,std,q01,q99}, "actions": {...}} of np arrays (used by V1
    serve_policy_v1.py). Mutates in place and returns it. No-op when disabled.
    """
    cfg = gripper_deploy_remap_cfg()
    if cfg is None or not norm:
        return norm
    dims, lo, hi = cfg
    for key in ("state", "actions"):
        if key not in norm:
            continue
        s = norm[key]
        m, sd, q1, q9 = _remap_gripper_arrays(s["mean"], s["std"], s.get("q01"), s.get("q99"), dims, lo, hi, tag=f" {key}")
        s["mean"], s["std"] = m, sd
        if q1 is not None:
            s["q01"] = q1
        if q9 is not None:
            s["q99"] = q9
    logging.info(f"[gripper-remap] applied to raw norm dims={dims} real=[{lo},{hi}]m")
    return norm
