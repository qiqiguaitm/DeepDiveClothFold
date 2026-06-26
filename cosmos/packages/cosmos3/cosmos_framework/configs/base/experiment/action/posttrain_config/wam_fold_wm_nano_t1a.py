# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``wam_fold_wm_nano_t1a`` — L1 "catch Ctrl-World" ablation of ``wam_fold_wm_nano``.

Identical to the FD world-model baseline except for the L1 SIGNAL-boost levers
(plan: docs/training/future_plans/plans/cosmos3_acwm_catch_ctrlworld_plan.md):

  * ① abstract temporal resolution: ``frame_stride=(2,4)`` → sample every 2-4th
    source frame → ~7.5-15hz (centered ~10hz, à la Ctrl-World's 5-10hz) so each
    predicted frame spans more real time → larger inter-frame motion → stronger
    action→pixel signal. ② the (2,4) range randomizes stride per-sample = speed
    augmentation. ③ shorter window: ``chunk_length=16`` (17 frames = 4*4+1 → 5
    latent frames, predict ~4 future steps) — Ctrl-World's short-prediction regime.
  * ④ low-σ weighting: flow ``shift`` 480: 5→2 — rebalance training off the
    high-noise (coarse-structure, action-irrelevant) regime toward the low-noise
    (fine-detail, action-relevant) regime.

Everything else (model, optimizer, FD mode, fresh-init action heads, token budget)
mirrors ``wam_fold_wm_nano`` so this is a clean single-group A/B vs the M1 control.
⑤ (static-window filtering) and L2 (history / joint-IDM loss / CFG) land in the
t1b/t1c configs.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
import cosmos_framework.configs.base.experiment.action.posttrain_config.wam_fold_wm_nano as _wm

cs = ConfigStore.instance()

wam_fold_wm_nano_t1a = copy.deepcopy(_wm.wam_fold_wm_nano)

# --- L1 ①②③: ~10hz strided + random-skip + shorter window ---
wam_fold_wm_nano_t1a["dataloader_train"]["data_source"] = L(_wm.build_wm_data_source)(
    chunk_length=16,
    fps=30.0,
    mode="forward_dynamics",
    frame_stride=(2, 4),
)

# --- L1 ④: low-σ weighting (shift 480: 5 -> 2) ---
# model.config is the NANO_MODEL_CONFIG dict; rectified_flow_training_config.shift
# is a plain {res: shift} dict. 480 is our training resolution (ActionDataPacker).
_shift = wam_fold_wm_nano_t1a["model"]["config"]["rectified_flow_training_config"]["shift"]
_shift["480"] = 2

wam_fold_wm_nano_t1a["job"]["name"] = "wam_fold_wm_nano_t1a"


for _item in [wam_fold_wm_nano_t1a]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
