# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``wam_fold_wm_nano_t1c`` — L1 + L2 full (⑥ history + ⑦ joint-IDM loss + ⑧ CFG↑).

= t1b (L1 + 2-frame history) plus the BIAS levers (plan §L2):
  * ⑦ ``mode="joint"`` — per-sample mix of forward_dynamics / inverse_dynamics /
    policy. The IDM & policy modes SUPERVISE the action (action_loss_weight=10 in the
    NANO rectified-flow config), forcing the action tokens to carry information →
    the strongest pressure to actually USE the action (vs pure-FD where action is
    clean, unsupervised conditioning and can be ignored).
  * ⑧ ``cfg_dropout_rate`` 0.1→0.2 — sharper action conditional-vs-unconditional
    contrast for stronger classifier-free guidance at inference. (A multiplier on the
    dependence ⑥/⑦ create, not a creator — only meaningful stacked on them.)

Both are benign w.r.t. the pretrained model: joint mode is a native training mode and
cfg dropout is a standard data knob the base was already trained with.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
import cosmos_framework.configs.base.experiment.action.posttrain_config.wam_fold_wm_nano as _wm
import cosmos_framework.configs.base.experiment.action.posttrain_config.wam_fold_wm_nano_t1b as _t1b

cs = ConfigStore.instance()

wam_fold_wm_nano_t1c = copy.deepcopy(_t1b.wam_fold_wm_nano_t1b)

# --- L2-⑦: joint FD+IDM(+policy) mode → supervises action → forces action use ---
wam_fold_wm_nano_t1c["dataloader_train"]["data_source"] = L(_wm.build_wm_data_source)(
    chunk_length=16,
    fps=30.0,
    mode="joint",
    frame_stride=(2, 4),
)
# --- L2-⑧: raise action CFG dropout (0.1 -> 0.2) ---
wam_fold_wm_nano_t1c["dataloader_train"]["data_packer"]["cfg_dropout_rate"] = 0.2
wam_fold_wm_nano_t1c["job"]["name"] = "wam_fold_wm_nano_t1c"


for _item in [wam_fold_wm_nano_t1c]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
