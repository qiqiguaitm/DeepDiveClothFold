# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``wam_fold_wm_nano_t1b`` — L1 + L2-⑥ (multi-frame history conditioning).

= t1a (L1 signal levers) + ``num_history_vision=2``: the first 2 LATENT frames are
clean history conditioning instead of 1 (plan §L2-⑥, "catch Ctrl-World"). With
chunk_length=16 → 17 pixel frames → 5 latent frames, so K=2 history + 3 predicted.

Weight-compatible / native (the model reads a general condition-index list and was
pretrained with varying clean-frame counts; inverse_dynamics conditions on all frames).
⚠ EVAL: ``omni_mot_model`` condition-image extraction hardcodes frame ``0:1`` — widen
to ``0:K`` before evaluating K>1 checkpoints (training is unaffected).
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

import cosmos_framework.configs.base.experiment.action.posttrain_config.wam_fold_wm_nano_t1a as _t1a

cs = ConfigStore.instance()

wam_fold_wm_nano_t1b = copy.deepcopy(_t1a.wam_fold_wm_nano_t1a)

# --- L2-⑥: 2 clean history latent frames ---
wam_fold_wm_nano_t1b["dataloader_train"]["data_packer"]["num_history_vision"] = 2
wam_fold_wm_nano_t1b["job"]["name"] = "wam_fold_wm_nano_t1b"


for _item in [wam_fold_wm_nano_t1b]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
