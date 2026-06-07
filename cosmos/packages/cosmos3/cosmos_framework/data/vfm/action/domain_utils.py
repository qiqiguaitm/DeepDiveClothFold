# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "fractal": 20,
    "wam_fold": 16,  # dual-arm Agilex Piper, 14-D joint-space (T-shirt fold) — visrobot01 rig
    "kairobot01": 17,  # dual-arm Agilex Piper, 14-D joint-space (T-shirt fold) — kairobot01 rig (distinct embodiment domain)
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "hand_pose": 57,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "agibotworld": 29,
    "fractal": 10,
    "wam_fold": 14,
    "kairobot01": 14,  # same dual-arm 14-D action layout as visrobot01 (wam_fold)
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]
