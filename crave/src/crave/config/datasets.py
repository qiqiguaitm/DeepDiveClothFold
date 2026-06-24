"""Dataset registry — one row per dataset, mirroring the legacy CFG dict.

`kind` selects the loader in crave.data.loaders:
  - "lerobot2"  : LeRobot v2 (parquet state + per-episode mp4)         e.g. VIS
  - "hdf5"      : HDF5 tree (qpos + JPEG-encoded frames)               e.g. XVLA
  - "lerobotv3" : LeRobot v3 (concat mp4 + parquet, state from cache)  e.g. coffee
"""
from __future__ import annotations

from dataclasses import dataclass

from crave.config.paths import REPO


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    kind: str
    root: str
    cam: str
    stride: int = 10
    maxep: int = 9999
    statecache: str | None = None   # lerobotv3: external proprio cache when not in parquet
    arm_cache: str | None = None    # kai0-family: tcc_*_armmask/feat_cache (key "f")
    raw_cache: str | None = None    # kai0-family: tcc_*_raw/feat_cache (key "f")


DATASETS: dict[str, DatasetConfig] = {
    "vis": DatasetConfig(
        "vis", "lerobot2",
        str(REPO / "kai0/data/Task_A/self_built/pure_vis600/base"),
        "observation.images.top_head", stride=10, maxep=560),
    "xvla": DatasetConfig(
        "xvla", "hdf5",
        str(REPO / "xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow"),
        "cam_high", stride=10, maxep=168),
    "coffee": DatasetConfig(
        "coffee", "lerobotv3",
        "/vePFS/tim/workspce/hf_cache/hub_default/datasets--lerobot--aloha_static_coffee/snapshots/b144896feb1f37398a862927b22cd3abdf005a6b",
        "observation.images.cam_high", stride=16, maxep=50,
        statecache=str(REPO / "temp/generalization_value_eval/coffee/feat_cache")),
    # ---- kai0-family (chunked parquet + top_head mp4 + tcc 3-path cache) ----
    "kai0_base": DatasetConfig(
        "kai0_base", "kai0", str(REPO / "kai0/data/Task_A/kai0_base"),
        "observation.images.top_head", stride=10,
        arm_cache=str(REPO / "temp/tcc_kai0_armmask/feat_cache"),
        raw_cache=str(REPO / "temp/tcc_kai0_raw/feat_cache")),
    "smooth800_dagger": DatasetConfig(
        "smooth800_dagger", "kai0", str(REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"),
        "observation.images.top_head", stride=10,
        arm_cache=str(REPO / "temp/tcc_smooth800_dagger_armmask/feat_cache"),
        raw_cache=str(REPO / "temp/tcc_smooth800_dagger_raw/feat_cache")),
    "vis0526": DatasetConfig(
        "vis0526", "kai0", str(REPO / "kai0/data/Task_A/vis_base/v3/2026-05-26-v3"),
        "observation.images.top_head", stride=10,
        arm_cache=str(REPO / "temp/tcc_vis0526_armmask/feat_cache"),
        raw_cache=str(REPO / "temp/tcc_vis0526_raw/feat_cache")),
}


def resolve_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]
