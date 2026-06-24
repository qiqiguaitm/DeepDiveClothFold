"""Static configuration: filesystem paths, encoder registry, dataset registry."""
from crave.config.datasets import DATASETS, DatasetConfig, resolve_dataset
from crave.config.encoders import ENCODERS, EncoderSpec, resolve
from crave.config.paths import CRAVE_DOCS, DOCS, HF_HUB, REPO, TEMP, out_dir, viz_dir

__all__ = [
    "REPO", "TEMP", "DOCS", "CRAVE_DOCS", "HF_HUB", "out_dir", "viz_dir",
    "ENCODERS", "EncoderSpec", "resolve",
    "DATASETS", "DatasetConfig", "resolve_dataset",
]
