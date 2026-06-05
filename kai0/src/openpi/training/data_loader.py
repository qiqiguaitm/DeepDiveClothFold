from collections.abc import Iterator, Sequence
import logging
import math
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms
from openpi.training.advantage_dataset import AdvantageLerobotDataset

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        import random, logging
        # Higher retry budget: the Task_A/kai0_advantage dataset has ~311/3055 episodes (≈10%)
        # with missing videos, so occasional runs of bad samples are expected.
        MAX_ATTEMPTS = 50
        for attempt in range(MAX_ATTEMPTS):
            try:
                idx = index if attempt == 0 else random.randint(0, len(self._dataset) - 1)
                return self._transform(self._dataset[idx])
            except Exception as e:
                if attempt < MAX_ATTEMPTS - 1:
                    logging.warning(f'DataLoader: skipping index {idx} due to {type(e).__name__}: {e}')
                    continue
                raise

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    episodes: list[int] | None = None,
) -> Dataset:
    """Create a dataset for training.

    Args:
        episodes: Optional list of meta.episodes keys (positions) to include.
            If None (default), uses all episodes. Used for train/val split.

    Multi-dataset:
        若 `data_config.repo_ids` 非空, 按列表逐个构造 LeRobotDataset 后用
        torch.utils.data.ConcatDataset 串起来。`episodes` 对多数据集不直接适用
        (每个数据集有各自的索引空间), 传了会 raise。
    """
    repo_ids = getattr(data_config, "repo_ids", None)
    if repo_ids:
        if episodes is not None:
            raise ValueError(
                "`episodes` filter is not supported with multi-dataset (repo_ids); "
                "split per-dataset or merge offline first."
            )
        return _create_concat_torch_dataset(repo_ids, data_config, action_horizon, model_config)

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Absolute local path repo_id: newer huggingface_hub (≥0.30) strictly validates repo_id
    # format and rejects "/abs/path". Pass it via `root=` with a dummy valid label instead so
    # it loads from local disk regardless of hub version. (older hub on cnsh 3.11 tolerated abs
    # repo_id; cnbj 3.12 venv hub 0.32 does not.)
    _root = None
    _repo_label = repo_id
    if os.path.isabs(repo_id):
        _root = repo_id
        _repo_label = "local/" + os.path.basename(repo_id.rstrip("/"))

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(_repo_label, root=_root)
    dataset = lerobot_dataset.LeRobotDataset(
        _repo_label,
        root=_root,
        episodes=episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend="pyav",
        tolerance_s=30.0,  # default 1e-4 is too strict; some kai/vis episodes have gaps up to ~7s
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _create_concat_torch_dataset(
    repo_ids,
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Build one LeRobotDataset per repo_id, apply per-dataset prompt-from-task if requested,
    then ConcatDataset them. Each dataset uses its own fps for delta_timestamps (robust to
    heterogeneous FPS, though in practice should all be 30)."""
    from torch.utils.data import ConcatDataset
    dataset_ids = getattr(data_config, "dataset_ids", None)
    if dataset_ids is not None and len(dataset_ids) != len(repo_ids):
        raise ValueError(
            f"dataset_ids length ({len(dataset_ids)}) != repo_ids length ({len(repo_ids)})"
        )
    parts: list[Dataset] = []
    for i, rid in enumerate(repo_ids):
        meta = lerobot_dataset.LeRobotDatasetMetadata(rid)
        ds = lerobot_dataset.LeRobotDataset(
            rid,
            delta_timestamps={
                key: [t / meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
            tolerance_s=30.0,  # match single-repo path; raw kai0 fps timestamps are jittery
        )
        transforms_to_apply: list = []
        if data_config.prompt_from_task:
            transforms_to_apply.append(_transforms.PromptFromLeRobotTask(meta.tasks))
        if dataset_ids is not None:
            transforms_to_apply.append(_transforms.InjectDatasetId(int(dataset_ids[i])))
        if transforms_to_apply:
            ds = TransformedDataset(ds, transforms_to_apply)
        parts.append(ds)
    if not parts:
        raise ValueError("repo_ids is empty after parsing")
    logging.info(
        "multi-dataset: concat %d LeRobotDatasets: %s", len(parts),
        ", ".join(str(r) for r in repo_ids),
    )
    return ConcatDataset(parts)

def create_advantage_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    config: _config.TrainConfig = None,
) -> Dataset:
    """Create LeRobot dataset for advantage estimator (progress/value) training."""
    split = getattr(config, "split", "all") or "all"

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    
   
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = AdvantageLerobotDataset(
        repo_id=repo_id,
        episodes=episodes_split_through_task(repo_id, split_type=split, shuffle=False),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend='pyav',
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def episodes_split_through_task(repo_id: str, split_ratio: float = 0.9, split_type: str = "train", shuffle: bool = False, random_seed: int = 42) -> list[int]:
    """Split the episodes by task."""
    assert split_type in ['all', "train", "val"], f"Invalid split type '{split_type}'. Must be 'all', 'train' or 'val'."
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    episodes_meta = dataset_meta.episodes
    episodes_index = [key for key in episodes_meta.keys()]
    # tasks = [episodes_meta[idx]["tasks"] for idx in episodes_index]
    if split_type == "all":
        return episodes_index
    else:
        episodes_index_dict = {}
        for idx in episodes_index:
            task = episodes_meta[idx]["tasks"]
            task = ''.join(task)
            if task not in episodes_index_dict.keys():
                episodes_index_dict[task] = []
            episodes_index_dict[task].append(idx)
        train_episodes = []
        val_episodes = []
        for task in episodes_index_dict.keys():
            length = len(episodes_index_dict[task])
            train_length = int(length * split_ratio)
            train_episodes.extend(episodes_index_dict[task][:train_length])
            val_episodes.extend(episodes_index_dict[task][train_length:])
        if split_type == "train":
            return train_episodes
        else:
            return val_episodes


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    norm_tf = (
        _transforms.DomainNormalize(getattr(data_config, "domain_norm_stats", None) or {}, use_quantiles=data_config.use_quantile_norm)
        if getattr(data_config, "domain_norm_stats", None) and not skip_norm_stats
        else _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    )
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            norm_tf,
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    norm_tf = (
        _transforms.DomainNormalize(getattr(data_config, "domain_norm_stats", None) or {}, use_quantiles=data_config.use_quantile_norm)
        if getattr(data_config, "domain_norm_stats", None) and not skip_norm_stats
        else _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    )
    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            norm_tf,
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    episodes: list[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        config=config,
        episodes=episodes,
    )


def _build_domain_weights(dataset, domain_sample_weights: dict) -> "np.ndarray":
    """Per-frame sampling weight = domain_sample_weights[task_index[frame]]. Unwraps TransformedDataset
    layers to reach the LeRobotDataset and reads its per-frame `task_index` (= domain, set at merge time).
    Lets a single pre-merged multi-domain dataset be balanced by probability (e.g. kai:vis -> 1:1) with
    NO disk copy and WITHOUT the broken datasets_yaml/ConcatDataset path."""
    d = dataset
    while not hasattr(d, "hf_dataset") and hasattr(d, "_dataset"):
        d = d._dataset
    if not hasattr(d, "hf_dataset"):
        raise ValueError("domain_sample_weights set but could not reach LeRobotDataset.hf_dataset to read task_index")
    ti = np.asarray(d.hf_dataset["task_index"]).reshape(-1)
    weights = np.ones(len(ti), dtype=np.float64)
    for dom, wt in domain_sample_weights.items():
        weights[ti == int(dom)] = float(wt)
    return weights


class _DomainWeightedJAXSampler(torch.utils.data.Sampler):
    """Weighted-with-replacement frame sampler that also shards across JAX processes (per-rank seed).
    Realizes domain_sample_weights as draw probabilities; each process draws an independent weighted
    stream so multi-host sees different samples. num_samples per replica ≈ frames/num_replicas."""

    def __init__(self, weights, num_replicas: int, rank: int, seed: int = 0):
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.double)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.num_samples = math.ceil(len(self.weights) / self.num_replicas)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank)
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    config: _config.TrainConfig = None,
    episodes: list[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        config: Full train config; used when advantage_estimator=True to create advantage dataset.
    """
    if config is not None and getattr(config, "advantage_estimator", False):
        dataset = create_advantage_torch_dataset(data_config, action_horizon, model_config, config)
    else:
        dataset = create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
        if getattr(data_config, "domain_sample_weights", None):
            weights = _build_domain_weights(dataset, data_config.domain_sample_weights)
            sampler = _DomainWeightedJAXSampler(
                weights, num_replicas=jax.process_count(), rank=jax.process_index(), seed=seed
            )
            logging.info(
                f"[domain-weighted sampler] weights={data_config.domain_sample_weights} "
                f"frames={len(weights)} per-replica-samples={sampler.num_samples}"
            )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class _JAXProcessSampler(torch.utils.data.Sampler):
    """Sampler that selects different indices for each JAX process (multi-node data sharding)."""

    def __init__(self, dataset, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0):
        import math
        self._len = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = math.ceil(self._len / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(self._len, generator=g).tolist()
        else:
            indices = list(range(self._len))
        # Pad to total_size so every replica gets the same count
        indices += indices[: self.total_size - len(indices)]
        # Take every num_replicas-th element starting at rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.local_devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        # Multi-process JAX: use process-rank-aware sampler so each process loads different samples
        if jax.process_count() > 1 and sampler is None:
            sampler = _JAXProcessSampler(
                dataset,
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=shuffle,
                seed=seed,
            )
            shuffle = False  # Sampler handles shuffling

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            # Larger prefetch (default is 2): absorbs periodic MP4-decode/vePFS-IO stalls so GPU doesn't idle.
            # Value chosen based on resource check (1.3 TB /dev/shm, 1.7 TB RAM available).
            # 16 × num_workers batches buffered; e.g. 16 × 16 = 256 batches cushion.
            prefetch_factor=16 if num_workers > 0 else None,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
