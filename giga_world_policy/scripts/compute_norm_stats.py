import pathlib
from typing import Any

import numpy as np
import numpydantic
import pydantic
import tyro
from giga_datasets import load_dataset
from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import DeltaActions, PadStatesAndActions
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1) for i in range(vector_length)]
        else:
            if vector_length != self._mean.size:
                raise ValueError('The length of new vectors does not match the initialized vector length.')
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError('Cannot compute statistics for less than 2 vectors.')

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


class TransformDataset(Dataset):
    def __init__(self, dataset, data_transforms, return_keys):
        self.dataset = dataset
        self.data_transforms = data_transforms
        self.return_keys = return_keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        for transform in self.data_transforms:
            data = transform(data)

        result = {}
        for key in self.return_keys:
            values = np.asarray(data[key], dtype=np.float64)
            result[key] = values.reshape(-1, values.shape[-1])
        return result


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def serialize_json_with_mask(norm_stats: dict[str, NormStats], delta_mask) -> str:
    """同 serialize_json,但把 delta_mask 内嵌进输出 json(单一真值源)。

    action 统计就是在该 mask 下算的,绑定写入可杜绝 train/eval/serve 三处 mask 漂移。
    action_repr 仅供人读:全 False=abs、含 True=delta(关节减 state、夹爪绝对则为 mixed)。
    """
    import json as _json

    payload = _json.loads(serialize_json(norm_stats))
    mask = [bool(x) for x in delta_mask]
    payload["delta_mask"] = mask
    payload["action_repr"] = "abs" if not any(mask) else ("delta" if all(mask) else "mixed")
    return _json.dumps(payload, indent=2)


class GetEmbodimentId:
    """Assign a fixed embodiment id (passed via --embodiment_id).

    每次只对单一数据集跑 norm_stats,embodiment_id 由调用方显式指定,不再从
    info['robot_type'] 查表(避免 robot_type 命名不在映射表里时 KeyError)。
    """

    def __init__(self, embodiment_id: int):
        self.embodiment_id = int(embodiment_id)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data['embodiment_id'] = self.embodiment_id
        return data


def compute_norm_stats(
    data_paths: list[str],
    output_path: str | pathlib.Path,
    embodiment_id: int,
    delta_mask: list[bool],
    sample_rate: float = 1.0,
    action_chunk: int = 50,
    action_dim: int = 32,
    num_workers: int = 64,
    tolerance_s: float = 1e-3,
    batch_size: int = 256,
) -> None:
    """Compute normalization statistics and write them to JSON.

    This function loads dataset(s), applies transforms (embodiment id assignment,
    delta action repacking, and padding), accumulates running statistics for
    states and actions, and writes the results to ``output_path``.

    Args:
        data_paths: List of dataset paths to process.
        output_path: Destination file path for the computed norm stats JSON.
        embodiment_id: Embodiment id used as the key for the delta mask mapping.
        delta_mask: Per-dimension mask (1=delta, 0=absolute)
        sample_rate: Fraction of samples to process, in the range [0, 1].
        action_chunk: Temporal window size used when computing action deltas.
        action_dim: Expected action dimensionality used for padding.
        num_workers: Number of PyTorch DataLoader worker processes to use.
        tolerance_s: Allowed deviation (s) from 1/fps in lerobot's timestamp sync check
            and action-chunk gather. Default 1e-3 (>float32 量化误差@~3600s 长 episode,
            且 <<半帧距 0.0167s,不会取错帧)。规整后的网格本应恰为 1/fps,此值只吸收
            float32 存储噪声。
    """

    delta_masks: dict[int, list[int] | None] = {embodiment_id: delta_mask}
    data_or_config = [
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info=dict(
                action=action_chunk,
            ),
            meta_name='meta',
            skip_video_decoding=True,
            tolerance_s=tolerance_s,
        )
        for data_path in data_paths
    ]
    dataset = load_dataset(data_or_config)

    data_transforms = [
        GetEmbodimentId(embodiment_id),
        DeltaActions(mask=delta_masks),
        PadStatesAndActions(action_dim=action_dim),
    ]

    keys = ['observation.state', 'action']
    stats = {key: RunningStats() for key in keys}

    num_frames = int(sample_rate * len(dataset))

    transform_dataset = TransformDataset(dataset, data_transforms, keys)
    # 全量(sample_rate>=1)时每帧都要过,shuffle 只会让 Arrow 随机访问、缓存失效拖慢数据
    # 生产(实测瓶颈在 worker 取数而非主循环);顺序访问快得多,且 mean/std 与顺序无关、
    # q01/q99 在全量下顺序依赖可忽略。只有抽样(<1)时才需 shuffle 取无偏随机子集。
    shuffle = sample_rate < 1.0
    dataloader = DataLoader(
        transform_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    # 批量喂入 RunningStats: 每个样本 state=(1,dim)/action=(chunk,dim) 形状固定,默认
    # collate 堆成 (B, rows, dim);reshape 成 (B*rows, dim) 后一次 update,摊薄 batch_size=1
    # 时逐帧 Python/直方图调用的开销(实测 ~300 it/s → 批量后快 1~2 个数量级)。
    # mean/std 与逐帧完全等价;q01/q99 为直方图近似,与逐帧/shuffle 一样有微小顺序依赖。
    # 末批可能令处理帧数略超 num_frames(<batch_size),对统计无影响。
    seen = 0
    pbar = tqdm(total=num_frames)
    for batch_data in dataloader:
        bsz = batch_data[keys[0]].shape[0]
        for key in keys:
            arr = batch_data[key].numpy()
            stats[key].update(arr.reshape(-1, arr.shape[-1]))
        seen += bsz
        pbar.update(bsz)
        if seen >= num_frames:
            break
    pbar.close()

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f'Writing stats to: {output_path}')
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_json_with_mask(norm_stats, delta_mask))


if __name__ == '__main__':
    tyro.cli(compute_norm_stats)