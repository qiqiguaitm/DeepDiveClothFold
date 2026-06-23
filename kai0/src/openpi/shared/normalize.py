import json
import logging
import os
import pathlib

import numpy as np
import numpydantic
import pydantic


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
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
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
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

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


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


# ── Deploy-time gripper frame remap ──────────────────────────────────────────
# A ckpt trained when the gripper was configured max_range=100 (old, off-spec)
# encodes the gripper action/state dims in that old physical frame (full open
# ≈ 0.08-0.10m). After re-zeroing the real grippers to the official 0-70mm
# frame (see docs/deployment/data_collection/gripper_calibration.md), deploying
# such a ckpt would over/under-command the gripper.
#
# Fix: AFFINE-remap each gripper dim from the ckpt's OWN training range
# [q01, q99] (i.e. the actual span the gripper covered in training, NOT a fixed
# ratio) onto the real robot range [lo, hi] (default 0-0.07m). Because the same
# remap is applied to both the `state` (proprio in) and `actions` (command out)
# norm_stats, the proprio normalization and action unnormalization stay
# consistent. Degenerate dims (q99≈q01, e.g. an unused 2nd gripper on a
# single-arm task) are left untouched. Gated by env so it only affects old
# ckpts when explicitly enabled.
#
#   KAI0_GRIPPER_DEPLOY_REMAP=1            enable (default off -> no-op)
#   KAI0_GRIPPER_REAL_RANGE="0.0,0.07"    real [closed,open] in meters (action units)
#   KAI0_GRIPPER_DIMS="6,13"              gripper dims (left,right) in the 14/32-dim vector


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
    """Affine-remap gripper `dims` from training range [q01,q99] (or mean±2std)
    onto [lo,hi]. Returns new float arrays (mean, std, q01, q99). Degenerate
    dims (range < 1e-6) are skipped. Input/None preserved for q01/q99.
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
            logging.info(f"[gripper-remap]{tag} dim {d} degenerate (range≈0), skipped")
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
    NormStats with remapped gripper dims; non-gripper dims unchanged.
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
