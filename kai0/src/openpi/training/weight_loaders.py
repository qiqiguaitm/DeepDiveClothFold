import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Whitelist for keys present in the *model* but absent in the *ckpt*: keep the
        # model's init for these (instead of erroring). Currently:
        #   .*lora.*                  — LoRA adapter ranks
        #   .*soft_prompt_hub.*       — X-VLA soft prompt hub (Track B, new keys when
        #                                warming up from a pi05 base ckpt that predates it)
        #   .*action_head_cond_hub.*  — Track C Action Head Cond Token (方案 A), same
        #                                pattern as soft_prompt_hub
        return _merge_params(loaded_params, params, missing_regex=".*(lora|soft_prompt_hub|action_head_cond_hub).*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaLocalWeightLoader(WeightLoader):
    """Like PaliGemmaWeightLoader but loads the big_vision PaliGemma .npz from a LOCAL path (offline
    clusters where the official GCS bucket is unreachable / anon-revoked). Tolerates a community-mirror
    export that (a) lacks the top-level 'params/' wrapper (keys are 'img/...'/'llm/...') and/or (b) is
    f16 — the merge casts every loaded array to the model param's dtype, so precision is unified at load.
    Path resolved from `npz_path` field, else env `PALIGEMMA_NPZ`."""

    npz_path: str = ""

    def load(self, params: at.Params) -> at.Params:
        import os
        path = self.npz_path or os.environ.get("PALIGEMMA_NPZ", "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"PaliGemma npz not found: {path!r} (set field npz_path or env PALIGEMMA_NPZ)")
        with open(path, "rb") as f:
            flat = dict(np.load(f, allow_pickle=False))
        tree = flax.traverse_util.unflatten_dict(flat, sep="/")
        sub = tree["params"] if "params" in tree else tree  # community npz often lacks the 'params/' wrapper
        loaded_params = {"PaliGemma": sub}
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
