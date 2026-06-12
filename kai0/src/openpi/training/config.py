"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

import openpi.policies.agilex_policy as agilex_policy
import openpi.policies.arx_policy as arx_policy

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

# Per-host paths, configured by setup_env.sh at the repo root.
#   KAI0_DATA_ROOT     — base dir of deepdive_kai0/kai0 (holds data/ and local checkpoints/)
#   PYTORCH_CKPT_BASE  — root for ADVANTAGE_TORCH PyTorch pretrained weights
# gs:// paths are resolved via OPENPI_DATA_HOME by openpi.shared.download.
_OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/workspace/openpi_cache"))
_KAI0_LOCAL_ROOT = os.environ.get("KAI0_LOCAL_ROOT", "/home/tim/data_local")
_KAI0_DATA_ROOT = os.environ.get("KAI0_DATA_ROOT", "/data1/tim/workspace/deepdive_kai0/kai0")
_PYTORCH_CKPT_BASE = os.environ.get("PYTORCH_CKPT_BASE", "/path/to/pytorch_ckpt_base")


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional multi-dataset fan-out: 每个元素是一个 LeRobot repo_id (本地路径或 HF id).
    # 当非 None 时, create_torch_dataset 会按列表逐个构造 LeRobotDataset 然后 ConcatDataset,
    # `repo_id` 仍保留用作 asset_id 回退 / 日志显示 (可以指任一条或共同目录).
    # 训练 CLI 侧通常不直接填这个, 用 DataConfigFactory.datasets_yaml 自动 populate.
    repo_ids: Sequence[str] | None = None
    # X-VLA soft prompt support: per-repo domain index, parallel to `repo_ids`.
    # Populated when yaml entries specify `domain_id`/`dataset_id`. None disables
    # the InjectDatasetId transform (back-compat: hard-prompt training never sets this).
    dataset_ids: Sequence[int] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # Per-domain (per-dataset) norm for a single PRE-MERGED multi-domain dataset: {domain_id: {feature: NormStats}}.
    # When set, transform_dataset uses DomainNormalize (picks norm by obs.dataset_id) instead of the single Normalize.
    # This is the healthy per-DS-norm path (single LeRobotDataset, NOT the broken datasets_yaml/ConcatDataset).
    domain_norm_stats: dict | None = None
    # Per-domain training sample weights for a single pre-merged dataset, {domain_id: weight}. When set,
    # create_torch_data_loader uses a domain-weighted sampler so e.g. kai:vis can be balanced to ~1:1 by
    # probability (no disk copy). dataset_id per frame derived from task_index (ReadDatasetIdFromTaskIndex).
    domain_sample_weights: dict | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Filter specific episodes from the dataset (None = use all).
    episodes: list[int] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI0_RTC:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05 | _model.ModelType.PI05_RTC:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


def _load_repos_and_domain_ids(path: str) -> tuple[list[str], list[int] | None]:
    """Like _load_repo_ids_yaml but also returns per-repo domain indices when present.

    Entries can be plain strings (no domain_id) or dicts with `root` + optional
    `domain_id`/`dataset_id`. Returns (repo_ids, domain_ids). domain_ids is None
    when no entry specified one (back-compat); otherwise it has the same length
    as repo_ids with -1 filled for unspecified entries (caller must reject mixed).
    """
    import json, os
    p = pathlib.Path(os.path.expanduser(path))
    if not p.is_file():
        raise FileNotFoundError(f"datasets_yaml not found: {p}")
    ext = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if ext in (".yaml", ".yml"):
        import yaml
        doc = yaml.safe_load(text)
    elif ext == ".json":
        doc = json.loads(text)
    elif ext == ".txt":
        doc = [ln.split("#", 1)[0].strip() for ln in text.splitlines() if ln.split("#", 1)[0].strip()]
    else:
        raise ValueError(f"unsupported datasets_yaml extension {ext!r}")
    if isinstance(doc, dict):
        doc = doc.get("roots") or doc.get("datasets") or doc.get("paths") or []
    if not isinstance(doc, list) or not doc:
        raise ValueError(f"{p} parsed to empty / non-list roots: {doc!r}")

    repo_ids: list[str] = []
    domain_ids: list[int] = []
    any_domain = False
    for entry in doc:
        if isinstance(entry, str):
            repo_ids.append(entry)
            domain_ids.append(-1)
        elif isinstance(entry, dict):
            root = entry.get("root") or entry.get("path") or entry.get("repo_id")
            if not root:
                raise ValueError(f"entry in {p} has no 'root'/'path'/'repo_id' key: {entry!r}")
            repo_ids.append(str(root))
            did = entry.get("domain_id", entry.get("dataset_id", -1))
            if did != -1:
                any_domain = True
            domain_ids.append(int(did))
        else:
            raise ValueError(f"unsupported entry type in {p}: {entry!r}")
    if not any_domain:
        return repo_ids, None
    if any(d < 0 for d in domain_ids):
        raise ValueError(
            f"{p}: some entries specify domain_id/dataset_id and others don't. "
            "Specify on every entry or none."
        )
    return repo_ids, domain_ids


def _load_repo_ids_yaml(path: str) -> list[str]:
    """Parse a YAML / JSON / TXT file listing dataset roots and return the normalized list.

    支持三种格式 (按文件扩展名分派):
      * `.yaml` / `.yml`  — top-level list 或 {"roots": [...]}
      * `.json`           — top-level list 或 [{"root": path}, ...]
      * `.txt`            — 一行一个路径, 允许井号 '#' 行尾/整行注释

    Examples (YAML):
        roots:
          - /transfer-shanghai/KAI0/Task_A/2026-04-16/base
          - /transfer-shanghai/KAI0/Task_A/2026-04-17/base
    """
    import json
    import os

    p = pathlib.Path(os.path.expanduser(path))
    if not p.is_file():
        raise FileNotFoundError(f"datasets_yaml not found: {p}")

    ext = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if ext in (".yaml", ".yml"):
        import yaml  # lazy: yaml is already a transitive dep of openpi
        doc = yaml.safe_load(text)
    elif ext == ".json":
        doc = json.loads(text)
    elif ext == ".txt":
        doc = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                doc.append(line)
    else:
        raise ValueError(
            f"unsupported datasets_yaml extension {ext!r}; use .yaml/.yml/.json/.txt"
        )

    # Normalize the various shapes into list[str]
    if isinstance(doc, dict):
        doc = doc.get("roots") or doc.get("datasets") or doc.get("paths") or []
    if not isinstance(doc, list) or not doc:
        raise ValueError(f"{p} parsed to empty / non-list roots: {doc!r}")

    out: list[str] = []
    for entry in doc:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            root = entry.get("root") or entry.get("path") or entry.get("repo_id")
            if not root:
                raise ValueError(f"entry in {p} has no 'root'/'path'/'repo_id' key: {entry!r}")
            out.append(str(root))
        else:
            raise ValueError(f"unsupported entry type in {p}: {entry!r}")

    # Dedup in order
    seen: set[str] = set()
    deduped = [x for x in out if not (x in seen or seen.add(x))]
    return deduped


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Optional: path to a YAML/JSON/TXT listing multiple dataset roots to concat for training.
    # 支持三种格式, 用扩展名区分:
    #   .yaml/.yml  — {roots: [path, path, ...]}  或顶层直接 list
    #   .json       — [{"root": path}, ...]  或顶层 list[str]
    #   .txt        — 每行一个路径, 井号 '#' 注释与空行会被忽略
    # 提供时 create_base_config 会把解析结果写进 DataConfig.repo_ids;
    # data_loader.create_torch_dataset 看到 repo_ids 就 ConcatDataset 多路数据。
    # `repo_id` 仍用作 asset_id 回退 (norm stats 所在资产目录), 所以要么显式设
    # `assets.asset_id`, 要么让 `repo_id` 指向任一有 norm_stats 的 dataset。
    datasets_yaml: str | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        repo_ids: list[str] | None = None
        dataset_ids: list[int] | None = None
        if self.datasets_yaml:
            repo_ids, dataset_ids = _load_repos_and_domain_ids(self.datasets_yaml)
        if repo_ids and not repo_id:
            # datasets_yaml 有但 repo_id 缺失: 退而求其次用第一个作为 asset_id 锚点
            repo_id = repo_ids[0]
            asset_id = asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            repo_ids=repo_ids,
            dataset_ids=dataset_ids,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type not in (ModelType.PI0, ModelType.PI0_RTC),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LerobotAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    episodes: list[int] | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # mask state out (set to all zeros)
    mask_state: bool = False

    # if insert progress into prompt
    insert_advantage_into_prompt: bool = False

    # π0.7-style metadata dropout: probability of stripping `prompt_suffix_marker`
    # (and everything after) from the prompt during training. 0.0 = disabled (default).
    # Reference: π0.7 paper Sec V-E.
    prompt_suffix_dropout_rate: float = 0.0
    # Suffix marker used by DropPromptSuffix transform. Text from this marker onward is dropped.
    # Default matches "...Quality: X/5" format used by AWBC Quality-style prompts.
    prompt_suffix_marker: str = ". Quality:"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        # Use local variables instead of modifying frozen self
        default_prompt = self.default_prompt
        repack_transforms = self.repack_transforms

        # if prompt_from_task is True, set default_prompt to None and add prompt to data transforms
        if self.base_config and self.base_config.prompt_from_task:
            default_prompt = None
            original_repack = self.repack_transforms.inputs[0]
            new_structure = dict(original_repack.structure)
            new_structure["prompt"] = "prompt"
            repack_transforms = _transforms.Group(
                inputs=[_transforms.RepackTransform(new_structure)]
            )

        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                )
            ],
            outputs=[agilex_policy.AgilexOutputs()],
        )
        if self.insert_advantage_into_prompt:
            data_transforms.inputs.insert(0, _transforms.InsertAdvantageIntoPrompt())
        # π0.7-style prompt-suffix dropout: only inserted when rate > 0; otherwise no-op.
        if self.prompt_suffix_dropout_rate > 0.0:
            data_transforms.inputs.insert(
                0,
                _transforms.DropPromptSuffix(
                    dropout_rate=self.prompt_suffix_dropout_rate,
                    suffix_marker=self.prompt_suffix_marker,
                ),
            )

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            # Assuming first 13 dimensions are joints and last dimension is gripper
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            episodes=self.episodes,
        )

@dataclasses.dataclass(frozen=True)
class KaiVisMergedDataConfig(LerobotAgilexDataConfig):
    """Per-DS-norm + domain-conditioning on a single PRE-MERGED kai+vis dataset.

    Healthy single-source path (one LeRobotDataset, NOT the broken datasets_yaml/ConcatDataset).
    Domain is carried per-frame via `task_index` (kai=0, vis=1) → ReadDatasetIdFromTaskIndex →
    obs.dataset_id, which drives (a) DomainNormalize (per-DS norm: kai frames kai-norm, vis frames
    vis-norm) and (b) the action-head domain token. `domain_weights` → domain_sample_weights gives a
    probability-balanced sampler (e.g. kai:vis → 1:1) with NO disk copy.
    """
    domain_weights: tuple = ()  # (w_dom0, w_dom1, ...); empty → uniform (no weighted sampler)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base = super().create(assets_dirs, model_config)
        root = self.repo_id
        domain_norm_stats = {
            0: _normalize.load(_download.maybe_download(f"{root}/norm_domain0_kai")),
            1: _normalize.load(_download.maybe_download(f"{root}/norm_domain1_vis")),
        }
        # prepend ReadDatasetIdFromTaskIndex so dataset_id is set per-frame (RepackTransform then preserves it)
        repack = _transforms.Group(
            inputs=[_transforms.ReadDatasetIdFromTaskIndex(), *base.repack_transforms.inputs]
        )
        domain_sample_weights = (
            {i: float(w) for i, w in enumerate(self.domain_weights)} if self.domain_weights else None
        )
        return dataclasses.replace(
            base,
            repack_transforms=repack,
            domain_norm_stats=domain_norm_stats,
            domain_sample_weights=domain_sample_weights,
        )


@dataclasses.dataclass(frozen=True)
class LerobotARXDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # if insert progress into prompt
    insert_advantage_into_prompt: bool = False

    episodes: list[int] | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # mask state out (set to all zeros)
    mask_state: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        # Use local variables instead of modifying frozen self
        default_prompt = self.default_prompt
        repack_transforms = self.repack_transforms

        # if prompt_from_task is True, set default_prompt to None and add prompt to data transforms
        if self.base_config and self.base_config.prompt_from_task:
            default_prompt = None
            original_repack = self.repack_transforms.inputs[0]
            new_structure = dict(original_repack.structure)
            new_structure["prompt"] = "prompt"
            repack_transforms = _transforms.Group(
                inputs=[_transforms.RepackTransform(new_structure)]
            )

        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ARXInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                )
            ],
            outputs=[arx_policy.ARXOutputs()],
        )
        if self.insert_advantage_into_prompt:
            data_transforms.inputs.insert(0, _transforms.InsertAdvantageIntoPrompt())
        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            # Assuming first 13 dimensions are joints and last dimension is gripper
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            episodes=self.episodes,
        )

@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000

    # ===== In-training eval configuration =====
    # --- Tensor-based eval (uses train dataloader episode split) ---
    # Fraction of episodes held out for eval (0.0 = disabled, uses all for train).
    val_ratio: float = 0.0
    # Early-phase eval interval (steps). 0 = disabled.
    eval_interval_early: int = 0
    # Late-phase eval interval (steps). 0 = disabled.
    eval_interval_late: int = 0
    # Number of evals at early interval before switching to late.
    eval_early_count: int = 3
    # Number of eval batches averaged per eval call.
    eval_batches: int = 4
    # Diffusion steps for eval sample_actions (fewer = faster but less accurate).
    eval_num_diffusion_steps: int = 10

    # --- Inline eval (video-based, reads mp4 frames from a standalone val/ dir) ---
    # Path to a val dataset root (contains data/chunk-000/episode_*.parquet + videos/).
    # None = disabled. Orthogonal to val_ratio (different val source).
    inline_eval_val_root: str | None = None
    # Number of query frames sampled per val episode.
    inline_eval_n_frames: int = 20
    # Run inline eval every Nth save_interval boundary (1 = every save).
    inline_eval_every: int = 1
    # X-VLA soft prompt: domain index stamped on every inline-eval sample. Required
    # when model.soft_prompt_num_domains > 0 (the val set comes from one specific
    # domain — pass the matching index, e.g. 1 for vis val with soft_prompt order
    # [kai=0, vis=1]). None means "no stamping" — fine for non-soft-prompt configs.
    inline_eval_dataset_id: int | None = None

#************************advantage estimator***************************
    advantage_estimator: bool = False
    is_train: bool = True  # * Only use partial data in training
    # split:    str  = None  # one of ['train_tasks', 'val_tasks', 'heldout_tasks']
    # * Bugfix, only use train_tasks for training
    split: str = 'all'  # * Only use training tasks for training, choose from ['train', 'val', 'all']
    drop_last: bool = True  # If true, will drop the last incomplete batch.
    skip_norm_stats: bool = False
#************************advantage estimator***************************
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # AWBC Advantage Estimator (RECAP Stage 1, PyTorch) — registered so eval.py / eval_adv_est.py can
    # get_config() to rebuild the model arch for the trained ckpt
    # kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1 (metadata: pi05 / gemma_2b /
    # action_horizon=50 / max_token_len=200 / advantage_estimator=True). Used for Stage-2 labeling +
    # V0 sanity (PyTorch), NOT JAX training. data= is a placeholder (eval only reads cfg.model).
    TrainConfig(
        name="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_dagger_full",
            default_prompt="Flatten and fold the cloth.",
        ),
        advantage_estimator=True,
        num_train_steps=100_000,
    ),

    # Vis-native AWBC pipeline S2 (plan: awbc_vis_task_a_full_pipeline_plan.md) — retrain AE on
    # the freshly stage-labeled vis set (vis_awbc_merged_stage = 1699ep, stage_progress_gt∈{0.25,0.75}).
    # model MUST be AdvantageEstimatorConfig (train_pytorch asserts it); loss_value=1/loss_action=0
    # (only regress stage-progress diff). Init pi05_base (strict=False, value head new). PyTorch torchrun.
    TrainConfig(
        name="ADVANTAGE_TORCH_VIS_AWBC",
        model=pi0_config.AdvantageEstimatorConfig(
            pi05=True, action_dim=32, action_horizon=50, max_token_len=200,
            loss_action_weight=0.0, loss_value_weight=1.0,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_awbc_merged_stage",
            default_prompt="Flatten and fold the cloth.",
        ),
        pytorch_weight_path="/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot/pi05_base",
        advantage_estimator=True,
        num_train_steps=100_000,
        save_interval=10_000,
        batch_size=144,
    ),

    # ===== Cross-embodiment: per-DS-norm + Action-Head conditioning (2026-06-05) =====
    # Single PRE-MERGED kai+vis dataset `kai_vis_merged` (kai0_base+kai0_dagger=domain0 6512ep,
    # A_smooth800_dagger_full=domain1/vis 1033ep) → healthy single-source path (NOT datasets_yaml).
    #  • per-DS norm: DomainNormalize picks kai/vis norm by obs.dataset_id (task_index-derived)
    #  • domain token: action_head_cond_num_domains=2 (infer fixed vis=1)
    #  • 1:1 balance by probability (domain_weights), NO disk copy (DomainWeightedSampler)
    # Health gate: vis inline MAE must be ~0.008 量级, NOT ≈0.47 (that = collapse path).
    TrainConfig(
        name="pi05_kaivis_perdsnorm_cond",
        model=pi0_config.Pi0Config(pi05=True, action_head_cond_num_domains=2),
        data=KaiVisMergedDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/kai_vis_merged",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
            domain_weights=(1.0, 3.970),  # FRAME-level 1:1: kai 5.78M frames / vis 1.46M frames = 3.970 (NOT ep ratio 6.30; kai eps longer)
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # from-PaliGemma 自训 (pi05_from_paligemma_base_training_plan.md 路径 B1, naive-from-base):
    # 不 warm-start PI 的 pi05_base, 改从 PaliGemma VLM base 起 (action expert 随机初始化) → 双本体
    # co-train. 单变量 vs pi05_kaivis_perdsnorm_cond = weight_loader (PaliGemmaLocalWeightLoader, 离线
    # 本地 npz) + LR/warmup/steps (B1: peak 3e-5 / warmup 3k / 150k step, 随机 action expert 需更激进起步)
    # + cnbj 8卡 (fsdp8) + 路径换 /vePFS-North-E. vis = A (A_smooth800_dagger_full).
    # 数据 kai_vis_merged (7544ep / 7.12M frames; kai 5.778M / vis 1.346M → domain_weights vis=4.2925).
    # 提交为 cnbj 闲时任务 + 自动 resume (yaml). 收敛判据 = val MAE 曲线 (非 train loss); ~150k 看是否在轨.
    TrainConfig(
        name="pi05_kaivis_from_paligemma",
        model=pi0_config.Pi0Config(pi05=True, action_head_cond_num_domains=2),
        data=KaiVisMergedDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/kai_vis_merged",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
            domain_weights=(1.0, 4.2925),
        ),
        weight_loader=weight_loaders.PaliGemmaLocalWeightLoader(
            npz_path="/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=3_000, peak_lr=3e-5, decay_steps=150_000, decay_lr=3e-6),
        ema_decay=0.9999,
        num_train_steps=150_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # EXP-2 (corrected_plan_a §7.2): isolate kai's contribution by removing vis_dagger.
    # Single-variable change vs pi05_kaivis_perdsnorm_cond: vis source = pure smooth800
    # (A_new_smooth_800/base, 811ep/0.93M frames) instead of A_smooth800_dagger_full.
    # FRAME-level 1:1 weight recomputed (kai 5.777M / vis 0.930M = 6.213). 8-card cnsh (fsdp=8).
    TrainConfig(
        name="pi05_kaivis_cond_visS800",
        model=pi0_config.Pi0Config(pi05=True, action_head_cond_num_domains=2),
        data=KaiVisMergedDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/kai_vis_s800_merged",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
            domain_weights=(1.0, 6.246),  # FRAME-level 1:1: kai 5,777,710 frames / vis(pure smooth800) 925,055 = 6.246 (norm-build exact)
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # X-VLA Exp1: Hard Prompt Mixed (kai data + "kai " prefix, vis data + "vis " prefix)
    # tasks.jsonl patched per dataset in xvla/data/mixed_hard/
    TrainConfig(
        name="xvla_exp1_hard_prompt_mixed",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/xvla/data/mixed_hard/kai0_base",  # norm_stats source
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/mixed_repos_hard.yaml",
            default_prompt=None,
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # Same exp1 hard prompt baseline as xvla_exp1_hard_prompt_mixed_uc but uses a single
    # PRE-MERGED lerobot dataset instead of multi-dataset ConcatDataset. Avoids the uc02
    # NCCL deadlock that happens when 3 LeRobotDataset instances each do slow metadata
    # init + tolerance check. The merged dataset preserves the kai/vis hard prompt
    # distinction via tasks.jsonl (2 entries, kai=0/vis=1) + per-row task_index column.
    TrainConfig(
        name="xvla_exp1_hard_prompt_merged_uc",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/xvla_exp1_hard_merged",
            default_prompt=None,
            base_config=DataConfig(prompt_from_task=True),  # uses tasks.jsonl per-task lookup
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/shared/ubuntu/workspace/base_init_ckpts/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # exp1 hard-prompt baseline, uc01+02 16-GPU version (mirrors xvla_exp1_hard_prompt_mixed
    # but with uc-local data paths). Pairs with stage 1 on volc for full resource utilization.
    TrainConfig(
        name="xvla_exp1_hard_prompt_mixed_uc",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/mixed_hard/kai0_base",
            datasets_yaml="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/mixed_repos_hard_uc.yaml",
            default_prompt=None,
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/shared/ubuntu/workspace/base_init_ckpts/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=64,  # uc cluster — high parallel decode
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ──────────────────── X-VLA 3-stage curriculum (Exp2) ────────────────────
    # Stage 1 — Warm up soft prompt + full model on official (kai) data.
    # Both kai0_base and kai0_dagger stamped domain_id=0.
    # Init: pi05_base ckpt. soft_prompt_hub initialized with N(0, 0.02) per X-VLA.
    TrainConfig(
        name="xvla_stage1_kai_warmup",
        model=pi0_config.Pi0Config(
            pi05=True,
            soft_prompt_num_domains=2,
            soft_prompt_len=32,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage1_kai_only.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        # Val: a small kai holdout would be more meaningful, but reuse vis_v2_merged_val
        # for direct comparability across the 3 stages. dataset_id=1 (vis) at eval time
        # tests how well the vis soft prompt (uninitialized in stage 1) generalizes.
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # Stage 2 — Freeze backbone, only train soft_prompt_hub on vis (~5k step, lr 5e-4).
    # Init: Stage 1 final ckpt. Goal: align the vis soft prompt slot before unfreezing.
    TrainConfig(
        name="xvla_stage2_soft_prompt_only_vis",
        model=pi0_config.Pi0Config(
            pi05=True,
            soft_prompt_num_domains=2,
            soft_prompt_len=32,
            freeze_mode="only_soft_prompt",  # documentation; the freeze_filter below enforces it
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage2_3_vis_only.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            # ← update to final stage 1 ckpt path once stage 1 finishes
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/xvla_stage1_kai_warmup/xvla_stage1_kai_warmup/49999/params"
        ),
        # Freeze everything except soft_prompt_hub (matches Pi0Config.freeze_mode but inline here).
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*soft_prompt_hub.*")),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=5e-4, decay_steps=5_000, decay_lr=5e-5),
        ema_decay=None,           # EMA meaningless with ~64K trainable params
        num_train_steps=5_000,
        keep_period=1_000,
        save_interval=1_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=1,
        inline_eval_dataset_id=1,
    ),

    # Stage 3 — Full unfreeze on vis (50k step, cosine 1.5e-5 → 1.5e-6).
    # Init: Stage 2 final ckpt (soft prompt already adapted, now jointly fine-tune all params).
    TrainConfig(
        name="xvla_stage3_full_finetune_vis",
        model=pi0_config.Pi0Config(
            pi05=True,
            soft_prompt_num_domains=2,
            soft_prompt_len=32,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage2_3_vis_only.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            # ← update to final stage 2 ckpt path once stage 2 finishes
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/xvla_stage2_soft_prompt_only_vis/xvla_stage2_soft_prompt_only_vis/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # ===================================================================================
    # pi05 + delta-joint-actions on Task_A/base (kai0_base 3055 ep) from pi05_base.
    # 2026-05-22 用户决策: 对比 absolute baseline (mixed_pure2_1800/exp1 etc) 看 delta 是否帮 cloth fold.
    # delta_action_mask = make_bool_mask(6, -1, 6, -1) → 12 joint dims become delta, 2 gripper stay absolute.
    # ===================================================================================
    TrainConfig(
        name="pi05_flatten_fold_task_a_base_delta",
        model=pi0_config.Pi0Config(pi05=True),  # no conditioning, vanilla pi05
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=True,  # ← 关键: delta 训练
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===================================================================================
    # E3.6 — NO conditioning, kai+vis joint via datasets_yaml (2026-05-22; renamed 2026-06-04).
    # ⚠️ MISNOMER FIX: previously named `xvla_e3_6_per_ds_norm_no_cond`, but per-dataset
    # norm DOES NOT EXIST in the code path. `transform_dataset` (data_loader.py:340) applies
    # a SINGLE norm_stats (loaded from repo_id=kai0_base, config.py:351) to the whole
    # ConcatDataset; InjectDatasetId only stamps domain_id, never switches norm. The original
    # intent ("per-DS norm rescues cross-embodiment training") was never actually tested.
    # Result: COLLAPSE (val MAE@1=0.4706 predict-zero), same as the conditioning cells — the
    # culprit is the datasets_yaml/ConcatDataset path itself, NOT conditioning or norm scale.
    # See docs/training/history/experiments/conditioning_vs_action_representation_ablation.md
    # and docs/training/analysis/pi05_cross_embodiment_training_deep_dive.md.
    # Old job/ckpt: t-20260522201522-s72th (traceability).
    # ===================================================================================
    TrainConfig(
        name="xvla_e3_6_single_norm_no_cond",
        model=pi0_config.Pi0Config(pi05=True),  # no soft prompt, no action cond
        data=LerobotAgilexDataConfig(
            # repo_id only sets the SINGLE norm_stats asset (kai0_base); data flows via datasets_yaml.
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/e3_6_no_cond_kai_vis_joint.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===================================================================================
    # Track C: Action Head Conditioning Token (方案 A) — 2026-05-22 选定
    # Concat 1 learnable domain token to action expert input. paligemma unaware of domain.
    # 1:1 sparse-prefix 对照 Track B Soft Prompt (32 tokens in VLM input).
    # See docs/deployment/strategy/cross_embodiment_strategy.md §5.3 for design rationale.
    # ===================================================================================

    # Track C single-stage (2026-05-22 决策修订): 直接 kai+vis joint 50k from pi05_base.
    # 用户决策放弃 3-stage curriculum (经讨论 action expert 端注入 Stage 2 边际价值低,
    # 训练时间减半且实证性价比更高). domain_id=0 (kai) / 1 (vis) 通过 datasets_yaml 区分.
    TrainConfig(
        name="xvla_actcond_single_stage_joint",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_head_cond_num_domains=2,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            # Balanced sampling: vis × 7 to match kai 6512 ep, 49/51 split (避免 kai dominate)
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage3_kai_vis_joint_balanced.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # pi05 + TAC on A_new_pure_200 (NEW SOTA dataset, 200 ep '-new' curated).
    # Same hparams as vis_v2_full_tac but with the pure_200 data — for direct
    # comparison: does TAC also work on small high-quality curated dataset?
    # vis_v2_full_tac (49999): MAE@1=0.0147, @50=0.1148 — paper RTC2 ablation
    TrainConfig(
        name="pi05_flatten_fold_a_new_pure_200_tac",
        model=pi0_config.Pi0Config(
            pi05=True,
            tac_enabled=True,
            tac_max_delay=6,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # pi05 + TAC v2 — same hparams as _tac, but trained AFTER the pi0.py:335
    # convention bug fix (commit 5b6b75c). The _tac (no-v2) ckpt was trained on
    # the buggy code where prefix time=1.0 fed pure noise instead of clean GT,
    # making TAC training a no-op (verified: TAC v7 chunk |diff| = baseline).
    # Use v2 to retain the buggy 26k ckpt as A/B baseline.
    TrainConfig(
        name="pi05_flatten_fold_a_new_pure_200_tac_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            tac_enabled=True,
            tac_max_delay=6,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # pi05 + TAC (Training-time Action Conditioning, paper 2512.05964) on vis_v2_full.
    # Same data/hparams as pi05_flatten_fold_vis_v2_full but with tac_enabled=True.
    # Compare paper-RTC2 (TAC fine-tune) trained from pi05_base for 50k step.
    TrainConfig(
        name="pi05_flatten_fold_vis_v2_full_tac",
        model=pi0_config.Pi0Config(
            pi05=True,
            tac_enabled=True,
            tac_max_delay=6,  # 30Hz × 200ms latency
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_full",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===================================================================================
    # R1 / R2 — PyTorch DDP 原生训练 (plan §10.8, 2026-05-23 PM).
    # 与 JAX baseline `pi05_flatten_fold_vis_v2_full` 同数据 + 同 hparams, 仅训练框架不同.
    # 出 ckpt 用于 realtime_vla 选项 X PyTorch+Triton 5-10× 加速部署路径.
    # 启动: torchrun --nproc_per_node=16 scripts/train_pytorch.py <name> --exp_name ...
    # ===================================================================================

    # R1: PyTorch + absolute action — cnbj paths (Robot-North-H20 16 H20).
    TrainConfig(
        name="pi05_pytorch_vis_v2_full_absolute",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_full",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params"
        ),
        # PyTorch native init from converted safetensors (synced to cnbj 2026-05-23).
        pytorch_weight_path="/vePFS-North-E/vis_robot/openpi_cache/modelscope_cache/lerobot/pi05_base",
        pytorch_training_precision="bfloat16",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # R2: PyTorch + delta action — cnsh paths (robot-task 16 A100, in parallel with R1).
    TrainConfig(
        name="pi05_pytorch_vis_v2_full_delta",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_full",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        pytorch_weight_path="/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot/pi05_base",
        pytorch_training_precision="bfloat16",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # pi05 vis-only training on vis_5day_recent (filtered 05-18~05-22 from vis_v2_full).
    # 498 ep / 827k frames. Same hparams as pi05_flatten_fold_vis_v2_full, single-node 8 H20.
    TrainConfig(
        name="pi05_flatten_fold_vis_5day_recent",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_5day_recent",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===================================================================================
    # A_mirror200_pi05_pytorch (2026-05-27, plan §1 in A_mirror200_pi05_pytorch.md)
    # pure_200 dataset (200 ep + hflip mirror) trained via PyTorch native, 与 JAX SOTA
    # (`task_a_new_pure_200_new_norm`, MAE@1=0.0065) 1:1 对照, 隔离 "PyTorch vs JAX 框架" 变量.
    # 8× GPU FSDP, batch 128, 50k step, lr 1.5e-5 → 1.5e-6.
    # ===================================================================================
    TrainConfig(
        name="pi05_pytorch_a_new_pure_200",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        pytorch_weight_path="/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot/pi05_base",
        pytorch_training_precision="bfloat16",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===================================================================================
    # A_0423_0527 dual init JAX (2026-05-27, plan: A_0423_0527_excl_calibration_drift.md)
    # ===================================================================================
    # Data root-cause probe Exp-1 (走停/犹豫/cloth loop 排查):
    #   plans/data_root_cause_probe_experiments.md — 验证 H1 "投放过程污染".
    #   两个数据集均来自 vis_base 5-22 + 5-26 (各 100 ep, 合 200 ep):
    #   - `_no_release`: 裁掉每个 ep 开头投放等待静止段 (~7% 帧). 313419 frames.
    #   - `_raw`       : 同两天但不裁 (对照, 隔离 "裁投放" vs "200ep 规模"). 336917 frames.
    #   单变量=是否裁投放. init=mixed_1_clean (与 smooth_800 work 锚点一致), 40k step.
    #   ⚠️ norm_stats 各自重算 (compute_norm_stats.py), gripper/wrist 现状不动.
    # ===================================================================================
    TrainConfig(
        name="pi05_flatten_fold_A_0522_0526_no_release",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0522_0526_no_release",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),
    TrainConfig(
        name="pi05_flatten_fold_A_0522_0526_raw",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0522_0526_raw",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===== v2v3 数据时窗实验 (data_window_scaling, plan: future_plans/plans/v2v3_data_window_scaling_experiments.md) =====
    # Exp-A: v2/2026-05-18 单日 raw (未裁), 201 ep — cnsh 16卡
    TrainConfig(
        name="pi05_flatten_fold_A_0518_v2_201",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0518_v2_201",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # Exp-B: v3 5-18~5-28 窗口 (8 日, 955 ep, 已裁 v3) — cnbj 16卡 (路径=cnbj vePFS)
    TrainConfig(
        name="pi05_flatten_fold_v3_0518_0528",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_v3_0518_0528",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # Exp-C: 全 v3 排 5-16 (1940 ep) — cnbj 16卡, Exp-B 完成后手动提交
    TrainConfig(
        name="pi05_flatten_fold_v3_all_no0516",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_v3_all_no0516",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # Exp-D: v3 排除嫌疑窗 5-19~5-27 (≤5-18 排5-16 + 5-28 = 1335 ep) — cnbj 16卡
    # 验证用户假说: 之前 v3 训练问题源于混入 5-19~5-27 脏数据。= Exp-C 去掉嫌疑窗。
    # plan: v2v3_data_window_scaling_experiments.md Exp-D
    TrainConfig(
        name="pi05_flatten_fold_v3_excl_0519_0527",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_v3_excl_0519_0527",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===== dagger 有效性 + 训练方式对比 (plan: future_plans/plans/dagger_validity_and_finetune_comparison.md) =====
    # Exp-A (D1): smooth800全量 + dagger全量 (~1033 ep, 从头重训) — cnbj 16卡
    TrainConfig(
        name="pi05_flatten_fold_A_smooth800_dagger_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_dagger_full",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # ===== Task_AV1 (Vertical Fold v1 新 SOP) 首次基线 (plan: future_plans/plans/pi05_task_av1_vertical_fold_v1_baseline.md) =====
    # clone of pi05_flatten_fold_A_smooth800_dagger_full, cnsh 路径. 数据=Task_AV1_200 (200ep date-ordered),
    # warm-start mixed_1_clean, prompt=B 规范 (train==deploy 一字不差), val=Task_AV1_200_val (45ep 留出). cnsh 8 A100, 50k.
    TrainConfig(
        name="pi05_task_av1_vfold_v1_200",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_AV1_200",
            default_prompt="Flatten and fold the cloth. Vertical Fold v1.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_AV1_200_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # 夹爪 action 裁剪实验 (gripper_action_clip_experiment.md §4) — clone of
    # pi05_flatten_fold_A_smooth800_dagger_full, 仅数据集换成 clip 版 (action[:,[6,13]] ≤5mm→0,
    # state/arm 不动) + cnsh 路径 + 各自 norm_stats。单变量 = 夹爪 action 裁剪。cnsh 8卡 (Robot-GPU 开发机队列)。
    TrainConfig(
        name="pi05_smooth800_dagger_clip_all",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_dagger_clip_all",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # AWBC (RECAP traditional route): warm-start from smooth800 BC, fine-tune on A_smooth800_dagger_all
    # with per-frame advantage prompt ("...Advantage: positive/negative", ra>=0 split, 75.3% positive).
    # Same pi05 arch (advantage carried in the text prompt, NOT a domain token). Infer: always "positive".
    # norm_stats recomputed on the AWBC dataset (incl dagger action distribution). cnsh 8卡.
    TrainConfig(
        name="pi05_flatten_fold_awbc",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc",
            default_prompt=None,
            base_config=DataConfig(prompt_from_task=True),  # per-frame task_index → tasks.jsonl advantage prompt
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_new_smooth_800_step49999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # AWBC ablation (awbc_implementation_plan.md §当前执行计划 / smooth800-only): 控制变量 = 去掉 dagger,
    # 只用 smooth800 的 advantage-labeled 帧 (806 ep, 22%neg/78%pos)。测 demo-only 数据的 advantage 信号
    # (η²≈3% 天花板) 是否足以让 AWBC 学到东西 —— 直接对照 pi05_flatten_fold_awbc (smooth800+全dagger).
    # 与上面 config 逐字段一致, 仅 repo_id 不同 (A_smooth800_awbc, advantage label 从 dagger_all 版按帧抽取,
    # estimator 是 per-episode 独立 → 标签等同). default_prompt=None → inline-eval 同样会 skip, 训完离线评 MAE.
    TrainConfig(
        name="pi05_flatten_fold_awbc_smooth800only",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_awbc",
            default_prompt=None,
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_new_smooth_800_step49999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # v3.2 idle-trimming Step-2 (idle_data_trimming §3): front-trim (v3) + middle selective idle-downsample.
    # Single-variable vs the v3 baseline. init mixed_1_clean, 50k, norm 各自重算. cnbj 8卡.
    # Exp-1: ≤5-10 early "work" window v3.2 (985 ep).
    TrainConfig(
        name="pi05_flatten_fold_v32_le0510",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_v32_le0510",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999, num_train_steps=50_000, keep_period=10_000, save_interval=2_000,
        num_workers=16, batch_size=128, fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200, inline_eval_every=4,
    ),
    # Exp-2: full v3 (excl 5-16) v3.2 (1940 ep).
    TrainConfig(
        name="pi05_flatten_fold_v32_all",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_v32_all",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS-North-E/vis_robot/shared_ckpt/Task_A/mixed_1_clean/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999, num_train_steps=50_000, keep_period=10_000, save_interval=2_000,
        num_workers=16, batch_size=128, fsdp_devices=8,
        inline_eval_val_root="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200, inline_eval_every=4,
    ),

    # Exp-B (D2): smooth800抽样 + dagger 1:1 (~454 ep), best ckpt step40000 微调 20k — cnsh 8卡
    TrainConfig(
        name="pi05_flatten_fold_A_smooth800_dagger_1to1_ft",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_dagger_1to1",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/shared_ckpt/Task_A/smooth800_step40000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=20_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=20_000,
        keep_period=5_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # Dataset = 4-23~5-27 EXCEPT 5-16/18/19/20/21 (校准漂移期, v7 发现).
    # 13 dates / ~1059 ep (排 Class C 107 + End-snap 5 截尾).
    # 用 build_A_0423_0527.py 生成数据, 同 hparams 双 init 对照:
    #   - `A_0423_0527_pi05_JAX`    (override weight_loader 用 pi05_base)
    #   - `A_0423_0527_mixed1_JAX`  (override weight_loader 用 mixed_1_clean)
    # 8× GPU FSDP, batch 128, 50k step, lr 1.5e-5 → 1.5e-6.
    # 验证 v7 校准漂移假说: 排除漂移段后真机应 work (smooth-class).
    # ===================================================================================
    TrainConfig(
        name="pi05_flatten_fold_A_0423_0527",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_0423_0527",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        # Default init: pi05_base (Run-A: A_0423_0527_pi05_JAX).
        # For Run-B (mixed_1 init), override --weight-loader.params-path on CLI.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # pi05 vis-only training on vis_v2_full (16 v2 dates 04-23 → 05-22, 1409 ep / 1.93M frames).
    # User request 2026-05-23: train pure vis baseline (no kai mix) from pi05_base.
    # 8 GPU, batch 128, 50k step, lr 1.5e-5 → 1.5e-6. Volc Beijing/Shanghai.
    TrainConfig(
        name="pi05_flatten_fold_vis_v2_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_full",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # pi05 vis-only training on vis_v2_merged 895 ep, 8 GPU, batch 128, 50k step.
    # User request 2026-05-23: train pure vis baseline (no kai mix) from pi05_base.
    TrainConfig(
        name="pi05_flatten_fold_vis_v2_merged_only",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_merged",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6,
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # 同上但 use_delta_joint_actions=True (Action Cond × delta 变体, 2026-05-22 PM 决策).
    # 与 xvla_actcond_single_stage_joint (absolute) 对比 delta 表示是否帮 Action Cond 路线。
    TrainConfig(
        name="xvla_actcond_single_stage_joint_delta",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_head_cond_num_domains=2,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage3_kai_vis_joint_balanced.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=True,  # ← 关键变化: delta joints (gripper 仍 absolute via mask)
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # ⚠️ 以下 3-stage configs (xvla_actcond_stage1/2/3) 保留作技术参考, 2026-05-22
    # 用户决策走 single-stage joint, 不再使用 3-stage curriculum 路线。

    # C-Stage 1 — kai warmup with action_head_cond_hub (50k step on kai0_base+dagger).
    # Init: pi05_base. action_head_cond_hub[0] (kai) gets trained, slot[1] (vis) random.
    TrainConfig(
        name="xvla_actcond_stage1_kai_warmup",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_head_cond_num_domains=2,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage1_kai_only.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # C-Stage 2 — Freeze backbone, only train action_head_cond_hub on vis (~5k step, lr 5e-4).
    # Init: C-Stage 1 final ckpt. Goal: align vis action-cond token slot before unfreezing.
    TrainConfig(
        name="xvla_actcond_stage2_vis_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_head_cond_num_domains=2,
            freeze_mode="only_action_head_cond",
        ),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage2_3_vis_only.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            # ← update to final C-stage 1 ckpt path once it finishes
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/xvla_actcond_stage1_kai_warmup/xvla_actcond_stage1_kai_warmup/49999/params"
        ),
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*action_head_cond_hub.*")),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=5e-4, decay_steps=5_000, decay_lr=5e-5),
        ema_decay=None,  # EMA meaningless with ~1K trainable params
        num_train_steps=5_000,
        keep_period=1_000,
        save_interval=1_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=1,
        inline_eval_dataset_id=1,
    ),

    # C-Stage 3 — Full unfreeze, joint finetune kai+vis (50k step).
    # Init: C-Stage 2 final ckpt. C3.0 终态 = paper E3.8 baseline.
    TrainConfig(
        name="xvla_actcond_stage3_joint_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_head_cond_num_domains=2,
        ),
        data=LerobotAgilexDataConfig(
            # Use the same merged kai+vis dataset as exp1 Hard Prompt baseline for direct comparability.
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/xvla_exp1_hard_merged",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/xvla/data/stage3_kai_vis_joint.yaml",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            # ← update to final C-stage 2 ckpt path once it finishes
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/xvla_actcond_stage2_vis_only/xvla_actcond_stage2_vis_only/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
        inline_eval_dataset_id=1,
    ),

    # Task A: 100 ep originals (no mirror) + init from Task_A/mixed_1.
    # Submitted as volc 8-GPU job, batch 128, 50k step, cosine LR 1.5e-5 → 1.5e-6.
    TrainConfig(
        name="pi05_flatten_fold_a_new_100_base_mixed1",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),


    # Task A: 5/16-v2 (2 ep) + 5/19-v2 (46 ep) = 48 ep merged. Init from pi05_base.
    # Dataset prepared by build_task_a_new_100_5_16_5_19.py + gen_episodes_stats.py,
    # norm_stats recomputed via compute_norm_states_fast.py. uc-NFS shared path so
    # uc02/03 read same. Use exp-name="task_a_new_100_new_norm_base_pi0.5" on CLI.
    TrainConfig(
        name="pi05_flatten_fold_a_new_100_5_16_5_18_base_pi0.5",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100_5_16_5_18",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/shared/ubuntu/workspace/base_init_ckpts/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=64,            # uc convention
        batch_size=120,            # uc single-host 8 GPU = 15/card
        fsdp_devices=8,
        inline_eval_val_root="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100_5_16_5_18_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),


    # ─────────────────────────────────────────────────────────────────────
    # X-VLA hard-prompt domain conditioning ablation (2026-05-18)
    # Two paired experiments; same init/optim/steps, only prompt prefix + data domain differ.
    # ─────────────────────────────────────────────────────────────────────

    # 实验1: kai0 official (kai0_base + kai0_dagger), prompt prefix "kai ", 16-GPU uc01+02 cluster.
    TrainConfig(
        name="pi05_flatten_fold_kai0_official_kai_prompt",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
            datasets_yaml="/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data/kai0_official_repos.yaml",
            default_prompt="kai Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/val_kai0_official",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),

    # 实验2: vis_v2_merged (10 v2 subdirs merged to 895 ep contiguous), prompt prefix "vis ", 16-GPU volc 2-host.
    TrainConfig(
        name="pi05_flatten_fold_vis_base_v2_all_vis_prompt",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged",
            default_prompt="vis Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=16,
        inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v2_merged_val",
        inline_eval_n_frames=200,
        inline_eval_every=4,
    ),


    # Task E: stand up the fallen box — 73 ep, small dataset, sim01 local 5090s.
    # Clean init from pi05_base (not Task A ckpt) — gf0 packaged and transferred via TOS to sim01.
    # Freeze PaliGemma backbone (img + LLM main tower), train only Action Expert (llm.*_1) and top-level
    # projections (action_in_proj, action_out_proj, time_mlp_*). Shrinks train_state from ~65 GB to ~22 GB.
    # Single GPU batch=4 on sim01 (multi-GPU NCCL has orphan-worker pinned-memory issue; investigate in v2).
    # See docs/training/task_e_master_plan.md.
    TrainConfig(
        name="pi05_stand_box_normal",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_DATA_ROOT}/data/Task_E/base",
            default_prompt="stand up the fallen box",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"  # resolves via OPENPI_DATA_HOME cache
        ),
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*PaliGemma.*"),        # freeze PaliGemma (img + llm main tower)
            nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")), # but keep Action Expert LLM layers trainable
        ),
        ema_decay=None,           # EMA meaningless with frozen backbone; saves ~6.5 GB/card
        num_train_steps=25_000,   # fewer params to learn → converges faster than full FT
        keep_period=5_000,
        save_interval=2_000,
        num_workers=4,
        batch_size=8,             # 2 GPU FSDP (GPU0+GPU3, memory-having NUMAs); per-device bs=8
        fsdp_devices=2,
    ),
    # Task_A A_new_pure_1200 — gf1 #25, mixed_1 init, only -new dirs
    TrainConfig(
        name="pi05_flatten_fold_a_new_pure_1200",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_DATA_ROOT}/data/Task_A/self_built/A_new_pure_1200/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=2_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_DATA_ROOT}/data/Task_A/self_built/A_new_pure_1200/val",
        inline_eval_n_frames=200,
        inline_eval_every=2,
    ),
    # Task_A A_new_pure_600 — uc02 50k 训练, mixed_1 init
    # Data on local SSD /home/tim/local_ckpts/data (avoid lsyncd /data/shared bottleneck)
    TrainConfig(
        name="pi05_flatten_fold_a_new_pure_600",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_LOCAL_ROOT}/data/Task_A/self_built/A_new_pure_600",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=2_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_LOCAL_ROOT}/data/Task_A/self_built/A_new_pure_600_val",
        inline_eval_n_frames=200,
        inline_eval_every=2,
    ),
    # Task_A mix_b6000_p1200 — 实验1: init from Task_A/mixed_1
    # 14,985 train ep (5021 official + 8x 1258 self_built dup, ~1:2 batch ratio)
    # val_official: 100 ep (held out from kai0_base+dagger)
    # val_self_built: 30 ep (held out from -new + mirror, paired)
    # 50k step, peak_lr=1.5e-5 cosine to 1.5e-6, warmup=1k, ema=0.9999.
    # inline_eval_val_root: val_self_built (more sensitive to fine-tune).
    TrainConfig(
        name="pi05_flatten_fold_mix_b6000_p1200_init_mixed_1",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_LOCAL_ROOT}/Task_A/self_built/mix_b6000_p1200/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
            assets=AssetsConfig(asset_id="mix_b6000_p1200"),  # ckpt-side norm_stats (sim01 inference)
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6
        ),
        ema_decay=0.9999,
        num_train_steps=50_000,
        keep_period=10_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_LOCAL_ROOT}/Task_A/self_built/mix_b6000_p1200/val_self_built",
        inline_eval_n_frames=200,
        inline_eval_every=2,
    ),
    # Task P Stage 3: 20k steps long run, cosine decay full horizon, peak_lr 1.5e-5
    # between Stage 1 (1.25e-5) and Stage 2 (2.5e-5). Observe loss trajectory + overfit onset.
    # save_interval=2000 → 10 eval points; ETA ~22h.
    TrainConfig(
        name="pi05_pick_place_box_kai0_unfreeze_20k",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_DATA_ROOT}/data/Task_P/base",
            default_prompt="pick and place in box",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=1.5e-5, decay_steps=20_000, decay_lr=1.5e-6
        ),
        ema_decay=0.999,
        num_train_steps=20_000,
        keep_period=2_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_DATA_ROOT}/data/Task_P/val",
        inline_eval_n_frames=200,
        inline_eval_every=1,
    ),
    # Task_P vis_base 2026-05-09 — uc03 (gf4) 20k 训练, mixed_1 init
    # Data on local SSD /home/tim/local_ckpts/data
    TrainConfig(
        name="pi05_task_p_vis_base_20260509_unfreeze_20k",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_LOCAL_ROOT}/data/Task_P/vis_base_2026_05_09/train",
            default_prompt="Pick and place on blue",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=1.5e-5, decay_steps=20_000, decay_lr=1.5e-6
        ),
        ema_decay=0.999,
        num_train_steps=20_000,
        keep_period=2_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_LOCAL_ROOT}/data/Task_P/vis_base_2026_05_09/val",
        inline_eval_n_frames=200,
        inline_eval_every=1,
    ),
    # Task_PS pick blue stack on red — uc03 (gf4) 2026-05-08 training, 211 ep total
    # (180 train + 31 val random split with seed=42, frames=145k). Same hparams as unfreeze_20k_v2.
    # mixed_1 init, action=state semantics. Sim01 inference uses sidecar JSON to override asset_id.
    TrainConfig(
        name="pi05_task_ps_mixed_1_unfreeze_20k",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_DATA_ROOT}/data/Task_PS_all/train",
            default_prompt="Pick blue block, stack on red",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=1.5e-5, decay_steps=20_000, decay_lr=1.5e-6
        ),
        ema_decay=0.999,
        num_train_steps=20_000,
        keep_period=2_000,
        save_interval=2_000,
        num_workers=16,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_DATA_ROOT}/data/Task_PS_all/val",
        inline_eval_n_frames=200,
        inline_eval_every=1,
    ),
    # Task P unfreeze_20k v2: same hparams as unfreeze_20k, dataset = base_v2 (KAI0/Task_P/base/2026-04-21-v2,
    # 100 ep / 30,175 frames). Control variable vs original = dataset version (v2 has 100 ep vs original 84 ep).
    TrainConfig(
        name="pi05_pick_place_box_kai0_unfreeze_20k_v2",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id=f"{_KAI0_DATA_ROOT}/data/Task_P/v2",
            default_prompt="pick and place in box",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_KAI0_DATA_ROOT}/checkpoints/Task_A/mixed_1/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=1.5e-5, decay_steps=20_000, decay_lr=1.5e-6
        ),
        ema_decay=0.999,
        num_train_steps=20_000,
        keep_period=2_000,
        save_interval=2_000,
        num_workers=8,
        batch_size=128,
        fsdp_devices=8,
        inline_eval_val_root=f"{_KAI0_DATA_ROOT}/data/Task_P/val",
        inline_eval_n_frames=200,
        inline_eval_every=1,
    ),

    #************************advantage estimator***************************


    # ─────────────────────────────────────────────────────────────────────
    # Inference-only entries for gf3-trained delta ckpts (2026-05-23, sim01).
    # Paths reference gf3 vePFS; not loaded at inference (sidecar overrides
    # asset_id, datasets_yaml as needed). Kept here so the bundle's
    # base_config_name resolves.
    # ─────────────────────────────────────────────────────────────────────

    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _load_extra_config_from_env() -> None:
    """Apply per-ckpt config override if OPENPI_EXTRA_CONFIG is set.

    Sidecar contract — JSON file at <ckpt>/train_config.json:
      {
        "base_config_name": "<TrainConfig.name from main config.py>",
        "override_asset_id": "<asset_id used in <ckpt>/assets/<asset_id>/norm_stats.json>"
      }

    Behavior: clones _CONFIGS_DICT[base_config_name] and overrides
    `data.assets.asset_id` with override_asset_id. Lets a packed ckpt run on sim01
    without editing src/openpi/training/config.py per-experiment.
    """
    extra = os.environ.get("OPENPI_EXTRA_CONFIG")
    if not extra:
        return
    p = pathlib.Path(extra)
    if not p.is_file():
        raise FileNotFoundError(f"OPENPI_EXTRA_CONFIG points to missing file: {extra}")
    import json as _json
    spec = _json.loads(p.read_text())
    base_name = spec["base_config_name"]
    if base_name not in _CONFIGS_DICT:
        raise ValueError(
            f"{extra}: base_config_name {base_name!r} not in _CONFIGS_DICT (size={len(_CONFIGS_DICT)})."
            " Sync src/openpi/training/config.py first."
        )
    base = _CONFIGS_DICT[base_name]
    data_kw: dict = {}
    new_asset_id = spec.get("override_asset_id")
    if new_asset_id is not None:
        data_kw["assets"] = AssetsConfig(asset_id=new_asset_id)
    new_yaml = spec.get("override_datasets_yaml")
    if new_yaml is not None:
        new_yaml_path = pathlib.Path(new_yaml)
        if not new_yaml_path.is_absolute():
            new_yaml_path = p.parent / new_yaml_path
        data_kw["datasets_yaml"] = str(new_yaml_path)
    if data_kw:
        new_data = dataclasses.replace(base.data, **data_kw)
        new_cfg = dataclasses.replace(base, data=new_data)
        _CONFIGS_DICT[base_name] = new_cfg


_load_extra_config_from_env()


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
