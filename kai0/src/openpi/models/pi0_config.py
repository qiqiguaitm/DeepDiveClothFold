import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_rtc import Pi0RTC


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Vision MLP LoRA: add rank-r LoRA adapters to both Dense layers of every
    # SigLIP MlpBlock. None/0 disables. Trainable adds ~(2 * 1152 + 2 * 4096) * rank
    # per block × 27 blocks ≈ 283K × rank params.
    vision_mlp_lora_rank: int | None = None
    vision_mlp_lora_alpha: float = 16.0

    # DCT frequency-domain loss on predicted actions (VLANeXt arXiv 2602.18532).
    # Penalizes high-frequency action jitter and emphasizes smooth trajectories.
    # Complements flow-matching MSE which weights all frequencies equally.
    use_dct_loss: bool = False
    dct_loss_weight: float = 0.1
    # Weights for lowest vs highest DCT frequency. Linearly interpolated across bins.
    # low_freq=1.0, high_freq=0.2 means low freqs counted 5x more than high freqs.
    dct_low_freq_weight: float = 1.0
    dct_high_freq_weight: float = 0.2

    # Image augmentation level for training: "mild" (default) or "aggressive".
    # "aggressive" is for deploy-robustness (D435→D405, pose/arm-spacing variation).
    augment_level: str = "mild"

    # X-VLA style soft prompt (per-domain learnable embeddings prepended to LLM input).
    # `soft_prompt_num_domains > 0` enables the hub; obs.dataset_id ∈ [0, num_domains).
    # `soft_prompt_len` = number of learnable tokens injected per sample.
    # When disabled (default 0), the model behaves identically to upstream pi0/pi05.
    soft_prompt_num_domains: int = 0
    soft_prompt_len: int = 0

    # Track C action head conditioning token (方案 A: Concat domain token at action expert input).
    # `action_head_cond_num_domains > 0` enables the hub; obs.dataset_id ∈ [0, num_domains).
    # The token is prepended to the suffix (action expert input only), so paligemma is
    # unaware of the domain — conditioning only modulates action denoising.
    # When disabled (default 0), the model behaves identically to upstream pi0/pi05 (and
    # to soft-prompt-only configs); old ckpts load fine because the new param is just
    # not instantiated. See docs/deployment/strategy/cross_embodiment_strategy.md §5.3 for design.
    action_head_cond_num_domains: int = 0

    # TAC (Training-time Action Conditioning) — paper 2512.05964 Algorithm 1.
    # When tac_enabled=True, compute_loss samples per-sample delay ∈ [0, tac_max_delay],
    # tokens in [0, delay) get time=1.0 (clean ground-truth action), postfix tokens get
    # sampled time. Loss is computed only on postfix tokens. adaRMS conditioning becomes
    # per-token (time emb shape (b, ah, emb)). tac_max_delay=6 for 30Hz × 200ms latency.
    tac_enabled: bool = False
    tac_max_delay: int = 6

    # Freeze policy. "none" = train all (default, LoRA still respected).
    # "only_soft_prompt" = freeze everything except `soft_prompt_hub` (Stage 2 of
    # the X-VLA 3-stage curriculum: align soft prompt for new domain before
    # unfreezing the rest).
    # "only_action_head_cond" = freeze everything except `action_head_cond_hub`
    # (Track C Stage 2 analogue).
    freeze_mode: str = "none"

    # ===== LeWM vision-encoder 变体 (pi05_from_paligemma_base_training_plan.md §9) =====
    # 默认 "siglip" → 行为与上游 pi0/pi05 完全一致 (旁路代码不触发)。仅当 "lewm" 时,
    # PyTorch 路径用 DINOv3-L/16(冻) + LeWM OctCompactor 替换 SigLIP (15 token vs 768)。
    # ⚠️ 纯加法: 这些字段全有默认值, 所有现有 config 不受影响。
    vision_encoder: str = "siglip"          # "siglip" (默认/官方) | "lewm"
    lewm_ckpt_path: str | None = None        # LeWM ckpt (取 compactor.{th,hl,hr}.*)
    lewm_dinov3_dir: str | None = None       # DINOv3-L/16 本地权重目录 (offline)
    lewm_freeze_compactor: bool = False      # True=冻 compactor(L-freeze) / False=随策略微调(L-nofreeze)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        # X-VLA Stage 2: freeze everything except soft_prompt_hub. Takes precedence
        # over LoRA filters — if you need LoRA + soft-prompt-only, compose manually.
        if self.freeze_mode == "only_soft_prompt":
            if self.soft_prompt_num_domains <= 0 or self.soft_prompt_len <= 0:
                raise ValueError(
                    "freeze_mode='only_soft_prompt' requires soft_prompt_num_domains>0 "
                    "and soft_prompt_len>0 (the module to keep trainable must exist)."
                )
            return nnx.Not(nnx_utils.PathRegex(".*soft_prompt_hub.*"))

        if self.freeze_mode == "only_action_head_cond":
            if self.action_head_cond_num_domains <= 0:
                raise ValueError(
                    "freeze_mode='only_action_head_cond' requires "
                    "action_head_cond_num_domains>0 (the module to keep trainable must exist)."
                )
            return nnx.Not(nnx_utils.PathRegex(".*action_head_cond_hub.*"))

        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(Pi0Config):
    """Config for Pi0RTC (real-time control) model. Uses same architecture as Pi0/Pi05 but sample_actions supports
    prev_action_chunk, inference_delay, execute_horizon for RTC guidance. Use this config when serving
    for RTC inference (e.g. agilex_inference_openpi_rtc.py). Set pi05=True for Pi05-based RTC (model_type PI05_RTC)."""

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_RTC if self.pi05 else _model.ModelType.PI0_RTC

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        from openpi.models.pi0_rtc import Pi0RTC

        return Pi0RTC(self, rngs=nnx.Rngs(rng))

    @override
    def load_pytorch(self, train_config, weight_path: str):
        """RTC model is JAX-only; use a JAX checkpoint with serve_policy and Pi0RTCConfig."""
        raise NotImplementedError(
            "Pi0RTC is only supported with JAX checkpoints. Use a checkpoint saved from OpenPi JAX training "
            "(params directory, not model.safetensors) and serve with --policy.config=pi05_rtc_flatten_fold_inference (or your RTC config name)."
        )


@dataclasses.dataclass(frozen=True)
class AdvantageEstimatorConfig(Pi0Config):
    # * Custom
    loss_action_weight: float = 1.0
    loss_value_weight: float = 1.0