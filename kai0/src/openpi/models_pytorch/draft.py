"""Speculative-inference draft head for kai0 (ported from Realtime-VLA FLASH).

A `DraftChunkHead` is a single-layer Gemma "query decoder" that runs over the
*already-computed* VLM prefix embeddings (img+lang tokens) plus an optional robot
state token, and emits the *entire* action chunk in **one forward pass** -- no
flow-matching denoise loop. That single-shot property is the speed source: the
draft proposes a whole chunk, the full Action Expert only *verifies* it.

This is an ADDITIVE module: it imports nothing from kai0's existing inference
path and changes no existing file. Nothing here runs unless explicitly
instantiated. See docs/deployment/inference/realtime_vla/flash_impl_log.md.

Differences vs the upstream FLASH `draft.py` (LIBERO, single-arm 7-D action):
  * `out_dim`   default 7 -> kai0 dual-arm joint chunk is **14-D**.
  * `state_dim` was hardcoded 32 -> parameterized (kai0 raw joint state is 14;
    pi05 folds state into the language prefix so the explicit state token can be
    disabled entirely via `use_state_token=False`).
Everything else mirrors FLASH line-for-line to keep future upstream merges cheap.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding


class DraftChunkHead(nn.Module):
    """One-layer Gemma query decoder over prefix embeddings -> full action chunk."""

    def __init__(
        self,
        *,
        img_dim: int,
        chunk_m: int,
        hidden_dim: int = 256,
        out_dim: int = 14,
        state_dim: int = 32,
        use_state_token: bool = True,
        num_heads: int | None = None,
        num_kv_heads: int = 1,
        head_dim: int | None = None,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
        gemma_config: object | None = None,
    ) -> None:
        super().__init__()
        self.chunk_m = int(chunk_m)
        self.out_dim = int(out_dim)
        self.state_dim = int(state_dim)
        self.num_layers = int(max(1, num_layers))
        self.use_state_token = bool(use_state_token)
        self.pose_rot_dim = int(min(6, self.out_dim))
        self.attn_implementation = str(attn_implementation)

        if gemma_config is None:
            # Build a fresh single-layer Gemma config (from-scratch draft).
            self.hidden_size = int(img_dim)
            self.num_heads = int(num_heads or self._resolve_num_heads(self.hidden_size))
            self.num_kv_heads = int(max(1, int(num_kv_heads)))
            self.head_dim = int(head_dim or max(1, self.hidden_size // self.num_heads))
            gemma_config = CONFIG_MAPPING["gemma"](
                head_dim=int(self.head_dim),
                hidden_size=int(self.hidden_size),
                intermediate_size=int(hidden_dim),
                num_attention_heads=int(self.num_heads),
                num_hidden_layers=int(self.num_layers),
                num_key_value_heads=int(self.num_kv_heads),
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype=str(dtype).replace("torch.", ""),
            )
            gemma_config._attn_implementation = self.attn_implementation  # noqa: SLF001
        else:
            # Reuse an existing VLM Gemma config (e.g. paligemma language_model.config)
            # so the single block is shape-compatible with init_from_vlm_layer() and
            # with this transformers build (incl. adaRMS-patched layers).
            self.hidden_size = int(gemma_config.hidden_size)
            self.num_heads = int(gemma_config.num_attention_heads)
            self.num_kv_heads = int(getattr(gemma_config, "num_key_value_heads", 1))
            self.head_dim = int(getattr(gemma_config, "head_dim", max(1, self.hidden_size // self.num_heads)))
            if int(img_dim) != self.hidden_size:
                raise ValueError(f"img_dim={img_dim} must equal gemma_config.hidden_size={self.hidden_size}")

        if self.use_state_token:
            self._state_token = nn.Linear(int(self.state_dim), int(self.hidden_size))
        else:
            self._state_token = None
        self._action_queries = nn.Embedding(int(self.chunk_m), int(self.hidden_size))
        self._gemma_block = GemmaDecoderLayer(gemma_config, layer_idx=0)
        self._gemma_blocks_tail = nn.ModuleList(
            [GemmaDecoderLayer(gemma_config, layer_idx=i) for i in range(1, self.num_layers)]
        )
        self._rotary_emb = GemmaRotaryEmbedding(gemma_config)
        self._action_head = nn.Linear(int(self.hidden_size), int(self.out_dim))

    @staticmethod
    def _resolve_num_heads(dim: int) -> int:
        for heads in (8, 4, 2, 1):
            if dim % heads == 0:
                return heads
        return 1

    @staticmethod
    def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
        cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def _build_attention_mask(
        self, *, prefix_pad_masks: torch.Tensor, prefix_att_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        if prefix_pad_masks.ndim != 2:
            raise ValueError(f"expected prefix_pad_masks to be (B,S), got shape={tuple(prefix_pad_masks.shape)}")
        b, s = int(prefix_pad_masks.shape[0]), int(prefix_pad_masks.shape[1])
        device = prefix_pad_masks.device
        if prefix_att_masks is None:
            prefix_att_masks = torch.zeros((b, s), device=device, dtype=torch.bool)
        if prefix_att_masks.ndim != 2 or tuple(prefix_att_masks.shape) != tuple(prefix_pad_masks.shape):
            raise ValueError(
                f"expected prefix_att_masks to match prefix_pad_masks shape={tuple(prefix_pad_masks.shape)}, "
                f"got {tuple(prefix_att_masks.shape)}"
            )

        # number of extra leading tokens between prefix and the action queries:
        # 1 state token if enabled, else 0.
        n_state = 1 if self.use_state_token else 0
        extra_pad = torch.ones((b, n_state), device=device, dtype=torch.bool)
        extra_att = torch.zeros((b, n_state), device=device, dtype=torch.bool)
        prefix_plus_pad = torch.cat([prefix_pad_masks.to(dtype=torch.bool), extra_pad], dim=1)
        prefix_plus_att = torch.cat([prefix_att_masks.to(dtype=torch.bool), extra_att], dim=1)
        prefix_mask = self._make_att_2d_masks(prefix_plus_pad, prefix_plus_att)

        query_count = int(self.chunk_m)
        total = int(prefix_mask.shape[1] + query_count)
        mask = torch.zeros((b, total, total), device=device, dtype=torch.bool)
        prefix_len = int(prefix_mask.shape[1])
        mask[:, :prefix_len, :prefix_len] = prefix_mask
        mask[:, prefix_len:, :prefix_len] = prefix_plus_pad[:, None, :]
        mask[:, prefix_len:, prefix_len:] = True
        return mask

    def _build_position_ids(self, *, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        b = int(prefix_pad_masks.shape[0])
        device = prefix_pad_masks.device
        n_state = 1 if self.use_state_token else 0
        extra_pad = torch.ones((b, n_state), device=device, dtype=torch.bool)
        query_pad = torch.ones((b, int(self.chunk_m)), device=device, dtype=torch.bool)
        pad_mask = torch.cat([prefix_pad_masks.to(dtype=torch.bool), extra_pad, query_pad], dim=1)
        return (torch.cumsum(pad_mask.to(dtype=torch.int64), dim=1) - 1).clamp_min(0)

    def init_from_vlm_layer(self, layer: nn.Module) -> None:
        """Warm-start the draft's single Gemma block from a trained VLM layer 0."""
        self._gemma_block.load_state_dict(layer.state_dict(), strict=True)

    def init_from_vlm_layers(self, layers) -> None:
        """Warm-start all draft blocks from a list of trained VLM layers.

        layers[0] -> head block, layers[1:] -> tail blocks. If fewer layers are
        supplied than the draft has, the last available layer is reused (so a
        1-layer source can still seed a multi-layer draft)."""
        layers = list(layers)
        if not layers:
            return
        self._gemma_block.load_state_dict(layers[0].state_dict(), strict=True)
        for i, blk in enumerate(self._gemma_blocks_tail):
            src = layers[i + 1] if (i + 1) < len(layers) else layers[-1]
            blk.load_state_dict(src.state_dict(), strict=True)

    def forward(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor | None = None,
        last_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del last_actions  # reserved for future history conditioning (parity w/ FLASH)
        if prefix_embs.ndim != 3:
            raise ValueError(f"expected prefix_embs to be (B,S,H), got shape={tuple(prefix_embs.shape)}")
        if prefix_pad_masks.ndim != 2:
            raise ValueError(f"expected prefix_pad_masks to be (B,S), got shape={tuple(prefix_pad_masks.shape)}")
        if prefix_att_masks.ndim != 2:
            raise ValueError(f"expected prefix_att_masks to be (B,S), got shape={tuple(prefix_att_masks.shape)}")
        if int(prefix_embs.shape[1]) != int(prefix_pad_masks.shape[1]) or int(prefix_embs.shape[1]) != int(
            prefix_att_masks.shape[1]
        ):
            raise ValueError("prefix_embs, prefix_pad_masks, and prefix_att_masks must have matching sequence lengths")
        if int(prefix_embs.shape[2]) != int(self.hidden_size):
            raise ValueError(f"expected prefix_embs hidden size={self.hidden_size}, got {int(prefix_embs.shape[2])}")

        b = int(prefix_embs.shape[0])
        block_dtype = self._gemma_block.self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=block_dtype)

        tokens = [prefix_embs]
        if self.use_state_token:
            if robot_state is None or robot_state.ndim != 2 or int(robot_state.shape[1]) != self.state_dim:
                raise ValueError(
                    f"expected robot_state to be (B,{self.state_dim}) when use_state_token=True, "
                    f"got {None if robot_state is None else tuple(robot_state.shape)}"
                )
            if int(robot_state.shape[0]) != b:
                raise ValueError("prefix_embs and robot_state must have matching batch dimensions")
            state_token = self._state_token(robot_state.to(dtype=self._state_token.weight.dtype))[:, None, :].to(
                dtype=block_dtype
            )
            tokens.append(state_token)

        query_ids = torch.arange(int(self.chunk_m), device=prefix_embs.device, dtype=torch.long)[None, :].expand(b, -1)
        query_tokens = self._action_queries(query_ids).to(dtype=block_dtype)
        tokens.append(query_tokens)
        hidden_states = torch.cat(tokens, dim=1)

        mask_2d = self._build_attention_mask(prefix_pad_masks=prefix_pad_masks, prefix_att_masks=prefix_att_masks)
        attention_mask = torch.where(
            mask_2d[:, None, :, :],
            torch.zeros((), device=hidden_states.device, dtype=block_dtype),
            torch.full((), torch.finfo(block_dtype).min, device=hidden_states.device, dtype=block_dtype),
        )
        position_ids = self._build_position_ids(prefix_pad_masks=prefix_pad_masks)
        position_embeddings = self._rotary_emb(hidden_states, position_ids)
        for blk in (self._gemma_block, *self._gemma_blocks_tail):
            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
                adarms_cond=None,
            )[0]

        query_hidden = hidden_states[:, -int(self.chunk_m) :, :].to(dtype=self._action_head.weight.dtype)
        return self._action_head(query_hidden).to(dtype=torch.float32)
