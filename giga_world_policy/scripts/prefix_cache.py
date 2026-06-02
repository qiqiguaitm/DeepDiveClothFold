"""Exact (lossless) constant-prefix KV cache for CasualWorldActionTransformer.

Key fact exploited: in the action-only inference path the self-attention mask is
    mask[:s_r_end, s_r_end:] = -inf
so the state+ref *prefix* tokens never attend to the action tokens. Their inputs
(ref latent, state, timestep=0) are identical at every denoising step, hence every
block's prefix hidden-states — and therefore the prefix self-attention K/V — are
INVARIANT across all N denoising steps.

So we run the 145-token prefix through the 30 blocks ONCE, cache each block's prefix
self-attn K/V, and per step run only the 48 action tokens whose self-attn keys/values
are [cached_prefix_kv ; action_kv]. This reproduces the original math exactly
(no approximation, no retraining), but does the prefix work once instead of N times
and shrinks every per-step FFN / cross-attn from 193 to 48 tokens.
"""
import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from world_action_model.models.transformer_wa_casual import _get_qkv_projections


def _apply_rotary_emb(hs, freqs_cos, freqs_sin):
    x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hs)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hs)


def _self_attn_collect(attn, hs, rotary_emb):
    """Self-attention that also RETURNS (k, v) — functional, no cache mutation (cudagraph-safe)."""
    q, k, v = _get_qkv_projections(attn, hs, None)
    q = attn.norm_q(q); k = attn.norm_k(k)
    q = q.unflatten(2, (attn.heads, -1)); k = k.unflatten(2, (attn.heads, -1)); v = v.unflatten(2, (attn.heads, -1))
    q = _apply_rotary_emb(q, *rotary_emb); k = _apply_rotary_emb(k, *rotary_emb)
    out = dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    out = out.flatten(2, 3).type_as(q)
    out = attn.to_out[0](out); out = attn.to_out[1](out)
    return out, k, v


def _block_collect(blk, h, enc, temb, rope):
    """WanTransformerBlock.forward replicated, returning (h_out, attn1_k, attn1_v)."""
    sh, sc, g, csh, csc, cg = (blk.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
    sh, sc, g, csh, csc, cg = (sh.squeeze(2), sc.squeeze(2), g.squeeze(2), csh.squeeze(2), csc.squeeze(2), cg.squeeze(2))
    nh = (blk.norm1(h.float()) * (1 + sc) + sh).type_as(h)
    ao, k, v = _self_attn_collect(blk.attn1, nh, rope)
    h = (h.float() + ao * g).type_as(h)
    nh = blk.norm2(h.float()).type_as(h)
    ao = blk.attn2(nh, enc, None, None)
    h = h + ao
    nh = (blk.norm3(h.float()) * (1 + csc) + csh).type_as(h)
    ff = blk.ffn(nh)
    h = (h.float() + ff.float() * cg).type_as(h)
    return h, k, v


class CachedSelfAttnProcessor:
    """Self-attention processor with a write/read prefix KV cache. Self-attn only."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        self.mode = "off"  # "off" | "write" | "read"
        self.cache_k = None
        self.cache_v = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        q, k, v = _get_qkv_projections(attn, hidden_states, None)
        q = attn.norm_q(q)
        k = attn.norm_k(k)
        q = q.unflatten(2, (attn.heads, -1))
        k = k.unflatten(2, (attn.heads, -1))
        v = v.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)

        if self.mode == "write":
            # persistent buffers (stable pointers) so a CUDA-graphed read step can alias them
            if self.cache_k is None or self.cache_k.shape != k.shape:
                self.cache_k = torch.empty_like(k)
                self.cache_v = torch.empty_like(v)
            self.cache_k.copy_(k)
            self.cache_v.copy_(v)
            kk, vv = k, v
        elif self.mode == "read":
            kk = torch.cat([self.cache_k, k], dim=1)
            vv = torch.cat([self.cache_v, v], dim=1)
        else:
            kk, vv = k, v

        out = dispatch_attention_fn(q, kk, vv, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                                    backend=self._attention_backend, parallel_config=self._parallel_config)
        out = out.flatten(2, 3).type_as(q)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class PrefixCachedRunner:
    """Two-pass exact runner for the action-only denoising loop."""

    def __init__(self, model):
        self.m = model
        self.procs = []
        for blk in model.blocks:
            proc = CachedSelfAttnProcessor()
            blk.attn1.set_processor(proc)
            self.procs.append(proc)
        self._step_compiled = None
        self._write_compiled = None

    def compile_step(self, mode="reduce-overhead"):
        self._step_compiled = torch.compile(self._step_impl, mode=mode, fullgraph=False)

    def compile_prepare(self, mode="reduce-overhead"):
        # whole prefix encode (setup + functional write-pass) compiled -> CUDA-graph covers
        # rope / Conv3d patch-embed / condition_embedder too, not just the block loop.
        self._prepare_compiled = torch.compile(self._prepare_core, mode=mode, fullgraph=False)
        self._kv_buf = None
        self._encp_buf = None

    @torch.no_grad()
    def _prepare_core(self, ref_latents, noisy_latents, enc_raw, state):
        m = self.m
        hidden = torch.cat([ref_latents, noisy_latents], dim=2)
        bs, _, nf, h, w = hidden.shape
        p_t, p_h, p_w = m.config.patch_size
        num_ref = (h // p_h) * (w // p_w)

        state_states = m.action_encoder(state)
        num_state = state_states.shape[1]
        video_rope = m.rope(hidden)
        vid = m.patch_embedding(hidden.to(state_states.dtype)).flatten(2).transpose(1, 2)
        prefix_hidden = torch.cat([state_states, vid[:, :num_ref]], dim=1)

        seq_p = prefix_hidden.shape[1]
        ts0 = torch.zeros(bs * seq_p, device=hidden.device, dtype=hidden.dtype)
        _, tproj_p, enc_p, _ = m.condition_embedder(ts0, enc_raw, None, timestep_seq_len=seq_p)
        tproj_p = tproj_p.unflatten(2, (6, -1))

        extra_state_rope = m.action_rope(state_states)
        rope0 = torch.cat([extra_state_rope[0][:, :num_state], video_rope[0][:, :num_ref]], dim=1)
        rope1 = torch.cat([extra_state_rope[1][:, :num_state], video_rope[1][:, :num_ref]], dim=1)

        ks, vs = [], []
        hp = prefix_hidden
        for blk in m.blocks:
            hp, k, v = _block_collect(blk, hp, enc_p, tproj_p, (rope0, rope1))
            ks.append(k); vs.append(v)
        return tuple(ks) + tuple(vs) + (enc_p,)

    def _set_mode(self, mode):
        for p in self.procs:
            p.mode = mode

    @torch.no_grad()
    def prepare(self, ref_latents, noisy_latents, encoder_hidden_states, state):
        m = self.m
        p_t, p_h, p_w = m.config.patch_size
        self.num_state = state.shape[1]
        self.num_ref = (ref_latents.shape[-2] // p_h) * (ref_latents.shape[-1] // p_w)
        self._bs = ref_latents.shape[0]
        self._inner = m.config.num_attention_heads * m.config.attention_head_dim
        self.enc_proj_raw = encoder_hidden_states
        self._set_mode("read")  # subsequent step() calls read the cache

        if getattr(self, "_prepare_compiled", None) is not None:
            torch.compiler.cudagraph_mark_step_begin()
            outs = self._prepare_compiled(ref_latents, noisy_latents, encoder_hidden_states, state)
            enc_p = outs[-1]
            kv = outs[:-1]
            nb = len(kv) // 2
            if self._kv_buf is None:  # persistent buffers -> stable pointers for the step CUDA graph
                self._kv_buf = ([torch.empty_like(kv[i]) for i in range(nb)],
                                [torch.empty_like(kv[nb + i]) for i in range(nb)])
                self._encp_buf = torch.empty_like(enc_p)
            for i in range(nb):
                self._kv_buf[0][i].copy_(kv[i]); self._kv_buf[1][i].copy_(kv[nb + i])
                self.procs[i].cache_k = self._kv_buf[0][i]
                self.procs[i].cache_v = self._kv_buf[1][i]
            self._encp_buf.copy_(enc_p)
            self.enc_proj = self._encp_buf
        else:
            self.enc_proj = self._prepare_eager(ref_latents, noisy_latents, encoder_hidden_states, state)

    @torch.no_grad()
    def _prepare_eager(self, ref_latents, noisy_latents, enc_raw, state):
        outs = self._prepare_core(ref_latents, noisy_latents, enc_raw, state)
        enc_p = outs[-1]; kv = outs[:-1]; nb = len(kv) // 2
        for i in range(nb):
            self.procs[i].cache_k = kv[i]; self.procs[i].cache_v = kv[nb + i]
        return enc_p

    def set_action_rope(self, action_chunk):
        m = self.m
        full_extra = m.action_rope(torch.zeros(self._bs, self.num_state + action_chunk, self._inner,
                                               device=self.enc_proj.device, dtype=self.enc_proj.dtype))
        self.action_rope = (
            full_extra[0][:, self.num_state:self.num_state + action_chunk].contiguous(),
            full_extra[1][:, self.num_state:self.num_state + action_chunk].contiguous(),
        )

    def step(self, action, noise_t):
        if self._step_compiled is not None:
            torch.compiler.cudagraph_mark_step_begin()
            return self._step_compiled(action, noise_t).clone()
        return self._step_impl(action, noise_t)

    # ---------- BAC: Block-wise Adaptive Caching ----------
    # Refresh step (computes all blocks, writes the delta cache) and cached step
    # (reads deltas for skipped blocks, no mutation -> CUDA-graph safe) are split,
    # because in-graph cache mutation forces inductor to drop CUDA graphs.
    def init_bac(self, num_blocks):
        self.delta_buf = [None] * num_blocks
        self._bac_mask = tuple(True for _ in range(num_blocks))
        self._refresh_compiled = None
        self._cached_compiled = None

    def compile_bac(self, refresh_mode="max-autotune-no-cudagraphs", cached_mode="reduce-overhead"):
        self._refresh_compiled = torch.compile(self._refresh_impl, mode=refresh_mode, fullgraph=False)
        self._cached_compiled = torch.compile(self._cached_impl, mode=cached_mode, fullgraph=False)

    def _cond(self, action, noise_t):
        m = self.m
        bs, seq_a, _ = action.shape
        action_states = m.action_encoder(action)
        ts = noise_t.reshape(1).expand(bs * seq_a).to(action_states.dtype)
        temb_a, tproj_a, _, _ = m.condition_embedder(ts, self.enc_proj_raw, None, timestep_seq_len=seq_a)
        return action_states, temb_a, tproj_a.unflatten(2, (6, -1))

    def _finish(self, ha, temb_a):
        m = self.m
        shift, scale = (m.scale_shift_table.unsqueeze(0).to(temb_a.device) + temb_a.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
        ha = (m.norm_out(ha.float()) * (1 + scale) + shift).type_as(ha)
        return m.action_decoder(ha)

    @torch.no_grad()
    def _refresh_impl(self, action, noise_t):
        ha, temb_a, tproj_a = self._cond(action, noise_t)
        for i, blk in enumerate(self.m.blocks):
            out = blk(ha, self.enc_proj, tproj_a, self.action_rope, None)
            d = out - ha
            if self.delta_buf[i] is None or self.delta_buf[i].shape != d.shape:
                self.delta_buf[i] = torch.empty_like(d)
            self.delta_buf[i].copy_(d)
            ha = out
        return self._finish(ha, temb_a)

    @torch.no_grad()
    def _cached_impl(self, action, noise_t):
        ha, temb_a, tproj_a = self._cond(action, noise_t)
        for i, blk in enumerate(self.m.blocks):
            if self._bac_mask[i]:
                ha = blk(ha, self.enc_proj, tproj_a, self.action_rope, None)  # recompute (no cache write)
            else:
                ha = ha + self.delta_buf[i]  # reuse cached residual -> skip block
        return self._finish(ha, temb_a)

    def step_refresh(self, action, noise_t):
        if self._refresh_compiled is not None:
            return self._refresh_compiled(action, noise_t)
        return self._refresh_impl(action, noise_t)

    def step_cached(self, action, noise_t, mask):
        self._bac_mask = tuple(mask)  # static schedule -> dynamo specializes, cudagraph-safe
        if self._cached_compiled is not None:
            torch.compiler.cudagraph_mark_step_begin()
            return self._cached_compiled(action, noise_t).clone()
        return self._cached_impl(action, noise_t)

    @torch.no_grad()
    def _step_impl(self, action, noise_t):
        m = self.m
        bs, seq_a, _ = action.shape
        action_states = m.action_encoder(action)

        ts = noise_t.reshape(1).expand(bs * seq_a).to(action_states.dtype)
        temb_a, tproj_a, _, _ = m.condition_embedder(ts, self.enc_proj_raw, None, timestep_seq_len=seq_a)
        tproj_a = tproj_a.unflatten(2, (6, -1))

        ha = action_states
        for blk in m.blocks:
            ha = blk(ha, self.enc_proj, tproj_a, self.action_rope, None)

        # final norm_out + per-token modulation (temb_a), then action decoder
        shift, scale = (m.scale_shift_table.unsqueeze(0).to(temb_a.device) + temb_a.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
        ha = (m.norm_out(ha.float()) * (1 + scale) + shift).type_as(ha)
        return m.action_decoder(ha)
