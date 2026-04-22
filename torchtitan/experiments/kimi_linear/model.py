# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchtitan-idiom port of MoonshotAI/Kimi-Linear.

Reference: ``reference/modeling_kimi.py`` (verbatim fork from HF
``moonshotai/Kimi-Linear-48B-A3B-Base``). We keep the HF code for
diffing but do NOT import it — the HF version assumes Transformers'
PreTrainedModel + Cache, which don't compose with torchtitan's
trainer, FSDP, PP, or cache adapter.

Architectural faithfulness (per Kimi Linear tech report §5):

* Every layer's attention is EITHER :class:`KimiDeltaAttention` (KDA,
  linear-attention variant via fla-core) OR :class:`KimiMLAAttention`
  (NoPE MLA, faithful to Kimi's spec — not the DSv3 MLA in
  ``torchtitan.models.deepseek_v3``). Alternation pattern is
  layer-index-driven by ``config.kda_layers`` / ``config.full_attn_layers``.
* Every layer's FFN is EITHER :class:`KimiMLP` (dense SwiGLU, used on
  the first ``first_k_dense_replace`` layers) OR :class:`KimiMoE`
  (sparse sigmoid-gated grouped-topk, composed from torchtitan's
  common :class:`TokenChoiceTopKRouter` + :class:`GroupedExperts`
  infrastructure to get a training-capable forward that the HF
  release lacks).
* Pre-norm + residual structure identical to Kimi's reference.

AttnRes weaving is implemented as a separate subclass in
``attn_res_model.py`` (Phase 4c), matching the
``AttnResLlama3Model`` pattern in ``../attn_res/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError as err:  # pragma: no cover - import-time guard
    raise ImportError(
        "Kimi Linear KDA path requires fla-core. Run `pip install fla-core`."
    ) from err


# ----- Config -------------------------------------------------------------- #

@dataclass(slots=True)
class KimiLinearConfig:
    """Torchtitan-flavored config for Kimi Linear.

    Mirrors ``reference/configuration_kimi.py:KimiLinearConfig`` but
    as a plain dataclass (no HF ``PretrainedConfig`` machinery). All
    fields kept identical to the HF config.json knobs for the 48B-A3B
    release; scaling-law variants (194M..528M) override the ones that
    change per size (hidden_size, num_hidden_layers, etc.).

    The 1-indexed ``kda_layers`` / ``full_attn_layers`` convention is
    preserved from the HF config.json (so literal copy-paste from HF
    works).
    """

    # ---- vocabulary / embedding ----
    vocab_size: int = 163840
    hidden_size: int = 2304
    tie_word_embeddings: bool = False

    # ---- depth / width ----
    num_hidden_layers: int = 27
    intermediate_size: int = 9216  # dense MLP intermediate (layer 0 + shared experts)

    # ---- MLA (full-attn layers) ----
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # no GQA for Kimi 48B
    q_lora_rank: int | None = None  # None = no Q compression
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    rope_theta: float = 10000.0

    # ---- KDA (linear-attn layers) ----
    # linear_attn_config structure preserved from HF config.json
    kda_num_heads: int = 32
    kda_head_dim: int = 128
    kda_short_conv_kernel_size: int = 4
    # 1-indexed layer lists
    kda_layers: list[int] = field(default_factory=list)
    full_attn_layers: list[int] = field(default_factory=list)

    # ---- MoE ----
    num_experts: int | None = 256
    num_experts_per_token: int = 8
    moe_intermediate_size: int = 1024
    moe_renormalize: bool = True
    moe_router_activation_func: Literal["sigmoid", "softmax"] = "sigmoid"
    num_shared_experts: int = 1
    routed_scaling_factor: float = 2.446
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1

    # ---- norm / act ----
    rms_norm_eps: float = 1e-5
    hidden_act: Literal["silu", "gelu"] = "silu"

    # ---- init ----
    initializer_range: float = 0.02

    # Derived convenience
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def is_mla(self) -> bool:
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope
        )

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 0

    def is_kda_layer(self, layer_idx: int) -> bool:
        """1-indexed match, preserving HF config.json convention."""
        return (layer_idx + 1) in self.kda_layers


# ----- RMSNorm ------------------------------------------------------------- #

class KimiRMSNorm(nn.Module):
    """RMSNorm in fp32 internally, cast back to input dtype.

    Faithful to ``reference/modeling_kimi.py:KimiRMSNorm``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x32).to(in_dtype)


# ----- Dense SwiGLU MLP --------------------------------------------------- #

class KimiMLP(nn.Module):
    """SwiGLU dense FFN. Used for layer 0 (pre-MoE dense replace) AND
    as the shared-experts module in MoE layers.

    Faithful to ``reference:KimiMLP`` (gate_proj, up_proj, down_proj).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Literal["silu", "gelu"] = "silu",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unknown hidden_act: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ----- MLA (NoPE variant) -------------------------------------------------- #

class KimiMLAAttention(nn.Module):
    """Multi-head Latent Attention, Kimi NoPE variant.

    Faithful port of ``reference:KimiMLAAttention``. Key differences
    vs. DSv3 MLA:

    * ``q_lora_rank is None`` — no Q compression, Q is projected
      directly to ``num_heads × q_head_dim``.
    * ``mla_use_nope=True`` — no RoPE applied; the "rot" split is
      vestigial naming. Position info carried by the KDA recurrence.
    * K is split into ``kv_lora_rank + qk_rope_head_dim`` halves from
      ``kv_a_proj_with_mqa``; the "rope" half is broadcast across
      heads (not per-head), matching Kimi's structural choice.

    No cache path — we only support training-time forward. HF's
    ``past_key_values`` / ``Cache`` machinery is not ported since
    torchtitan training doesn't invoke incremental decoding.
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.use_nope = config.mla_use_nope
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** -0.5

        assert self.q_lora_rank is None, (
            "This port targets Kimi 48B-A3B's q_lora_rank=null path; "
            "q-compression not implemented yet."
        )
        assert self.use_nope, (
            "Only mla_use_nope=True is currently supported (Kimi 48B-A3B "
            "config). RoPE-on-MLA is not ported."
        )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal mask; no KV cache.

        Args:
            x: ``[B, T, D]`` hidden states.
        Returns:
            ``[B, T, D]`` attention output.
        """
        B, T, _ = x.shape

        # Q path: direct projection -> (B, T, H, q_head_dim) -> (B, H, T, q_head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # KV path: (B, T, kv_lora + qk_rope)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Expand low-rank KV to full heads:
        #   kv_b_proj: (kv_lora_rank) -> (num_heads * (qk_nope_head_dim + v_head_dim))
        kv_expanded = self.kv_b_proj(self.kv_a_layernorm(k_pass))
        kv_expanded = kv_expanded.view(
            B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_pass_expanded, v = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # k_rot is broadcast across heads: (B, T, qk_rope_head_dim) -> (B, H, T, qk_rope)
        k_rot = k_rot.view(B, 1, T, self.qk_rope_head_dim).expand(
            B, self.num_heads, T, self.qk_rope_head_dim
        )

        # Concat nope + rot halves (NO RoPE application under mla_use_nope)
        q_full = torch.cat((q_pass, q_rot), dim=-1)
        k_full = torch.cat((k_pass_expanded, k_rot), dim=-1)

        # Standard scaled-dot-product attention with causal mask
        # torch SDPA handles causal efficiently on CUDA; fall back to manual on CPU.
        attn_out = F.scaled_dot_product_attention(
            q_full, k_full, v, is_causal=True, scale=self.scaling,
        )  # (B, H, T, v_head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn_out)


# ----- KDA (Kimi Delta-rule Attention) ------------------------------------ #

class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention — linear-attention variant using
    fla-core's gated delta rule kernel.

    Faithful port of ``reference:KimiDeltaAttention`` minus the
    HF ``Cache`` / ``cu_seqlens`` / padding-aware fast-path (training
    fixed-seqlen doesn't exercise those).
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.kda_short_conv_kernel_size
        self.head_dim = config.kda_head_dim
        self.num_heads = config.kda_num_heads

        projection_size = self.head_dim * self.num_heads
        projection_k_size = projection_size  # k heads == v heads for Kimi

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        # Short causal convolutions with silu activation on q/k/v
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        # A_log: per-head log-decay parameter, init uniform in log([1, 16])
        # fla-core 0.5.0 expects shape [H]; HF reference had [1, 1, H, 1]
        # but it's fed through fused_kda_gate which reshapes internally.
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            )
        )

        # dt_bias: per-(head, head_dim) bias, shape [H * K]. Applied
        # inside fused_kda_gate as softplus(g + dt_bias). Kept zero-init
        # to reproduce HF reference's default init behavior.
        self.dt_bias = nn.Parameter(
            torch.zeros(projection_size, dtype=torch.float32)
        )

        # Low-rank forget-gate and output-gate projections
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        # Beta: per-head, per-token scalar (delta-rule learning rate)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Output RMSNorm with sigmoid-gated modulation from g, then o_proj
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid",
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without KV cache, fixed seq_len.

        Args:
            x: ``[B, T, D]`` hidden states.
        Returns:
            ``[B, T, D]`` KDA output.
        """
        _, T, _ = x.shape
        # mode selection matches reference: chunk for long, recurrent for short
        # training gate: chunk required (ref asserts this)
        mode = "fused_recurrent" if T <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 1) Q/K/V projection + short causal conv with silu
        q, _ = self.q_conv1d(x=self.q_proj(x), cache=None, output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), cache=None, output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), cache=None, output_final_state=False)

        # 2) Forget-gate g: (B,T,D) low-rank via f_a/f_b, reshape to
        #    (B, T, H, K) for fla-core 0.5.0's fused_kda_gate API:
        #      fused_kda_gate(g: [..., H, K], A_log: [H], dt_bias: [H*K])
        #      → [..., H, K] log-decay
        g_raw = self.f_b_proj(self.f_a_proj(x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        g = fused_kda_gate(g_raw, self.A_log, dt_bias=self.dt_bias)

        # 3) Beta: per-head, per-token learning-rate (delta-rule)
        beta = self.b_proj(x).float().sigmoid()

        # 4) Reshape to (..., H, D) for KDA kernel
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # 5) Run KDA op
        kda_fn = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o, _ = kda_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        # 6) Output gate + norm
        g_out = self.g_b_proj(self.g_a_proj(x))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g_out)  # FusedRMSNormGated: o * sigmoid(g_out), normed

        # 7) Reshape back and project
        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o)


# ----- MoE (training-capable via torchtitan.models.common.moe) ------------ #

class KimiMoE(nn.Module):
    """Kimi's sigmoid-gated grouped-topk MoE, implemented via
    torchtitan's training-capable MoE primitives.

    The HF reference's :class:`KimiSparseMoeBlock` raises
    NotImplementedError in training mode (line 667 of
    ``reference/modeling_kimi.py``) — it's inference-only. Since we
    only care about training here, we rebuild the MoE forward using
    torchtitan common building blocks:

    * :class:`TokenChoiceTopKRouter` — supports sigmoid scoring,
      grouped topk (``num_expert_groups`` / ``num_limited_groups``),
      ``route_norm`` (Kimi's ``moe_renormalize``), ``route_scale``
      (Kimi's ``routed_scaling_factor``), and ``expert_bias``
      (Kimi's ``e_score_correction_bias``).
    * :class:`GroupedExperts` — grouped-GEMM SwiGLU experts,
      training-capable, with a for-loop fallback for CPU.
    * Shared experts (``num_shared_experts``): a single
      :class:`KimiMLP` instance whose output is added to the routed
      output unconditionally.

    Load-balancing hook: ``expert_bias`` is registered as a buffer on
    the router and updated externally by torchtitan's
    ``register_moe_load_balancing_hook`` at optimizer-step time. This
    mirrors DSv3's auxiliary-loss-free routing protocol.
    """

    def __init__(self, config: KimiLinearConfig) -> None:
        super().__init__()
        from torchtitan.models.common import Linear
        from torchtitan.models.common.moe import (
            GroupedExperts,
            TokenChoiceTopKRouter,
        )

        assert config.num_experts is not None and config.num_experts > 0

        gate_cfg = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        )
        router_cfg = TokenChoiceTopKRouter.Config(
            num_experts=config.num_experts,
            gate=gate_cfg,
            num_expert_groups=(
                config.num_expert_group if config.num_expert_group > 1 else None
            ),
            num_limited_groups=(
                config.topk_group if config.num_expert_group > 1 else None
            ),
            top_k=config.num_experts_per_token,
            score_func=config.moe_router_activation_func,
            route_norm=config.moe_renormalize,
            route_scale=config.routed_scaling_factor,
        )
        experts_cfg = GroupedExperts.Config(
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
            num_experts=config.num_experts,
        )

        self.router = router_cfg.build()
        self.experts = experts_cfg.build()
        # expert_bias buffer for auxiliary-loss-free load balancing
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )

        # Shared experts: treat as one bigger MLP with intermediate =
        # moe_intermediate_size * num_shared_experts (Kimi reference
        # `shared_experts = KimiMLP(intermediate_size=moe_int * num_shared)`).
        if config.num_shared_experts > 0:
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.num_shared_experts
                ),
                hidden_act=config.hidden_act,
            )
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, D]``.
        Returns:
            ``[B, T, D]`` MoE output (routed + shared).
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # Router: produces top-k routing + per-expert token counts
        top_scores, selected_experts, num_tokens_per_expert = self.router(
            x_flat, expert_bias=self.expert_bias
        )

        # Experts forward: torchtitan's GroupedExperts expects
        # (selected_experts, top_scores, num_tokens_per_expert, x_flat)
        # and produces an aggregated per-token output. The exact
        # GroupedExperts.forward signature is called by torchtitan's
        # MoE module; here we call it directly for simplicity.
        expert_out = self.experts(
            x=x_flat,
            selected_experts_indices=selected_experts,
            top_scores=top_scores,
            num_tokens_per_expert=num_tokens_per_expert,
        )
        expert_out = expert_out.view(B, T, D)

        if self.shared_experts is not None:
            expert_out = expert_out + self.shared_experts(x)

        return expert_out


# ----- Decoder layer ------------------------------------------------------- #

class KimiDecoderLayer(nn.Module):
    """One transformer block: pre-norm + attention + residual +
    pre-norm + MoE/MLP + residual.

    Faithful to ``reference:KimiDecoderLayer``.
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Attention: KDA vs MLA by layer index
        if config.is_kda_layer(layer_idx):
            self.self_attn: nn.Module = KimiDeltaAttention(config, layer_idx)
            self.is_linear_attn = True
        elif config.is_mla:
            self.self_attn = KimiMLAAttention(config, layer_idx)
            self.is_linear_attn = False
        else:
            raise NotImplementedError(
                f"Layer {layer_idx}: neither KDA nor MLA configured."
            )

        # FFN: dense MLP for the first `first_k_dense_replace` layers, MoE otherwise.
        # Kimi's reference uses `layer_idx >= first_k_dense_replace` AND
        # `layer_idx % moe_layer_freq == 0`; we follow that.
        if (
            config.is_moe
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.ffn: nn.Module = KimiMoE(config)
            self.is_moe = True
        else:
            self.ffn = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
            self.is_moe = False

        self.input_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # FFN block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.ffn(x)
        x = residual + x
        return x


# ----- Top-level model ----------------------------------------------------- #

class KimiLinearModel(nn.Module):
    """Kimi Linear stack: embed -> decoder layers -> final RMSNorm -> LM head.

    No KV cache, no generation path. Training / loss is expected to be
    wired by the torchtitan trainer (cross-entropy over logits).

    ``_return_only_new_blocks`` and ``layers_per_block`` attributes
    are defined here so the Phase-3 PP cache adapter can toggle
    forward output shape once ``KimiLinearAttnResModel`` subclass
    adds the AttnRes block machinery. In the base (non-AttnRes) class
    the flag is ignored — forward always returns full hidden_states.
    """

    def __init__(self, config: KimiLinearConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        if config.tie_word_embeddings:
            # Not used on 48B-A3B (tie_word_embeddings=False) but kept for
            # smaller debug flavors that might tie.
            self.lm_head.weight = self.embed_tokens.weight

        # Hook for AttnRes subclass + PP adapter.
        self._return_only_new_blocks: bool = False

    def forward(
        self, input_ids: torch.Tensor | None = None, *, inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass. Returns logits of shape ``[B, T, vocab_size]``.

        Either ``input_ids`` ``[B, T]`` or ``inputs_embeds`` ``[B, T, D]``
        must be supplied. The inputs_embeds path is used by PP stages
        after stage 0 (where stage 0 does the embedding and passes the
        hidden state downstream).
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide either input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of input_ids / inputs_embeds.")

        h = (
            self.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)

    def init_weights(self, init_range: float | None = None) -> None:
        """Normal init with std=initializer_range. Embedding + Linear
        layers; norms stay at default ones. Called by torchtitan
        trainer after device placement.
        """
        std = init_range if init_range is not None else self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
