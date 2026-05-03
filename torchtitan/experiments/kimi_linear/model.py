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
from torch.distributed.tensor import DTensor

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError as err:  # pragma: no cover - import-time guard
    raise ImportError(
        "Kimi Linear KDA path requires fla-core. Run `pip install fla-core`."
    ) from err


# ----- Config -------------------------------------------------------------- #

@dataclass(kw_only=True, slots=True)
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

    This class carries the Kimi model hyperparameters only. The
    torchtitan ``BaseModel.Config`` shim — ``KimiLinearSpec`` — lives
    in this module below and wraps one of these for ModelSpec
    registration.
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
# Use torch's ``nn.RMSNorm`` directly. Faithful to HF reference's
# ``KimiRMSNorm`` (same math: fp32 variance, cast back to input dtype).
# ``torchtitan.models.common.rmsnorm.RMSNorm`` is a Module-protocol
# wrapper around ``nn.RMSNorm``; we don't need the Config plumbing here
# since we're not going through the torchtitan Config.build() chain for
# the ported Kimi Linear backbone.


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


class KimiMLAInnerAttention(nn.Module):
    """SDPA-only inner attention module for KimiMLAAttention.

    Mirrors the DSv3 ``inner_attention`` pattern: pulls
    :func:`F.scaled_dot_product_attention` into a separate (parameter-free)
    submodule so the TP plan can wrap it with ``PrepareModuleInput(...,
    use_local_output=True)``. Under TP, the q/k/v projections produce DTensors
    sharded along the head axis; ``use_local_output=True`` converts them to
    plain Tensors before SDPA's internal kernel-selection dispatcher runs,
    avoiding the "aten.bmm got mixed Tensor and DTensor" failure inside the
    mem-efficient cutlass kernel path.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale,
        )


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
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # SDPA-only sub-module so the TP plan can wrap it with
        # use_local_output=True (DSv3 pattern). Has no parameters.
        self.inner_attention = KimiMLAInnerAttention()

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

        # Standard scaled-dot-product attention with causal mask.
        # PyTorch's default SDPA backend selection picks the right
        # kernel here: for Kimi MLA's asymmetric head_dim (Q/K=192,
        # V=128), flash-attention rejects (requires Q/K/V same dim)
        # and cuDNN attention is runtime-disabled in PyTorch 2.11,
        # so the *mem-efficient cutlass kernel* (fmha_cutlassF_bf16,
        # flash-style fused) is selected by default.
        #
        # Routing through ``self.inner_attention`` (a parameterless
        # submodule) is the DSv3 pattern: it lets ``apply_tp_kimi_linear``
        # wrap this call with ``PrepareModuleInput(use_local_output=True)``
        # so q/k/v are converted from DTensor (sharded on the head axis)
        # to plain Tensors before SDPA's mem-efficient cutlass kernel
        # path sees them — avoiding "aten.bmm got mixed Tensor and
        # DTensor" inside SDPA's internal dispatcher.
        attn_out = self.inner_attention(
            q_full, k_full, v, scale=self.scaling,
        )  # (B, H, T, v_head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn_out)


# ----- KDA (Kimi Delta-rule Attention) ------------------------------------ #


def _to_local_if_dtensor(t):
    """Strip DTensor wrapping for fla-core triton kernels.

    fla-core's chunk_kda / fused_kda_gate / ShortConvolution are Triton
    kernels that don't dispatch through DTensor. Under TP, KDA's
    self_attn is NoParallel-wrapped (params become DTensor(Replicate)
    on tp_mesh) and incoming x is also DTensor at the parent's
    boundary. KDA forward stashes the DTensor mesh+placements, strips
    DTensor from x and from each weight at the kernel call site, runs
    the kernels on plain tensors (each rank computes redundantly under
    Replicate), and re-DTensors at the end so the parent NoParallel
    output hook composes correctly.

    isinstance(t, DTensor) is the safe check that dynamo's fake-tensor
    mode honors (``hasattr(t, "to_local")`` is unreliable: dynamo's
    type tracking can elide attribute lookups on DTensor parameters).
    """
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def _local_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply ``linear`` with both weight and (optional) bias unwrapped to local.

    Used by :class:`KimiDeltaAttention.forward` so each projection can
    operate in plain-Tensor land alongside the fla-core triton kernels,
    even when the parent NoParallel(self_attn) wrap makes ``linear.weight``
    a DTensor(Replicate) on tp_mesh.
    """
    weight = _to_local_if_dtensor(linear.weight)
    bias = (
        _to_local_if_dtensor(linear.bias)
        if linear.bias is not None
        else None
    )
    return F.linear(x, weight, bias)


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
        # Under TP, the parent KimiDecoderLayer's NoParallel(self_attn)
        # wraps this forward: x arrives as DTensor(Replicate) on tp_mesh,
        # and all child params (q/k/v projections, conv1d weights,
        # A_log, dt_bias, FusedRMSNormGated) are DTensors on the same
        # mesh. The standard nn.Linear ops (DTensor x × DTensor weight)
        # dispatch correctly through DTensor's op set; the fla-core
        # triton kernels (causal_conv1d in ShortConvolution,
        # fused_kda_gate, chunk_kda, FusedRMSNormGated) do not. We
        # stash the input's DTensor metadata, run the body in plain-
        # tensor land, and re-DTensor at the end so the parent
        # NoParallel hook's prepare_output sees a DTensor.
        in_mesh = None
        in_placements = None
        if isinstance(x, DTensor):
            in_mesh = x.device_mesh
            in_placements = x.placements
        x = _to_local_if_dtensor(x)
        _, T, _ = x.shape
        # mode selection matches reference: chunk for long, recurrent for short
        # training gate: chunk required (ref asserts this)
        mode = "fused_recurrent" if T <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 1) Q/K/V projection + short causal conv with silu.
        # _local_linear unwraps DTensor weight to local before F.linear.
        # ShortConvolution.forward is patched at TP-init time to handle
        # DTensor input/weight by to_local + re-DTensor; we feed plain
        # x here so the patch is a no-op when x is already plain.
        q, _ = self.q_conv1d(
            x=_local_linear(self.q_proj, x), cache=None, output_final_state=False,
        )
        k, _ = self.k_conv1d(
            x=_local_linear(self.k_proj, x), cache=None, output_final_state=False,
        )
        v, _ = self.v_conv1d(
            x=_local_linear(self.v_proj, x), cache=None, output_final_state=False,
        )

        # 2) Forget-gate g: (B,T,D) low-rank via f_a/f_b, reshape to
        #    (B, T, H, K) for fla-core 0.5.0's fused_kda_gate API:
        #      fused_kda_gate(g: [..., H, K], A_log: [H], dt_bias: [H*K])
        #      → [..., H, K] log-decay
        g_raw = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        g = fused_kda_gate(
            g_raw,
            _to_local_if_dtensor(self.A_log),
            dt_bias=_to_local_if_dtensor(self.dt_bias),
        )

        # 3) Beta: per-head, per-token learning-rate (delta-rule)
        beta = _local_linear(self.b_proj, x).float().sigmoid()

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
        g_out = _local_linear(self.g_b_proj, _local_linear(self.g_a_proj, x))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)
        # FusedRMSNormGated.forward is patched at TP-init time too, so
        # it handles DTensor weight transparently. We pass plain o + g_out
        # here (both are plain after the to_local+linear chain).
        o = self.o_norm(o, g_out)  # o * sigmoid(g_out), normed

        # 7) Reshape back and project
        o = rearrange(o, "b t h d -> b t (h d)")
        out = _local_linear(self.o_proj, o)

        # Re-wrap the output as DTensor so the parent NoParallel hook
        # gets the type it expects. Replicate placement matches the
        # incoming x's placement (input_layernorm output).
        if in_mesh is not None and in_placements is not None:
            out = DTensor.from_local(
                out, in_mesh, in_placements, run_check=False,
            )
        return out


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
        # Full reuse: torchtitan.models.common.moe.MoE already wires
        # router + TokenReorderer + GroupedExperts + shared_experts +
        # expert_bias buffer + auxiliary-loss-free load balancing. We
        # just translate Kimi's config knobs into MoE.Config.
        from torchtitan.models.common.feed_forward import FeedForward
        from torchtitan.models.common.linear import Linear
        from torchtitan.models.common.moe import (
            GroupedExperts,
            MoE,
            TokenChoiceTopKRouter,
        )

        assert config.num_experts is not None and config.num_experts > 0

        router_cfg = TokenChoiceTopKRouter.Config(
            num_experts=config.num_experts,
            gate=Linear.Config(
                in_features=config.hidden_size,
                out_features=config.num_experts,
                bias=False,
            ),
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
            # torch._grouped_mm fuses all expert GEMMs into one batched call.
            # For-loop path (use_grouped_mm=False) launches one GEMM per
            # expert per layer, which hurts tensor core utilization badly
            # on small per-expert batches (typical at LOCAL_BS<=8). Requires
            # PyTorch ≥ 2.5 with grouped_mm support; works on Hopper / Ada /
            # Blackwell; CPU path raises so MoE forward is GPU-only.
            use_grouped_mm=True,
        )

        # Shared experts — Kimi's reference uses KimiMLP at
        # intermediate = moe_int * num_shared_experts. We swap to
        # torchtitan's FeedForward for consistency with MoE.Config;
        # the SwiGLU math is identical.
        shared_cfg = None
        if config.num_shared_experts > 0:
            shared_dim = config.moe_intermediate_size * config.num_shared_experts
            shared_cfg = FeedForward.Config(
                w1=Linear.Config(
                    in_features=config.hidden_size,
                    out_features=shared_dim,
                    bias=False,
                ),
                w2=Linear.Config(
                    in_features=shared_dim,
                    out_features=config.hidden_size,
                    bias=False,
                ),
                w3=Linear.Config(
                    in_features=config.hidden_size,
                    out_features=shared_dim,
                    bias=False,
                ),
            )

        moe_cfg = MoE.Config(
            num_experts=config.num_experts,
            experts=experts_cfg,
            router=router_cfg,
            score_before_experts=True,
            load_balance_coeff=1e-3,
            shared_experts=shared_cfg,
        )
        self._moe = moe_cfg.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._moe(x)


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

        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
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
        # ModuleDict (not ModuleList) so pipeline_module_split preserves
        # layer-id string keys and the adapter's layer_to_stage discovery
        # works unchanged. Matches the attn_res/ experiment's pattern.
        self.layers = nn.ModuleDict(
            {str(i): KimiDecoderLayer(config, i)
             for i in range(config.num_hidden_layers)}
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        if config.tie_word_embeddings:
            # Not used on 48B-A3B (tie_word_embeddings=False) but kept for
            # smaller debug flavors that might tie.
            self.lm_head.weight = self.embed_tokens.weight

        # Hook for AttnRes subclass + PP adapter.
        self._return_only_new_blocks: bool = False

    def forward(self, tokens: torch.Tensor, *,
                inputs_embeds: torch.Tensor | None = None,
                vision_embeds: torch.Tensor | None = None,
                image_mask: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        """Forward pass with PP-split awareness.

        Args:
            tokens: Either ``[B, T]`` int64 token ids (stage 0 / non-PP)
                OR ``[B, T, D]`` hidden state from upstream PP stage
                (middle / last). Dispatch is decided by presence of
                ``self.embed_tokens`` (pipeline_module_split strips it
                off non-first stages).
            inputs_embeds: Optional ``[B, T, D]`` pre-computed
                embeddings. When provided, ``embed_tokens`` is skipped
                entirely (`tokens` is ignored as long as it's a valid
                placeholder dispatch on the right device). Used by
                multimodal training where image-token positions are
                replaced with vision-projector outputs before the LM
                forward — keeps the call as a single FSDP-root forward.
            **kwargs: Ignored. Accepts ``attention_masks=None`` and
                ``positions=...`` that torchtitan's Trainer / Validator
                may inject for FlexAttention / CP paths — Kimi Linear
                uses plain SDPA + KDA Triton kernels and doesn't need
                them.

        Returns:
            * Non-last PP stage: ``[B, T, D]`` hidden state to forward
              to the next stage.
            * Last stage / non-PP: ``[B, T, vocab_size]`` logits.
        """
        if inputs_embeds is not None:
            h = inputs_embeds
        elif self.embed_tokens is not None:
            h = self.embed_tokens(tokens)
            # Multimodal scatter: replace embed positions for image tokens
            # with externally-supplied vision_embeds. Done INSIDE this
            # forward so FSDP sees a single root call (calling
            # embed_tokens externally would split the root).
            if vision_embeds is not None and image_mask is not None:
                h = h.clone()
                h[image_mask] = vision_embeds.reshape(-1, vision_embeds.size(-1)).to(h.dtype)
        else:
            h = tokens  # middle/last PP stage: tokens IS the hidden state
        for layer in self.layers.values():
            h = layer(h)
        if self.norm is not None:
            h = self.norm(h)
        if self.lm_head is not None:
            return self.lm_head(h)
        return h  # middle PP stage: ship hidden state downstream

    def verify_module_protocol(self) -> None:
        """No-op: our internals are plain nn.Module (not the torchtitan
        ``Module`` protocol), since KimiLinearModel ports the HF
        reference layer-by-layer rather than going through the Config
        chain. Trainer calls this post-build; overriding as no-op keeps
        the FSDP + loss + optimizer paths intact without requiring every
        sub-module to register as a ``Module.Config``-built instance.
        """
        return None

    def get_attention_masks(self, *args, **kwargs):
        """Return ``None`` — KDA + MLA both use plain SDPA / Triton paths
        and don't take an external ``attention_masks`` kwarg through
        ``forward``. torchtitan's Validator and Trainer call this to
        precompute attention masks for FlexAttention/VarlenAttention
        models; for our SDPA-style stack the right answer is no mask
        passthrough.

        Defined as method (not raise NotImplementedError) so the trainer
        and validator paths don't crash on AttributeError. Returning
        ``None`` causes ``extra_kwargs["attention_masks"] = None`` and
        our forward signature ``(tokens)`` simply ignores extra kwargs
        the trainer might try to pass.
        """
        return None

    def init_weights(self, init_range: float | None = None, **kwargs) -> None:
        """Initialize *all* parameters and buffers from scratch.

        This must be exhaustive because torchtitan's trainer flow is
        ``meta-build → parallelize_fn (FSDP wrap) → to_empty(device=cuda)
        → init_weights``. ``to_empty`` discards every value set inside
        ``__init__`` (including RMSNorm.weight=1 defaults, KDA's A_log,
        dt_bias, ShortConvolution kernels, MoE expert weights, and
        load-balance buffers). Anything we forget here stays at whatever
        garbage ``torch.empty`` left on the device — which silently
        zeroes RMSNorm scales and produces near-uniform logits with no
        learning signal.
        """
        std = init_range if init_range is not None else self.config.initializer_range

        # Pass 1: leaf modules with well-typed init contracts.
        for m in self.modules():
            cls_name = type(m).__name__
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif cls_name in ("ShortConvolution", "FusedRMSNormGated"):
                # fla-core modules ship reset_parameters()
                m.reset_parameters()

        # Pass 2: KDA per-layer raw Parameters (A_log, dt_bias) that
        # don't belong to any nn.Module subclass we can dispatch on.
        for layer in self.layers.values():
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if hasattr(attn, "A_log"):
                # Match KimiDeltaAttention.__init__: log(uniform(1, 16))
                attn.A_log.data.uniform_(1.0, 16.0).log_()
            if hasattr(attn, "dt_bias"):
                nn.init.zeros_(attn.dt_bias)

        # Pass 3: torchtitan MoE — GroupedExperts holds raw [E, ...]
        # parameter tensors (not nn.Linear), and MoE/router carry
        # auxiliary-loss-free load-balance buffers that must start at 0.
        for m in self.modules():
            cls_name = type(m).__name__
            if cls_name == "GroupedExperts":
                for name in ("w1", "w2", "w3"):
                    p = getattr(m, name, None)
                    if isinstance(p, nn.Parameter):
                        nn.init.normal_(p, mean=0.0, std=std)
            elif cls_name == "MoE":
                for buf_name in ("expert_bias", "tokens_per_expert"):
                    buf = getattr(m, buf_name, None)
                    if buf is not None:
                        buf.zero_()


# ----- ModelSpec shim: BaseModel.Config wrapper --------------------------- #

# Imports at module bottom to keep the KimiLinear* classes usable as plain
# nn.Modules without dragging the torchtitan.protocols.model chain in
# when used from the CPU tests.


@dataclass(kw_only=True, slots=True)
class KimiLinearSpec:
    """``BaseModel.Config``-compatible shim that wraps a
    :class:`KimiLinearConfig` and an optional ``num_blocks`` (None =
    plain :class:`KimiLinearModel`; integer N = :class:`KimiLinearAttnResModel`
    with ``num_blocks=N``).

    Methods implemented for torchtitan integration:

    * :meth:`build` — returns the constructed model instance (either
      :class:`KimiLinearModel` or :class:`KimiLinearAttnResModel`).
    * :meth:`update_from_config` — no-op for Kimi Linear: MLA uses
      NoPE (``mla_use_nope=True``) so no RoPE max_seq_len to propagate,
      and KDA is seq-len-agnostic (short conv + recurrent state).
    * :meth:`get_nparams_and_flops` — trainer uses this for MFU
      reporting. Returns (n_params, forward+backward FLOPs per step).

    Deliberately NOT inheriting from ``BaseModel.Config`` at class
    definition to keep the module importable in CPU tests without
    pulling in the ``torchtitan.protocols`` dependency chain. The
    trainer only needs duck-typing on ``build`` /
    ``update_from_config`` / ``get_nparams_and_flops``.
    """

    kimi_config: KimiLinearConfig
    num_blocks: int | None = None
    param_init: dict | None = None  # torchtitan BaseModel.Config contract

    def build(self, **kwargs):
        # Local import to defer the attn_res_model dep chain.
        from torchtitan.experiments.kimi_linear.attn_res_model import (
            KimiLinearAttnResModel,
        )
        if self.num_blocks is None:
            return KimiLinearModel(self.kimi_config)
        return KimiLinearAttnResModel(
            self.kimi_config, num_blocks=self.num_blocks
        )

    def update_from_config(self, *, trainer_config, **kwargs) -> None:
        """No-op: Kimi Linear's NoPE-MLA + KDA are seq-len-agnostic.

        (If a future variant adds RoPE'd MLA, propagate ``training.seq_len``
        into ``self.kimi_config.rope_theta`` or per-layer rope knobs here.)
        """
        return None

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int,
    ) -> tuple[int, int]:
        """(total_n_params, flops_per_TOKEN) for MFU reporting.

        Matches torchtitan's MoE convention in
        ``torchtitan.models.utils.get_moe_model_nparams_and_flops``:

            flops_per_token = 6 * activated_non_embedding
                            + 6 * n_mla_layers * n_heads * head_dims * seq_len

        — a ``6 * W`` constant per-token linear term plus an attention
        term that's linear in seq_len (the O(N²) MLA softmax, counted
        per-token as O(N)). KDA layers have linear-attention FLOPs in
        seq_len, negligible relative to MLA here; we omit them.

        Activated params: dense + shared_expert + router + routed*top_k/num_experts.

        Embedding excluded from the linear term (FLOPs-free lookup).
        """
        nparams_total = 0
        nparams_embedding = 0
        nparams_dense = 0
        nparams_router = 0
        nparams_shared = 0
        nparams_routed = 0
        for name, p in model.named_parameters():
            nparams_total += p.numel()
            if "embed_tokens" in name or "lm_head" in name:
                # lm_head is tied to embeddings in Kimi scaling-law configs,
                # but not always — only exclude embed_tokens.
                if "embed_tokens" in name:
                    nparams_embedding += p.numel()
                # Treat both as dense for non-attention FLOPs; embedding
                # lookup is free, lm_head is a real projection.
                nparams_dense += p.numel()
            elif ".moe.shared_experts" in name:
                nparams_shared += p.numel()
            elif ".moe.router" in name or ".moe.gate" in name:
                nparams_router += p.numel()
            elif ".moe.experts" in name:
                nparams_routed += p.numel()
            else:
                nparams_dense += p.numel()

        cfg = self.kimi_config
        top_k = cfg.num_experts_per_token
        n_experts = cfg.num_experts or 1
        nparams_active_linear = (
            nparams_dense - nparams_embedding
            + nparams_shared + nparams_router
            + nparams_routed * top_k // n_experts
        )

        # MLA attention FLOPs: only full_attn_layers (MLA), KDA has linear
        # attention we approximate as zero in this term.
        n_mla_layers = len(cfg.full_attn_layers) if cfg.full_attn_layers else 0
        head_dims_attn = (
            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim
        )
        attn_flops_per_token = (
            6 * n_mla_layers * cfg.num_attention_heads * head_dims_attn * seq_len
        )

        flops_per_token = 6 * nparams_active_linear + attn_flops_per_token
        return nparams_total, flops_per_token

    def to_dict(self) -> dict:
        """Serialize to a plain dict for logging / checkpoint metadata.

        Trainer calls this on the model_config to pretty-print the
        configuration before building. We flatten the wrapped
        :class:`KimiLinearConfig` dataclass into this dict so the log
        shows the actual Kimi hyperparameters (not just a reference).
        """
        import dataclasses
        out = dataclasses.asdict(self.kimi_config)
        out["__spec__"] = {
            "num_blocks": self.num_blocks,
            "model_class": (
                "KimiLinearAttnResModel" if self.num_blocks is not None
                else "KimiLinearModel"
            ),
        }
        return out

    @property
    def layers(self) -> list[None]:
        """Fake list of length ``num_hidden_layers`` for torchtitan
        pipeline_llm's ``num_layers = len(model_config.layers)`` check.

        Kimi Linear's per-layer config is not a standalone dataclass
        (KDA/MLA/MoE types vary per layer), so we don't expose a real
        list of per-layer Config objects. This property gives
        pipeline_llm the count it needs. Downstream consumers that
        iterate layers should use the built model's ``model.layers``
        (nn.ModuleList) directly.
        """
        return [None] * self.kimi_config.num_hidden_layers

    @property
    def num_hidden_layers(self) -> int:
        """Expose num_hidden_layers at the spec level so adapter code
        (pipeline_adapter._inject_kimi_linear_fqns) can get layer count
        without reaching into kimi_config.
        """
        return self.kimi_config.num_hidden_layers
