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
``attn_res_model.py``, matching the ``AttnResLlama3Model`` pattern
this experiment grew out of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.models.common.linear import Linear as _TTLinear


class Linear(_TTLinear):
    """Module-protocol-compliant Linear with ``nn.Linear``-style constructor.

    Inherits from ``torchtitan.models.common.linear.Linear`` (= ``nn.Linear
    + Module``) so instances satisfy
    ``Float8LinearConverter.verify_module_protocol`` (see
    torchtitan/components/quantization/float8.py:185, which checks for
    exactly that ``Linear`` class). Overrides ``__init__`` to accept
    ``nn.Linear``-style positional args instead of the parent's
    ``Config``-based constructor, avoiding a global rewrite of all 18
    call sites in this experiment.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias=bias)

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
    # Gated MLA (K3 delta): sigmoid output gate, near-identity init so a
    # non-gated-MLA-pretrained checkpoint's function is ~preserved at
    # step 0 (graft-viable: a near-identity gate init keeps the
    # pretrained function intact). PROVISIONAL: exact gate form
    # reconciles at 7.27. Off by default (plain MLA = validated path).
    mla_gated: bool = False
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
    # Wired by KimiLinearSpec.update_from_config from config.parallelism
    # BEFORE build; consumed by KimiMoE to populate the upstream
    # module-internal MoE sharding configs (EP/TP). False = the
    # previously validated FSDP/PP plain path, untouched.
    moe_enable_ep: bool = False
    moe_enable_tp: bool = False
    topk_group: int = 1

    # ---- norm / act ----
    rms_norm_eps: float = 1e-5
    hidden_act: Literal["silu", "gelu", "situ"] = "silu"
    # SiTU (Sigmoid Tanh Unit), K3's activation. Official config.json ships
    # activation_situ_beta=4.0 and activation_situ_linear_beta=25.0; both are
    # only read when hidden_act == "situ".
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float | None = 25.0

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


# ----- SiTU activation ---------------------------------------------------- #


def situ_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """K3's Sigmoid Tanh Unit, gated form (reference: SituAndMul).

    ``situ(g) = beta * tanh(g / beta) * sigmoid(g)`` -- a soft-clipped SiLU:
    the tanh caps the magnitude at +/- beta while sigmoid keeps the SiLU-like
    gating shape near 0. When ``linear_beta`` is set the linear branch is
    clipped the same way before the product. Computed in fp32 and cast back,
    as the reference does, because the product of two saturating nonlinearities
    is sensitive to bf16 rounding near the caps.
    """
    g = gate.float()
    u = up.float()
    out = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (out * u).to(gate.dtype)


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
        hidden_act: Literal["silu", "gelu", "situ"] = "silu",
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
    ) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.hidden_act = hidden_act
        self._situ_beta = situ_beta
        self._situ_linear_beta = situ_linear_beta
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        elif hidden_act == "situ":
            # SiTU is gated over BOTH branches, so there is no elementwise
            # act_fn to apply to the gate alone; forward dispatches instead.
            self.act_fn = None
        else:
            raise ValueError(f"Unknown hidden_act: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.hidden_act == "situ":
            return self.down_proj(
                situ_and_mul(
                    gate, up, self._situ_beta, self._situ_linear_beta
                )
            )
        return self.down_proj(self.act_fn(gate) * up)


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


def _cp_all_to_all_headseq(
    x: torch.Tensor, cp_group, seq_to_head: bool
) -> torch.Tensor:
    """Differentiable Ulysses all-to-all swapping which of (seq, head) is
    sharded across the CP group.

    seq_to_head=True:  ``[B, T/cp, H, K]`` (seq-sharded) -> ``[B, T, H/cp, K]``.
    seq_to_head=False: ``[B, T, H/cp, K]`` -> ``[B, T/cp, H, K]``.

    Numerics (round-trip and per-head chunk_kda parity) validated
    bit-exact against a single-rank reference; backward is the
    transposed all-to-all via torch.distributed.nn.functional.
    """
    import torch.distributed.nn.functional as dist_nn

    cp = dist.get_world_size(cp_group)
    B, d1, d2, K = x.shape
    if seq_to_head:
        t_loc, num_heads = d1, d2
        # [B, T/cp, H, K] -> [cp, B, T/cp, H/cp, K] (split heads by dest)
        x_split = (
            x.reshape(B, t_loc, cp, num_heads // cp, K)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        out = dist_nn.all_to_all_single(
            torch.empty_like(x_split), x_split, group=cp_group
        )
        # recv[s] holds src s's T/cp for THIS rank's head subset -> stack seq
        return (
            out.permute(1, 0, 2, 3, 4)
            .reshape(B, cp * t_loc, num_heads // cp, K)
            .contiguous()
        )
    t_full, h_loc = d1, d2
    t_loc = t_full // cp
    x_split = (
        x.reshape(B, cp, t_loc, h_loc, K).permute(1, 0, 2, 3, 4).contiguous()
    )
    out = dist_nn.all_to_all_single(
        torch.empty_like(x_split), x_split, group=cp_group
    )
    # out[s] = src s's head subset for THIS rank's seq shard; put T/cp
    # before the src(cp) axis so reshape stacks heads in ascending order.
    return (
        out.permute(1, 2, 0, 3, 4)
        .reshape(B, t_loc, cp * h_loc, K)
        .contiguous()
    )


class KimiMLAAttention(nn.Module):
    """Multi-head Latent Attention, Kimi NoPE variant.

    Faithful port of ``reference:KimiMLAAttention``. Key differences
    vs. DSv3 MLA:

    * ``q_lora_rank`` — when None, Q is projected directly to
      ``num_heads x q_head_dim`` (the 48B-A3B path). When set (K3 ships
      1536) Q goes through the compression pair
      ``q_a_proj -> q_a_layernorm -> q_b_proj``, mirroring DSv3.
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

        assert self.use_nope, (
            "Only mla_use_nope=True is currently supported (Kimi 48B-A3B "
            "config). RoPE-on-MLA is not ported."
        )

        if self.q_lora_rank is None:
            # 48B-A3B path: Q straight to H * q_head_dim.
            self.q_proj = Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            # K3 path (official config: q_lora_rank=1536). Same shape as
            # DSv3's wq_a/wq_b pair, and the same structure this class
            # already uses for KV (kv_a_proj_with_mqa -> kv_a_layernorm ->
            # kv_b_proj), so the TP plan reuses that registration pattern.
            self.q_a_proj = Linear(
                self.hidden_size, self.q_lora_rank, bias=False
            )
            self.q_a_layernorm = nn.RMSNorm(
                self.q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # Gated MLA (K3 delta, provisional): per-head sigmoid gate on the
        # attention output before o_proj. Near-identity init: the gate
        # projection is zero-init and a +LARGE bias makes sigmoid(.)~=1,
        # so at step 0 gated_out ~= plain attn_out (graft-preserving).
        self.mla_gated = config.mla_gated
        if self.mla_gated:
            self.attn_gate_proj = Linear(
                self.hidden_size, self.num_heads, bias=True
            )

        # SDPA-only sub-module so the TP plan can wrap it with
        # use_local_output=True (DSv3 pattern). Has no parameters.
        self.inner_attention = KimiMLAInnerAttention()

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Q projection, with or without the compression pair.

        Returns the flat ``[..., num_heads * q_head_dim]`` tensor; callers
        reshape. Kept as one method so the direct and CP forward paths cannot
        drift apart.
        """
        if self.q_lora_rank is None:
            return self.q_proj(x)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal mask; no KV cache.

        Args:
            x: ``[B, T, D]`` hidden states.
        Returns:
            ``[B, T, D]`` attention output.
        """
        # Context parallel: Ulysses path (seq-local projections,
        # all-to-all seq<->head, full-seq SDPA on this rank's head
        # subset). Handles both plain x and DTensor x (TP), so there is
        # no silent CP skip under TP anymore.
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            return self._forward_cp(x, cp_group)
        B, T, _ = x.shape

        # Q path: direct projection -> (B, T, H, q_head_dim) -> (B, H, T, q_head_dim)
        q = (
            self._project_q(x)
            .view(B, T, self.num_heads, self.q_head_dim)
            .transpose(1, 2)
        )
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

        attn_out = attn_out.transpose(1, 2)  # (B, T, H, Dv)
        if self.mla_gated:
            # Per-head sigmoid gate from x; near-identity at init.
            gate = torch.sigmoid(self.attn_gate_proj(x))  # (B, T, H)
            attn_out = attn_out * gate.unsqueeze(-1)
        attn_out = attn_out.reshape(B, T, -1)
        out = self.o_proj(attn_out)
        return out

    def _forward_cp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """Ulysses CP forward.

        Tensor-name legend (shape suffixes): B batch, L local seq (T/cp),
        T full seq, H local head count before CP split (num_heads/tp),
        G CP-local head count (H/cp), Q q_head_dim, N qk_nope_head_dim,
        V v_head_dim, R qk_rope_head_dim, W concatenated feature dim.

        Input x is ``[B, L, D]`` -- plain, or DTensor(Replicate on
        tp_mesh) under TP. Projections run through their (possibly
        TP-wrapped) modules at seq length L; the CP collectives operate
        on plain local tensors only, in the same gap where the TP plan
        already strips DTensor (inner_attention use_local_output). Under
        TP the head axis is already tp-sharded, so this rank computes
        num_heads/(tp*cp) heads over the full sequence. No rank ever
        materializes ``[B, T, D]`` hidden states: activation memory
        follows the Ulysses contract, unlike the previous all-gather-SP
        path which kept O(T x D) per rank at any cp degree.
        """
        import torch.distributed.nn.functional as dist_nn

        cp_size = dist.get_world_size(cp_group)
        B, t_loc, _ = x.shape

        q_BLE = self._project_q(x)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot_BLR = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_BLF = self.kv_b_proj(self.kv_a_layernorm(k_pass))

        # Leave DTensor land (no-ops when TP is off). All CP collectives
        # below run on plain local tensors on the cp sub-mesh group.
        q_BLE = _to_local_if_dtensor(q_BLE)
        kv_BLF = _to_local_if_dtensor(kv_BLF)
        k_rot_BLR = _to_local_if_dtensor(k_rot_BLR)

        kv_head_dim = self.qk_nope_head_dim + self.v_head_dim
        h_loc = q_BLE.shape[-1] // self.q_head_dim
        if h_loc % cp_size != 0:
            raise ValueError(
                f"MLA CP: local head count {h_loc} is not divisible by "
                f"cp={cp_size} (num_attention_heads must divide tp*cp)"
            )

        # One fused all-to-all for q and kv (concat on the feature axis).
        qkv_BLHW = torch.cat(
            [
                q_BLE.view(B, t_loc, h_loc, self.q_head_dim),
                kv_BLF.view(B, t_loc, h_loc, kv_head_dim),
            ],
            dim=-1,
        )
        qkv_BTGW = _cp_all_to_all_headseq(qkv_BLHW, cp_group, seq_to_head=True)
        q_BTGQ, k_pass_BTGN, v_BTGV = torch.split(
            qkv_BTGW,
            [self.q_head_dim, self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        t_full = t_loc * cp_size
        h_cp = h_loc // cp_size

        # k_rot is broadcast across heads (headless): all-gather the seq
        # shards (differentiable -> reduce-scatter backward) and expand
        # onto this rank's head subset. Tiny tensor (R per token).
        k_rot_BTR = torch.cat(
            dist_nn.all_gather(k_rot_BLR.contiguous(), group=cp_group), dim=1
        )
        k_BTGQ = torch.cat(
            [
                k_pass_BTGN,
                k_rot_BTR.view(B, t_full, 1, self.qk_rope_head_dim).expand(
                    B, t_full, h_cp, self.qk_rope_head_dim
                ),
            ],
            dim=-1,
        )

        attn_BGTV = self.inner_attention(
            q_BTGQ.transpose(1, 2),
            k_BTGQ.transpose(1, 2),
            v_BTGV.transpose(1, 2),
            scale=self.scaling,
        )
        attn_BLHV = _cp_all_to_all_headseq(
            attn_BGTV.transpose(1, 2).contiguous(), cp_group, seq_to_head=False
        )
        if self.mla_gated:
            # Per-head sigmoid gate from the seq-local x; pointwise per
            # (b, t, h), so it applies after the heads return seq-local.
            gate_BLH = torch.sigmoid(
                _to_local_if_dtensor(self.attn_gate_proj(x))
            )
            attn_BLHV = attn_BLHV * gate_BLH.unsqueeze(-1)
        out = self.o_proj(
            attn_BLHV.reshape(B, t_loc, h_loc * self.v_head_dim)
        )
        return out


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

        self.q_proj = Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = Linear(self.hidden_size, projection_size, bias=False)

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
        self.f_a_proj = Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = Linear(self.head_dim, projection_size, bias=False)
        self.g_a_proj = Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = Linear(self.head_dim, projection_size, bias=False)

        # Beta: per-head, per-token scalar (delta-rule learning rate)
        self.b_proj = Linear(self.hidden_size, self.num_heads, bias=False)

        # Output RMSNorm with sigmoid-gated modulation from g, then o_proj
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid",
        )
        self.o_proj = Linear(projection_size, self.hidden_size, bias=False)

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
        # Context parallel: Ulysses path (seq-local projections,
        # all-to-all seq<->head, full-seq conv + scan on this rank's head
        # subset). chunk_kda is bit-exactly per-head independent
        # (kda_ulysses_cp_probe), so head-sharding the scan is exact.
        # MLA layers get the same treatment in KimiMLAAttention.
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            out = self._forward_cp(x, cp_group)
            if in_mesh is not None and in_placements is not None:
                out = DTensor.from_local(
                    out, in_mesh, in_placements, run_check=False,
                )
            return out
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

        # 6) Output gate (computed before the head-shard so the slice below
        # covers it too).
        g_out = _local_linear(self.g_b_proj, _local_linear(self.g_a_proj, x))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)

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

    def _forward_cp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """Ulysses CP forward for KDA.

        Tensor-name legend (shape suffixes): B batch, L local seq (T/cp),
        T full seq, H head count (KDA is never tp-sharded), G CP-local
        head count (H/cp), K head_dim, C flattened head-subset channels
        (G*K).

        Input x is the plain local ``[B, L, D]`` shard (caller already
        stripped DTensor). Projections run seq-local at L; one fused
        all-to-all moves (q, k, v, g_raw, g_out, beta) to full-seq
        head-subset layout; the causal short conv, fused_kda_gate, and
        chunk_kda then run on the full sequence for this rank's G heads
        (conv weights channel-sliced -- depthwise conv, exact; validated
        bit-exact vs ShortConvolution). No rank materializes the full
        sequence at hidden dim D.

        Gradient note: each rank's param-grad contribution covers its
        (seq shard x head subset) sector with zeros elsewhere; FSDP's
        dp_shard_cp mesh reduces over cp, reconstructing full grads --
        the same contract the previous all-gather-SP path relied on.
        """
        from fla.modules.conv.causal_conv1d import causal_conv1d

        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        B, t_loc, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        if num_heads % cp_size != 0:
            raise ValueError(
                f"KDA CP: num_heads {num_heads} is not divisible by "
                f"cp={cp_size}"
            )
        h_cp = num_heads // cp_size
        h0 = cp_rank * h_cp

        # 1) Seq-local projections at L (no cross-seq ops here).
        q_BLHK = _local_linear(self.q_proj, x).view(
            B, t_loc, num_heads, head_dim
        )
        k_BLHK = _local_linear(self.k_proj, x).view(
            B, t_loc, num_heads, head_dim
        )
        v_BLHK = _local_linear(self.v_proj, x).view(
            B, t_loc, num_heads, head_dim
        )
        g_raw_BLHK = _local_linear(
            self.f_b_proj, _local_linear(self.f_a_proj, x)
        ).view(B, t_loc, num_heads, head_dim)
        g_out_BLHK = _local_linear(
            self.g_b_proj, _local_linear(self.g_a_proj, x)
        ).view(B, t_loc, num_heads, head_dim)
        beta_BLH1 = _local_linear(self.b_proj, x).unsqueeze(-1)

        # 2) One fused all-to-all: seq-shard -> full-seq head-subset.
        packed_BLHW = torch.cat(
            [q_BLHK, k_BLHK, v_BLHK, g_raw_BLHK, g_out_BLHK, beta_BLH1],
            dim=-1,
        )
        packed_BTGW = _cp_all_to_all_headseq(
            packed_BLHW, cp_group, seq_to_head=True
        )
        q_BTGK, k_BTGK, v_BTGK, g_raw_BTGK, g_out_BTGK, beta_BTG1 = (
            torch.split(
                packed_BTGW,
                [head_dim, head_dim, head_dim, head_dim, head_dim, 1],
                dim=-1,
            )
        )
        t_full = t_loc * cp_size

        mode = "fused_recurrent" if t_full <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 3) Short causal conv on the full sequence, weights sliced to
        # this rank's head-subset channels (depthwise conv -> exact).
        def conv_subset(conv: ShortConvolution, x_BTGK: torch.Tensor):
            w_CW = _to_local_if_dtensor(conv.weight).squeeze(1)[
                h0 * head_dim : (h0 + h_cp) * head_dim
            ]
            b_C = (
                _to_local_if_dtensor(conv.bias)[
                    h0 * head_dim : (h0 + h_cp) * head_dim
                ]
                if conv.bias is not None
                else None
            )
            y_BTC, _ = causal_conv1d(
                x_BTGK.reshape(B, t_full, h_cp * head_dim),
                weight=w_CW,
                bias=b_C,
                activation=conv.activation,
                backend=conv.backend,
            )
            return y_BTC.view(B, t_full, h_cp, head_dim)

        q_BTGK = conv_subset(self.q_conv1d, q_BTGK)
        k_BTGK = conv_subset(self.k_conv1d, k_BTGK)
        v_BTGK = conv_subset(self.v_conv1d, v_BTGK)

        # 4) Forget gate + beta on the head subset (A_log/dt_bias sliced).
        g_BTGK = fused_kda_gate(
            g_raw_BTGK,
            _to_local_if_dtensor(self.A_log)[h0 : h0 + h_cp],
            dt_bias=_to_local_if_dtensor(self.dt_bias)
            .view(num_heads, head_dim)[h0 : h0 + h_cp]
            .reshape(-1),
        )
        beta_BTG = beta_BTG1.squeeze(-1).float().sigmoid()

        # 5) KDA scan on this rank's heads over the full sequence.
        kda_fn = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o_BTGK, _ = kda_fn(
            q=q_BTGK,
            k=k_BTGK,
            v=v_BTGK,
            g=g_BTGK,
            beta=beta_BTG,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )
        o_BTGK = self.o_norm(o_BTGK, g_out_BTGK)

        # 6) All-to-all back to seq-shard full-head layout, then o_proj.
        o_BLHK = _cp_all_to_all_headseq(o_BTGK, cp_group, seq_to_head=False)
        out = _local_linear(
            self.o_proj, o_BLHK.reshape(B, t_loc, num_heads * head_dim)
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
        from torchtitan.models.common.config_utils import (
            make_token_dispatcher_config,
        )
        from torchtitan.models.common.moe import (
            GroupedExperts,
            MoE,
            RoutedExperts,
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

        # TODO(kimi-parity): upstream removed score_before_experts; Kimi's
        # reference applies router scores BEFORE the experts. Verify the
        # fixed upstream ordering against the official 48B ckpt (the
        # SGLang-side A/B from PR15 is the harness) before training.
        moe_cfg = MoE.Config(
            num_experts=config.num_experts,
            routed_experts=RoutedExperts.Config(
                inner_experts=experts_cfg,
                token_dispatcher=make_token_dispatcher_config(
                    num_experts=config.num_experts,
                    top_k=config.num_experts_per_token,
                    comm_backend="standard",
                    hidden_dim=config.hidden_size,
                ),
            ),
            router=router_cfg,
            load_balance_coeff=1e-3,
            shared_experts=shared_cfg,
        )
        if config.moe_enable_ep or config.moe_enable_tp:
            # Upstream (post-merge) parallelizes MoE module-internally:
            # sharding configs are declared on the Config BEFORE build,
            # then _moe.parallelize(parallel_dims) distributes states and
            # wires the token dispatcher (see parallelize.py). Same
            # expert-param TP layout as deepseek_v3.
            import spmd_types as spmd

            from torchtitan.models.common.moe_sharding import (
                set_moe_sharding_config,
            )

            set_moe_sharding_config(
                moe_cfg,
                enable_ep=config.moe_enable_ep,
                enable_sp=False,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )
        self._moe = moe_cfg.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._moe(x)
        if isinstance(out, DTensor):
            # Module-internal MoE parallelization (EP/TP) emits DTensor
            # (Partial on the TP axis before reduction). This model's
            # boundary convention is plain tensors (PP P2P, AttnRes
            # stacking, fla kernels) -- reduce to Replicate (the required
            # TP all-reduce) and unwrap. to_local's default grad
            # placement is the tensor's own Replicate.
            if any(not p.is_replicate() for p in out.placements):
                out = out.redistribute(
                    placements=[Replicate()] * len(out.placements)
                )
            out = out.to_local()
        return out


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
    are defined here so the cross-stage cache adapter can toggle
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
        self.lm_head = Linear(
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
                if "weight" not in m._parameters:
                    # Packed-MXFP4 LoRA base: quantize_base_mxfp4 dropped
                    # base.weight (split qdata/scale storage); the packed
                    # values come from the checkpoint, not init.
                    continue
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif cls_name in (
                "ShortConvolution", "FusedRMSNormGated", "KimiLoRALinear",
            ):
                # fla-core modules + the LoRA wrapper ship reset_parameters()
                # (LoRA: kaiming lora_a, zero lora_b -- the generic Linear
                # pass above only covers their nn.Linear children).
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
            # Gated MLA near-identity init: zero the gate projection
            # weight and set a large positive bias so sigmoid(gate)~=1
            # at step 0 (gated_out ~= plain attn_out; graft-preserving).
            gate_proj = getattr(attn, "attn_gate_proj", None)
            if gate_proj is not None:
                nn.init.zeros_(gate_proj.weight)
                nn.init.constant_(gate_proj.bias, 6.0)  # sigmoid(6)=0.9975

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
    # Graft gate: alpha-gated AttnRes reads (alpha=0 == exact identity
    # with the plain backbone at step 0). For grafting onto pretrained
    # weights; from-scratch flavors keep the paper's ungated read.
    attn_res_gated: bool = False
    # LoRA (module-level; see lora.py). rank=None disables. When set,
    # target projections are wrapped (lora_b zero-init -> step-0
    # identity) and the base freezes EXCEPT the AttnRes graft params
    # (alpha-fullparam exception).
    lora_rank: int | None = None
    lora_alpha: float = 16.0
    lora_quantize_base: str | None = None  # 'nf4' => QLoRA

    def build(self, **kwargs):
        # Local import to defer the attn_res_model dep chain.
        from torchtitan.experiments.kimi_k3.attn_res_model import (
            KimiLinearAttnResModel,
        )
        if self.num_blocks is None:
            model = KimiLinearModel(self.kimi_config)
        else:
            model = KimiLinearAttnResModel(
                self.kimi_config,
                num_blocks=self.num_blocks,
                gated=self.attn_res_gated,
            )
        if self.lora_rank is not None:
            from torchtitan.experiments.kimi_k3.lora import apply_lora

            apply_lora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                quantize_base=self.lora_quantize_base,
            )
        return model

    def update_from_config(self, *, config, **kwargs) -> None:
        """Wire parallelism knobs the model must know BEFORE build.

        Signature matches ``BaseModel.Config.update_from_config``
        (keyword ``config`` = the Trainer.Config).

        MoE EP/TP: upstream parallelizes MoE module-internally via
        sharding configs declared at config-build time; KimiMoE reads
        these flags when constructing its MoE.Config. Seq-len needs no
        propagation (NoPE-MLA + KDA are seq-len-agnostic).
        """
        parallelism = getattr(config, "parallelism", None)
        if parallelism is not None:
            self.kimi_config.moe_enable_ep = (
                parallelism.expert_parallel_degree > 1
            )
            self.kimi_config.moe_enable_tp = (
                parallelism.tensor_parallel_degree > 1
            )
        return None

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int,
    ) -> tuple[int, int]:
        """(total_n_params, flops_per_TOKEN) for MFU reporting.

        Follows torchtitan's MoE convention in
        ``torchtitan.models.utils.get_moe_model_nparams_and_flops``
        (6x = fwd 2x + bwd 4x), extended for this architecture:

            flops_per_token = 6 * activated_non_embedding          (linear)
                            + 6 * n_mla * n_heads * head_dims * seq (MLA)
                            + 12 * n_kda * kda_heads * kda_dim^2    (KDA)
                            + 6 * (2*n_layers + 1) * (N+1) * hidden (AttnRes)

        * MLA: O(seq) per token (softmax attention counted per-token).
        * KDA: linear attention -- the per-head [kda_head_dim x
          kda_head_dim] recurrent state is written (delta-rule update)
          and read (output) once per token, seq-len INDEPENDENT; the 2
          state touches give the 12x (= 6 * 2) factor. Projections are
          already inside the 6*W linear term.
        * AttnRes (only when ``num_blocks`` is set): each sub-layer read
          mixes up to N block sources + the partial block per token
          (softmax over sources + weighted sum over hidden), twice per
          layer (attn + mlp reads) plus the final read.

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

        # MLA attention FLOPs: only full_attn_layers (softmax, O(seq)/token).
        n_mla_layers = len(cfg.full_attn_layers) if cfg.full_attn_layers else 0
        head_dims_attn = (
            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim
        )
        attn_flops_per_token = (
            6 * n_mla_layers * cfg.num_attention_heads * head_dims_attn * seq_len
        )

        # KDA linear-attention state ops: per token each head writes and
        # reads its [kda_head_dim x kda_head_dim] recurrent state once.
        n_kda_layers = (
            len(cfg.kda_layers)
            if cfg.kda_layers
            else cfg.num_hidden_layers - n_mla_layers
        )
        kda_flops_per_token = (
            12 * n_kda_layers * cfg.kda_num_heads * cfg.kda_head_dim**2
        )

        # AttnRes source mixing: 2 reads per layer + the final read, each
        # mixing up to (num_blocks + 1) sources over hidden_size.
        if self.num_blocks is not None:
            attn_res_flops_per_token = (
                6
                * (2 * cfg.num_hidden_layers + 1)
                * (self.num_blocks + 1)
                * cfg.hidden_size
            )
        else:
            attn_res_flops_per_token = 0

        flops_per_token = (
            6 * nparams_active_linear
            + attn_flops_per_token
            + kda_flops_per_token
            + attn_res_flops_per_token
        )
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

    @property
    def dim(self) -> int:
        """veRL's torchtitan engine matches flavors by (dim, n_layers,
        vocab_size); expose the Decoder.Config-style names."""
        return self.kimi_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.kimi_config.vocab_size

    def traverse(self, config_cls, *, recurse: bool = False, _prefix: str = ""):
        """Config-tree leaf: yield nothing.

        The Kimi Linear model is built as plain modules from
        :class:`KimiLinearConfig`, not from a ``Configurable.Config``
        tree, so there are no nested component configs to expose.
        Implemented because the Trainer chain requires it on every
        model config (``has_quantization``, the override mechanism via
        ``ModelSpec.traverse``).
        """
        return iter(())


@dataclass(kw_only=True, slots=True)
class KimiLinearFloat8Spec(KimiLinearSpec):
    """:class:`KimiLinearSpec` whose ``build()`` swaps eligible
    ``nn.Linear`` modules to torchao ``Float8Linear``.

    The Kimi Linear model is constructed as plain modules, not from a
    ``Linear.Config`` tree, so ``Float8LinearConverter.convert``'s
    config traversal cannot apply here. Instead the swap happens
    module-level right after construction (on the meta device, before
    parallelize/init), mirroring the converter's ``module_filter_fn``
    semantics: all dims divisible by 16, filtered FQNs skipped.
    Additionally every Linear inside a :class:`KimiDeltaAttention` is
    skipped structurally -- KDA and MLA layers share the ``self_attn``
    attribute name, so no FQN substring can express "KDA only".
    ``init_weights`` still covers swapped modules because torchao's
    ``Float8Linear`` subclasses ``nn.Linear``.
    """

    torchao_float8_config: object = None
    filter_fqns: list[str] = field(default_factory=list)

    def build(self, **kwargs):
        from torchao.float8 import convert_to_float8_training

        # Explicit base call: zero-arg super() breaks under
        # @dataclass(slots=True), which recreates the class object.
        model = KimiLinearSpec.build(self, **kwargs)

        kda_linear_fqns = {
            f"{name}.{sub_name}"
            for name, m in model.named_modules()
            if isinstance(m, KimiDeltaAttention)
            for sub_name, sub in m.named_modules()
            if sub_name and isinstance(sub, nn.Linear)
        }

        def _filter(mod: nn.Module, fqn: str) -> bool:
            return (
                mod.in_features % 16 == 0
                and mod.out_features % 16 == 0
                and fqn not in kda_linear_fqns
                and not any(f in fqn for f in self.filter_fqns)
            )

        return convert_to_float8_training(
            model,
            config=self.torchao_float8_config,
            module_filter_fn=_filter,
        )

    def traverse(self, config_cls, *, recurse: bool = False, _prefix: str = ""):
        """Yield a single synthetic Float8Linear.Config marker.

        The Float8 swap here is module-level (``build()``), so there is
        no real config tree to report. Config-tree consumers -- today
        only ``has_quantization``, which gates the misleading-under-fp8
        MFU metric -- still need to see that quantization is active.
        The marker's dims are placeholders (16x16, the fp8 alignment
        unit); treat it strictly as a boolean signal, never as a real
        layer description.
        """
        from torchtitan.components.quantization.float8 import Float8Linear

        if (
            self.torchao_float8_config is not None
            and Float8Linear is not None
            and issubclass(Float8Linear.Config, config_cls)
        ):
            fqn = (
                f"{_prefix}.module_level_float8_swap"
                if _prefix
                else "module_level_float8_swap"
            )
            marker = Float8Linear.Config(
                in_features=16,
                out_features=16,
                _torchao_config=self.torchao_float8_config,
            )
            yield fqn, marker, None, None
        else:
            yield from ()
