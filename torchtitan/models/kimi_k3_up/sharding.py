# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative tensor-parallel sharding for Kimi K3, in the per-model idiom.

Every other model in the tree carries one of these -- llama3 (74 lines), qwen3
(111), gpt_oss (127), kimi_k2_7 (145), deepseek_v3 (192), qwen3_5 (343) -- and
their ``parallelize.py`` holds no imperative ``ColwiseParallel`` /
``RowwiseParallel`` entries at all. The declarations are set on the CONFIG tree
before ``build()``, and ``Module.parallelize()`` applies them at runtime against
whatever mesh is actually enabled, so a file like this can be populated
unconditionally.

The MLA half is the same shape as ``deepseek_v3/sharding.py``, which is the
canonical precedent: low-rank compressions and their norms stay Replicate,
expansions are colwise, and the output projection is rowwise. Those placements
were also measured independently on the AttnRes tree across four full
three-arm matrix runs and agree with deepseek_v3 one for one.

The KDA half has no upstream precedent. All eight of its projections are
Replicate, which is what NoParallel gave them: their outputs feed fla's kernels,
and KDA unwraps at the call site, so the kernels see plain tensors either way and
a head-sharded weight would only produce a shape the kernel cannot use.

Sequence parallel is threaded through as ``enable_sp`` the way the other models
do it, so the norms and the activation layouts follow TP without being tied to
it.

EP is deliberately absent here; see the note at the bottom of the file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.module import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3_up.model import (
        KimiDeltaAttention,
        KimiK3Model,
        KimiK3TransformerBlock,
        KimiLatentMoE,
        KimiMLAAttention,
    )


def _replicate_weight() -> ShardingConfig:
    """A weight that stays Replicate on the TP axis.

    Still distributed rather than left plain: a plain weight against a DTensor
    activation mixes the two kinds inside the matmul, which is the failure this
    costs a debugging session to find. deepseek_v3's sharding.py says the same
    thing about its low-rank projections.
    """
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on every Kimi K3 sub-config."""
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    # The model-level attention-residual tail. The projection reduces D to a
    # single score (a pseudo-query), so there is no axis worth sharding.
    config.output_res_norm.sharding_config = norm_config(enable_sp=False)
    config.output_res_proj.sharding_config = _replicate_weight()
    for layer_cfg in config.layers:
        _set_kimi_k3_layer_sharding(layer_cfg, enable_sp=enable_sp)
    if config.vision_encoder is not None:
        _set_vision_encoder_sharding(config.vision_encoder)


def _set_kimi_k3_layer_sharding(
    layer_cfg: "KimiK3TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """One hybrid block: attention XOR delta_attention, FFN XOR MoE."""
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    # The two per-layer attention-residual reads. Their norms sit on values that
    # are already gathered, and their projections produce one score per token, so
    # both stay Replicate rather than following the SP layout of the stream.
    layer_cfg.attention_res_norm.sharding_config = norm_config(enable_sp=False)
    layer_cfg.attention_res_proj.sharding_config = _replicate_weight()
    layer_cfg.ffn_res_norm.sharding_config = norm_config(enable_sp=False)
    layer_cfg.ffn_res_proj.sharding_config = _replicate_weight()

    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )

    if layer_cfg.attention is not None:
        _set_mla_sharding(layer_cfg.attention, attn_x_layout, enable_sp=enable_sp)
    else:
        assert layer_cfg.delta_attention is not None
        _set_kda_sharding(layer_cfg.delta_attention, attn_x_layout)

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )
    else:
        assert layer_cfg.moe is not None
        _set_latent_moe_sharding(layer_cfg.moe, attn_x_layout)


def _set_mla_sharding(
    attention: "KimiMLAAttention.Config",
    attn_x_layout,
    *,
    enable_sp: bool,
) -> None:
    """K3's Gated MLA. Same shape as deepseek_v3's, plus the output gate."""
    attention.sharding_config = ShardingConfig(
        in_src_shardings={"x_BLD": attn_x_layout},
        in_dst_shardings={"x_BLD": dense_activation_placement(tp=spmd.R)},
    )
    # Compressions and their norms stay Replicate: the output axis is the lora
    # rank, not a head axis, so sharding it would split a dimension the
    # expansion has to see whole.
    attention.wq_a.sharding_config = _replicate_weight()
    attention.q_norm.sharding_config = _replicate_weight()
    attention.wkv_a.sharding_config = _replicate_weight()
    attention.kv_norm.sharding_config = _replicate_weight()
    # Expansions are per-head, so they are colwise.
    attention.wq_b.sharding_config = colwise_config()
    attention.wkv_b.sharding_config = colwise_config()
    # K3's output gate is per-head and full rank (report Eq. 6), so it shards
    # with the heads it gates.
    attention.gate.sharding_config = colwise_config()
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)


def _set_kda_sharding(
    delta_attention: "KimiDeltaAttention.Config",
    attn_x_layout,
) -> None:
    """KDA. Every projection Replicate, because fla's kernels are the consumer.

    The kernels do not dispatch through DTensor and the module unwraps at the
    call site, so head-sharding any of these produces a local shape the kernel
    cannot consume while gaining nothing. The convolutions and the gated norm
    are Replicate for the same reason.
    """
    delta_attention.sharding_config = ShardingConfig(
        in_src_shardings={"x_BLD": attn_x_layout},
        in_dst_shardings={"x_BLD": dense_activation_placement(tp=spmd.R)},
    )
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "forget_a",
        "forget_b",
        "beta",
        "output_gate",
        "output_proj",
    ):
        getattr(delta_attention, name).sharding_config = _replicate_weight()
    for name in ("q_conv", "k_conv", "v_conv"):
        getattr(delta_attention, name).sharding_config = _replicate_weight()
    # Their KimiRMSNormGated is a torchtitan Module in plain PyTorch, so unlike
    # fla's FusedRMSNormGated it can hold a declaration at all.
    delta_attention.output_norm.sharding_config = _replicate_weight()


def _set_latent_moe_sharding(
    moe: "KimiLatentMoE.Config",
    attn_x_layout,
) -> None:
    """The latent MoE's TP-only declarations.

    The latent projections are Replicate, not colwise as their shape suggests:
    ``routed_down``'s output feeds the experts, whose entry contract is
    Replicate, and ``routed_up`` consumes what they return. Measured -- a
    colwise ``routed_down`` fails inside the expert path, and this is the rule
    that took two attempts to learn: a declaration has to match the kind
    contracts on either side of the module, not the shape of the module.
    """
    moe.sharding_config = ShardingConfig(
        in_src_shardings={"x_BLD": attn_x_layout},
        in_dst_shardings={"x_BLD": dense_activation_placement(tp=spmd.R)},
    )
    moe.routed_down.sharding_config = _replicate_weight()
    moe.routed_norm.sharding_config = norm_config(enable_sp=False)
    moe.routed_up.sharding_config = _replicate_weight()
    # The router's gate is core's module and core's moe_sharding owns its
    # declaration; it is not set here.
    if moe.shared_experts is not None:
        set_dense_ffn_sharding(
            moe.shared_experts,
            attn_x_layout=attn_x_layout,
            enable_sp=False,
        )


def _set_vision_encoder_sharding(encoder: "KimiK3VisionEncoder.Config") -> None:
    """The vision tower's TP declarations.

    Only the block MLP is sharded, which is what the AttnRes tree validated:
    ``fc1`` is dim -> intermediate and ``fc2`` back, with an elementwise GELU
    between them, so colwise/rowwise is exact and the activation shard commutes
    with the nonlinearity. Everything else replicates.

    The attention is left replicated here even though this tower COULD shard it:
    unlike the AttnRes tree's fused ``wqkv``, this one has separate wq/wk/wv, so
    head-sharding is expressible as four declarations rather than a Linear split.
    That is new capability rather than a port of something measured, and the
    vision tower is where the DTensor boundaries are most delicate, so it belongs
    in its own change with its own matrix run. Report sec 5.2.3 asks for a
    genuinely parallel tower, so it is worth doing next.
    """
    replicate = _replicate_weight()
    encoder.patch_embed_proj.sharding_config = replicate
    encoder.final_norm.sharding_config = norm_config(enable_sp=False)
    encoder.block.norm1.sharding_config = norm_config(enable_sp=False)
    encoder.block.norm2.sharding_config = norm_config(enable_sp=False)
    for name in ("wq", "wk", "wv", "proj"):
        getattr(encoder.block.attn, name).sharding_config = replicate
    encoder.block.mlp.fc1.sharding_config = colwise_config()
    encoder.block.mlp.fc2.sharding_config = rowwise_config(output_sp=False)
    encoder.projector.linear_1.sharding_config = replicate
    encoder.projector.linear_2.sharding_config = replicate
    encoder.projector.post_norm.sharding_config = norm_config(enable_sp=False)


# Expert parallel is not declared here, and the reason is structural rather
# than a missing afternoon's work.
#
# Core's declarative EP lives in ``common/moe_sharding.py::set_moe_sharding_config``,
# which writes ``moe_cfg.routed_experts.inner_experts.sharding_config``, and the
# dispatcher's meshes are wired from ``moe_cfg.routed_experts.token_dispatcher``.
# Both paths require core's nesting:
#
#     moe_cfg.routed_experts (RoutedExperts.Config)
#         .inner_experts     (GroupedExperts.Config)
#         .token_dispatcher
#
# This model flattens it: ``routed_experts`` IS the GroupedExperts config, with
# ``inner_experts`` supplied as a module-level property, and ``token_dispatcher``
# hangs off the MoE config directly. So EP here is not merely unimplemented --
# the config shape cannot reach core's EP machinery, which is what every other
# MoE model uses. Restoring the nesting is the enabling change.
