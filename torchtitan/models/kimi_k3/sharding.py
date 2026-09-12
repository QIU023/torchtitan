# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3. Same pattern as ``qwen3_5/sharding.py``.

Declarations only: functions here set ``ShardingConfig`` on sub-configs of an
already-built config tree, and ``model.parallelize()`` applies them through the
Module protocol. Nothing here touches a mesh or a device.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
    token_id_placement,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.vision_encoder_sharding import set_moonvit_sharding_config
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model
    from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder


DP = MeshAxisName.DP
TP = MeshAxisName.TP


def _set_inner_kda_sharding(inner_kda: Module.Config) -> None:
    """Set the local boundary around Attention Gym's KDA kernels."""
    token_channels = SpmdType({DP: spmd.V}, partition_spec=spmd.PartitionSpec(DP, None))
    token_heads = SpmdType(
        {DP: spmd.V}, partition_spec=spmd.PartitionSpec(DP, None, None)
    )
    parameter = SpmdType({DP: spmd.R})

    inner_kda.sharding_config = ShardingConfig(
        in_src_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_heads,
            "raw_beta_TH": token_channels,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        in_dst_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_heads,
            "raw_beta_TH": token_channels,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        out_src_shardings=token_heads,
        out_dst_shardings=token_heads,
        local_map=LocalMapConfig(
            in_grad_placements=(
                token_channels,
                token_channels,
                token_channels,
                token_heads,
                token_channels,
                parameter,
                parameter,
                parameter,
                parameter,
                parameter,
            ),
        ),
    )


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config",
    *,
    enable_ep: bool,
    enable_tp: bool = False,
    enable_sp: bool = False,
) -> None:
    """Declare Kimi K3's sharding: vision buffers, experts, and tensor parallel.

    Without tensor parallelism the vision buffers replicate across DP ranks,
    the routed experts shard on the expert axis (``set_moe_sharding_config``
    declares that layout; its input boundary lifts the plain incoming
    activations itself) and the KDA kernel runs behind a DP-local boundary.

    With ``enable_tp`` the head and feature axes shard: MLA and KDA are
    head-parallel, the MoE's latent projections wrap core's expert sharding,
    and the vision tower takes Kimi K2.5's plan (colwise / rowwise linears,
    invariant norms and tables, K3's projector norm after its second linear).
    With ``enable_sp`` the token stream between modules carries the TP-axis
    Shard(0) of sequence parallel: norms compute on the shard, the attention
    module boundaries gather it and the rowwise outputs reduce-scatter back
    (the llama3 template); without it the stream stays whole on the TP axis.
    """
    if not enable_tp:
        if config.vision_encoder is not None:
            _set_vision_buffer_sharding(config.vision_encoder)
        for layer in config.layers:
            if layer.delta_attention is not None:
                _set_inner_kda_sharding(layer.delta_attention.inner_kda)
    _set_moe_sharding(config, enable_ep=enable_ep, enable_sp=enable_sp)
    if enable_tp:
        _set_tensor_parallel_sharding(config, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_moe_sharding(
    config: "KimiK3Model.Config", *, enable_ep: bool, enable_sp: bool
) -> None:
    for layer in config.layers:
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )


def _set_vision_buffer_sharding(config: "KimiK3VisionEncoder.Config") -> None:
    """Declare the K3 rotary buffer replicated across DP ranks."""
    config.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": SpmdType({DP: spmd.R})},
    )


def _stream_param_config(*, enable_sp: bool) -> ShardingConfig:
    """Weight that reads and feeds the token stream between the TP modules.

    Without SP the stream is invariant on TP and every module converts it
    I -> R on entry, so the gradient reaching this weight is identical on every
    rank: invariant, where replicated would sum the copies. Under SP the stream
    is the sequence shard and the gradient a per-rank partial: replicated, and
    FSDP sums it. The rule ``norm_config`` applies to the norms on that stream.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        }
    )


def _tp_replicate_config() -> ShardingConfig:
    """Weight replicated on the TP axis, with no activation boundary declared.

    The replicated member of the colwise/rowwise family, which core does not
    have: declaring the activation boundaries would lift the input to a
    DTensor while ``Linear.forward`` unwraps its own weight to local.
    """
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


def _set_mla_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """Head-parallel TP for MLA.

    The projections that produce or consume the head axis split on it; the
    two compressions stay whole because they are rank-sized, not head-sized.
    """
    attention_cfg.wq_b.sharding_config = colwise_config()
    attention_cfg.wkv_b.sharding_config = colwise_config()
    attention_cfg.gate.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    if enable_sp:
        # The module boundary gathers the sequence shard on the way in -- the
        # attention core needs the full sequence -- and wo reduce-scatters
        # back to Shard(0), the GQA pattern.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x_TD": dense_sequence_parallel_placement()},
            in_dst_shardings={
                "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            },
        )
    else:
        # The block stream is invariant on TP while the attention body is
        # replicated with sharded heads: entering converts
        # I -> R (no-op forward, all-reduce of the input gradient in backward,
        # since wq_b/wkv_b are colwise); wo's rowwise boundary hands the
        # stream back invariant.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={
                "x_TD": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            },
            in_dst_shardings={
                "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            },
        )
    attention_cfg.wq_a.sharding_config = _tp_replicate_config()
    attention_cfg.wkv_a.sharding_config = _tp_replicate_config()
    # Inside the replicated body the activations are R and the norms feed the
    # colwise wq_b/wkv_b, so their weights take partial gradients: replicated,
    # state only (norm_config's invariant [T, D] boundary is the stream's).
    for name in ("q_norm", "kv_norm"):
        getattr(attention_cfg, name).sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
        )
    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_kda_sharding(delta_attention_cfg, *, enable_sp: bool) -> None:
    """Head-parallel TP for KDA.

    The delta rule is independent per head, so the projections that produce
    or consume the head axis split on it, the per-head state (``A_log``,
    ``dt_bias``, the depthwise convolutions) shards with the heads, and the
    kernel runs on the local heads behind the ``local_map`` on ``inner_kda``
    that ``_set_inner_kda_sharding`` installs; this redeclares it with the head
    axis sharded on tp. The one low-rank compression, ``forget_a``, is
    rank-sized and stays whole.
    """
    for name in ("q_proj", "k_proj", "v_proj", "forget_b", "beta", "output_gate"):
        getattr(delta_attention_cfg, name).sharding_config = colwise_config()
    delta_attention_cfg.forget_a.sharding_config = _tp_replicate_config()
    delta_attention_cfg.output_proj.sharding_config = rowwise_config(
        output_sp=enable_sp
    )
    head_param = dense_param_placement(tp=spmd.S(0))
    for name in ("q_conv", "k_conv", "v_conv"):
        getattr(delta_attention_cfg, name).sharding_config = ShardingConfig(
            state_shardings={"weight": head_param}
        )
    delta_attention_cfg.output_norm.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)}
    )
    kda_module_config = ShardingConfig(
        state_shardings={"A_log": head_param, "dt_bias": head_param}
    )
    if enable_sp:
        # Every projection reads the stream, so the module boundary gathers
        # the sequence shard once; output_proj reduce-scatters back.
        kda_module_config.in_src_shardings = {
            "x_TD": dense_sequence_parallel_placement()
        }
        kda_module_config.in_dst_shardings = {
            "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        }
    else:
        # The MLA boundary's twin: the invariant stream enters replicated (the
        # projections are colwise, so the backward all-reduces the input
        # gradient) and output_proj's rowwise exit hands it back invariant.
        kda_module_config.in_src_shardings = {
            "x_TD": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        }
        kda_module_config.in_dst_shardings = {
            "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        }
    delta_attention_cfg.sharding_config = kda_module_config
    # The kernel boundary: token-sharded on dp, head-sharded on tp; the per-head
    # parameters shard with the heads.
    features = dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
    heads = attention_activation_placement()
    inputs = {
        "query_TC": features,
        "key_TC": features,
        "value_TC": features,
        "raw_gate_THK": heads,
        "raw_beta_TH": features,
        "conv_q_weight_C1W": head_param,
        "conv_k_weight_C1W": head_param,
        "conv_v_weight_C1W": head_param,
        "A_log_H": head_param,
        "dt_bias_HK": head_param,
    }
    delta_attention_cfg.inner_kda.sharding_config = ShardingConfig(
        in_src_shardings=inputs,
        in_dst_shardings=inputs,
        out_src_shardings=heads,
        out_dst_shardings=heads,
        local_map=LocalMapConfig(
            in_grad_placements=tuple(layout for name, layout in inputs.items()),
        ),
    )


def _set_tensor_parallel_sharding(
    config: "KimiK3Model.Config", *, enable_sp: bool, enable_ep: bool
) -> None:
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    layer_input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    if config.vision_encoder is not None:
        _shard_decoder_after_embedding_scatter(
            config, layer_input_layout, enable_sp=enable_sp
        )
        set_moonvit_sharding_config(config.vision_encoder, projector_norm="post_norm")
    config.output_res_norm.sharding_config = norm_config(enable_sp=enable_sp)
    config.output_res_proj.sharding_config = _stream_param_config(enable_sp=enable_sp)
    for layer in config.layers:
        for name in (
            "attention_norm",
            "ffn_norm",
            "attention_res_norm",
            "ffn_res_norm",
        ):
            cfg = getattr(layer, name, None)
            if cfg is not None:
                cfg.sharding_config = norm_config(enable_sp=enable_sp)
        for name in ("attention_res_proj", "ffn_res_proj"):
            cfg = getattr(layer, name, None)
            if cfg is not None:
                cfg.sharding_config = _stream_param_config(enable_sp=enable_sp)
        if layer.attention is not None:
            _set_mla_sharding(layer.attention, enable_sp=enable_sp)
        if layer.delta_attention is not None:
            _set_kda_sharding(layer.delta_attention, enable_sp=enable_sp)
        if layer.feed_forward is not None:
            set_dense_ffn_sharding(
                layer.feed_forward,
                attn_x_layout=layer_input_layout,
                enable_sp=enable_sp,
            )
        if layer.moe is not None:
            # routed_down feeds the experts. Without EP they are TP-sharded and
            # hand back partial input gradients (replicated weight); with EP
            # they are whole on every tp rank, so it follows the stream's rule,
            # as routed_up does from the reduced norm output.
            layer.moe.routed_down.sharding_config = (
                _stream_param_config(enable_sp=enable_sp)
                if enable_ep
                else _tp_replicate_config()
            )
            routed_up_cfg = _stream_param_config(enable_sp=enable_sp)
            routed_norm_cfg = norm_config(enable_sp=enable_sp)
            if not enable_sp:
                # The experts' rowwise output is Partial on TP: the norm's
                # boundary ("x", nn.RMSNorm.forward's argument) reduces it,
                # routed_up re-enters Partial so core's MoE exit reduces it
                # with the shared experts' output, and that exit returns to
                # the invariant stream.
                routed_norm_cfg.in_src_shardings = {
                    "x": dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
                }
                routed_norm_cfg.in_dst_shardings = {
                    "x": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
                }
                routed_up_cfg.out_src_shardings = dense_activation_placement(
                    tp=spmd.I, cp=spmd.S(0)
                )
                routed_up_cfg.out_dst_shardings = dense_activation_placement(
                    tp=spmd.P, cp=spmd.S(0)
                )
                if layer.moe.sharding_config is not None:
                    layer.moe.sharding_config.out_dst_shardings = (
                        dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
                    )
            layer.moe.routed_up.sharding_config = routed_up_cfg
            layer.moe.routed_norm.sharding_config = routed_norm_cfg


def _block_residual_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdType:
    """Placement for the ``(tokens, entries, hidden)`` block-residual stack.

    The 3-D twin of ``dense_activation_placement``: DP shards the tokens,
    ``tp`` says how the TP axis holds the stack (``S(0)`` under sequence
    parallel, invariant or replicated otherwise).
    """
    if isinstance(tp, spmd.Shard):
        return SpmdType(
            {DP: spmd.V, TP: spmd.V},
            partition_spec=spmd.PartitionSpec((DP, TP), None, None),
        )
    return SpmdType(
        {DP: spmd.V, TP: tp}, partition_spec=spmd.PartitionSpec(DP, None, None)
    )


def _shard_decoder_after_embedding_scatter(
    config: "KimiK3Model.Config", layer_input_layout: SpmdType, *, enable_sp: bool
) -> None:
    """Keep ``tok_embeddings`` replicated on TP and restore the layout at layer 0.

    The vision scatter writes features at arbitrary sequence positions, so it
    needs the whole sequence on every rank (Kimi K2.5's and Qwen3.5's
    pattern). Layer 0's input boundary then takes the stream, and the empty
    block-residual stack that is cut from it, to the decoder's layout: the
    sequence shard under SP, invariant otherwise.
    """
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": token_id_placement()},
        in_dst_shardings={"input": token_id_placement()},
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
        out_dst_shardings=replicated,
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    config.layers[0].sharding_config = ShardingConfig(
        in_src_shardings={
            "x_TD": replicated,
            "block_residual_TND": _block_residual_placement(tp=spmd.R),
        },
        in_dst_shardings={
            "x_TD": layer_input_layout,
            "block_residual_TND": _block_residual_placement(
                tp=spmd.S(0) if enable_sp else spmd.I
            ),
        },
    )
