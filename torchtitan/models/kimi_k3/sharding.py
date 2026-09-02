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
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config", *, enable_ep: bool, enable_sp: bool = False
) -> None:
    """Declare the sharding expert parallel acts on.

    The routed experts shard on the expert axis; ``set_moe_sharding_config``
    declares that layout, and its input boundary lifts the plain incoming
    activations itself, so no decoder-level declaration is needed.
    """
    for layer in config.layers:
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=enable_ep,
                # TODO: flip to True from the caller once the
                # tensor-parallel PR lands; with EP alone the internals run
                # without sequence parallel.
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
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
    attention_cfg.wq_a.sharding_config = _tp_replicate_config()
    attention_cfg.wkv_a.sharding_config = _tp_replicate_config()
    attention_cfg.q_norm.sharding_config = norm_config(enable_sp=False)
    attention_cfg.kv_norm.sharding_config = norm_config(enable_sp=False)
    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_kda_sharding(delta_attention_cfg, *, enable_sp: bool) -> None:
    """Head-parallel TP for KDA.

    The delta rule is independent per head, so the projections that produce
    or consume the head axis split on it, the per-head state (``A_log``,
    ``dt_bias``, the depthwise convolutions) shards with the heads, and the
    kernel runs on the local heads behind a ``local_map`` on ``inner_kda``.
    The one low-rank compression, ``forget_a``, is rank-sized and stays whole.
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
    delta_attention_cfg.sharding_config = kda_module_config
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
        # Packed-document boundaries, the same on every tp rank (None under flex).
        "cu_seqlens": dense_activation_placement(tp=spmd.I, cp=spmd.V),
    }
    delta_attention_cfg.inner_kda.sharding_config = ShardingConfig(
        in_src_shardings=inputs,
        in_dst_shardings=inputs,
        out_src_shardings=heads,
        local_map=LocalMapConfig(in_grad_placements=tuple(inputs.values())),
    )


def set_tensor_parallel_sharding_config(
    config: "KimiK3Model.Config", *, enable_sp: bool = False
) -> None:
    """Declare the sharding tensor parallel acts on.

    Head and feature axes shard. With ``enable_sp`` the token stream between
    modules carries the TP-axis Shard(0) of sequence parallel: norms compute
    on the shard, the attention module boundaries gather it (the cores need
    the full sequence) and the rowwise outputs reduce-scatter back, the
    llama3 template; without it the stream stays whole on the TP axis. The
    MoE internals are declared by ``set_kimi_k3_sharding_config``; this adds
    the latent projections around them.
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    config.output_res_norm.sharding_config = norm_config(enable_sp=enable_sp)
    config.output_res_proj.sharding_config = _tp_replicate_config()
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
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
                cfg.sharding_config = _tp_replicate_config()
        if layer.attention is not None:
            _set_mla_sharding(layer.attention, enable_sp=enable_sp)
        if layer.delta_attention is not None:
            _set_kda_sharding(layer.delta_attention, enable_sp=enable_sp)
        if layer.feed_forward is not None:
            set_dense_ffn_sharding(
                layer.feed_forward, attn_x_layout=attn_x_layout, enable_sp=enable_sp
            )
        if layer.moe is not None:
            # The MoE module boundary gathers the sequence shard under SP
            # without EP, so the latent pair and its norm see the whole stream.
            layer.moe.routed_down.sharding_config = _tp_replicate_config()
            layer.moe.routed_up.sharding_config = _tp_replicate_config()
            layer.moe.routed_norm.sharding_config = norm_config(enable_sp=False)
