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

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model


def _tp_replicate_config() -> ShardingConfig:
    """Weight replicated on the TP axis, with no activation boundary declared.

    Not core's ``vision_invariant_linear_config``: that one declares all four
    activation boundaries, which lifts the input to a DTensor, and
    ``common/linear.py``'s ``Linear.forward`` unwraps its own weight to local
    before the matmul -- so the two meet as "aten.mm.default got mixed
    torch.Tensor and DTensor". Core's own dense ``colwise_config`` and
    ``rowwise_config`` leave those boundaries None for the same reason; this is
    the replicated member of that family, which core does not have.
    """
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


def _set_mla_sharding(attention_cfg) -> None:
    """Head-parallel TP for MLA.

    Not ``set_gqa_attention_sharding``: that one asserts a GQAttention.Config,
    and this attention has no fused qkv linear. The shape is the same though --
    the projections that produce or consume the head axis split on it, the two
    compressions stay whole because they are rank-sized rather than head-sized.
    """
    attention_cfg.wq_b.sharding_config = colwise_config()
    attention_cfg.wkv_b.sharding_config = colwise_config()
    attention_cfg.gate.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config()
    attention_cfg.wq_a.sharding_config = _tp_replicate_config()
    attention_cfg.wkv_a.sharding_config = _tp_replicate_config()
    attention_cfg.q_norm.sharding_config = norm_config(enable_sp=False)
    attention_cfg.kv_norm.sharding_config = norm_config(enable_sp=False)
    # The kernel boundary: FlexAttention indexes plain mask tensors, so the
    # declared q/k/v drop to locals inside a local_map region -- the same
    # helper qwen3_5's full-attention layers use, and the same (T, N, H)
    # activation family.
    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_kda_sharding(delta_attention_cfg) -> None:
    """KDA is invariant at TP, and has to be.

    Its kernels are fla triton and never see a DTensor, so nothing declared
    here could reach them. Declaring the module invariant says exactly that,
    and keeps every parameter on one mesh so clip_grad_norm_ can stack them.
    """
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "q_conv",
        "k_conv",
        "v_conv",
        "forget_a",
        "forget_b",
        "beta",
        "output_gate",
        "output_proj",
    ):
        getattr(delta_attention_cfg, name).sharding_config = _tp_replicate_config()
    delta_attention_cfg.output_norm.sharding_config = norm_config(enable_sp=False)
    # A_log and dt_bias are the module's OWN parameters, so only a
    # module-level declaration reaches them.
    delta_attention_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": SpmdLayout({MeshAxisName.DP: spmd.R, MeshAxisName.TP: spmd.I}),
            "dt_bias": SpmdLayout({MeshAxisName.DP: spmd.R, MeshAxisName.TP: spmd.I}),
        }
    )


def set_tensor_parallel_sharding_config(config: "KimiK3Model.Config") -> None:
    """Declare the sharding tensor parallel acts on.

    Sequence parallel is not offered: the stream stays Replicate on the TP
    axis and only head/feature axes shard.

    Only the parts that upstream already has a helper for. Everything else is
    left undeclared on purpose rather than filled in with a guess -- an
    undeclared module is inert, a wrongly declared one is a silent numerics
    change.
    """
    set_decoder_sharding_config(config, enable_sp=False)
    # The FINAL aggregation pair, same treatment as the per-layer ones below:
    # both sit on the block stream, which TP does not split, and
    # _apply_attention_residual multiplies their weights together -- one
    # declared and one not is a mixed mul.
    if config.output_res_norm is not None:
        config.output_res_norm.sharding_config = norm_config(enable_sp=False)
    if config.output_res_proj is not None:
        config.output_res_proj.sharding_config = _tp_replicate_config()
    attn_x_layout = dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    for layer in config.layers:
        # Every norm and the residual projections stay whole: they sit on the
        # block stream, which TP does not split. Left undeclared they meet
        # DTensor activations as plain tensors.
        for name in (
            "attention_norm",
            "ffn_norm",
            "attention_res_norm",
            "ffn_res_norm",
        ):
            cfg = getattr(layer, name, None)
            if cfg is not None:
                cfg.sharding_config = norm_config(enable_sp=False)
        for name in ("attention_res_proj", "ffn_res_proj"):
            cfg = getattr(layer, name, None)
            if cfg is not None:
                cfg.sharding_config = _tp_replicate_config()
        if layer.attention is not None:
            _set_mla_sharding(layer.attention)
        if layer.delta_attention is not None:
            _set_kda_sharding(layer.delta_attention)
        if layer.feed_forward is not None:
            set_dense_ffn_sharding(
                layer.feed_forward,
                attn_x_layout=attn_x_layout,
                enable_sp=False,
            )
        if layer.moe is not None:
            # The latent pair is Kimi's addition to core's MoE, so
            # set_moe_sharding_config does not know it. It stays whole: it
            # compresses to a rank, not heads or experts.
            layer.moe.routed_down.sharding_config = _tp_replicate_config()
            layer.moe.routed_up.sharding_config = _tp_replicate_config()
            layer.moe.routed_norm.sharding_config = norm_config(enable_sp=False)
            set_moe_sharding_config(
                layer.moe,
                enable_ep=False,
                enable_sp=False,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )
