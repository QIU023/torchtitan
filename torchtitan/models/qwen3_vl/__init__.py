# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.common import Embedding, Linear, RoPE, TransformerBlock
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3VLModel
from .parallelize import parallelize_qwen3_vl
from .parallelize_pp import pipeline_qwen3_vl
from .perceiver_resampler_projector import Qwen3VLPerceiverResamplerProjector
from .state_dict_adapter import Qwen3VLStateDictAdapter
from .vision_encoder import Qwen3VLVisionEncoder

__all__ = [
    "parallelize_qwen3_vl",
    "pipeline_qwen3_vl",
    "Qwen3VLModel",
    "Qwen3VLPerceiverResamplerProjector",
    "qwen3_vl_configs",
    "QWEN3_VL_SPECIAL_TOKENS",
]

QWEN3_VL_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}

_EPS = 1e-6


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _vl_linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )


def _qwen3_vl_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _qwen3_vl_q_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _vl_vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    n_layers: int,
    n_heads: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    deepstack_visual_indices: list[int],
    in_channels: int = 3,
) -> Qwen3VLVisionEncoder.Config:
    """Build a fully-specified Qwen3VLVisionEncoder.Config."""
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)
    return Qwen3VLVisionEncoder.Config(
        dim=dim,
        ffn_dim=ffn_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=num_position_embeddings,
        deepstack_visual_indices=deepstack_visual_indices,
        patch_embed_proj=_vl_linear(patch_dim, dim),
        attn_qkv=_vl_linear(dim, dim * 3),
        attn_proj=_vl_linear(dim, dim),
        mlp_fc1=_vl_linear(dim, ffn_dim),
        mlp_fc2=_vl_linear(ffn_dim, dim),
        merger_fc1=_vl_linear(merged_hidden_size, merged_hidden_size),
        merger_fc2=_vl_linear(merged_hidden_size, out_hidden_size),
        param_init=_POS_EMBED_INIT,
    )


_RESAMPLER_PARAM_INIT = {
    # Latents init follows Flamingo's truncated-normal small-sigma
    # convention (matches Q-Former queries init for sibling-projector
    # parity).
    "latents": partial(nn.init.trunc_normal_, mean=0.0, std=0.02),
    # Temporal-pos table uses the standard 0.02 trunc-normal init too;
    # see PR doc Open Issues re: scale choice.
    "temporal_pos.weight": partial(nn.init.trunc_normal_, mean=0.0, std=0.02),
}


def _vl_resampler_config(
    *,
    in_features: int,
    lm_dim: int,
    num_latents: int = 64,
    num_layers: int = 6,
    n_heads: int = 16,
    ffn_mult: int = 2,
    t_max: int = 32,
) -> Qwen3VLPerceiverResamplerProjector.Config:
    """Build a fully-specified Perceiver Resampler projector config.

    Args:
        in_features: KV input dim. Set to ViT hidden dim (e.g. 1152 for
            Qwen3-VL-8B) when bypassing the in-encoder spatial merger;
            set to ``lm_dim`` (e.g. 4096) when consuming post-merger
            features.
        lm_dim: LM hidden dim (output dim). 4096 for Qwen3-VL-8B.
        num_latents: Fixed-length output token count. 64 is the target
            for sibling-projector parity (Q-Former-64); ~26x compression
            over the stock 3-cam x 4-frame visual budget.
        num_layers: Number of (self-attn + cross-attn + FFN) blocks.
        n_heads: Number of attention heads in self/cross-attn.
        ffn_mult: FFN hidden-dim multiplier (FFN hidden = ffn_mult * lm_dim).
            Default 2 (smaller than Q-Former's 4) because the resampler
            block also carries a self-attn sub-layer.
        t_max: Temporal positional embedding table size. Larger frame
            indices clamp to ``t_max - 1``.
    """
    ffn_hidden = ffn_mult * lm_dim
    return Qwen3VLPerceiverResamplerProjector.Config(
        in_features=in_features,
        lm_dim=lm_dim,
        num_latents=num_latents,
        num_layers=num_layers,
        n_heads=n_heads,
        ffn_mult=ffn_mult,
        t_max=t_max,
        # Self-attn over latents: lm_dim -> lm_dim everywhere.
        self_attn_q_proj=_vl_linear(lm_dim, lm_dim),
        self_attn_k_proj=_vl_linear(lm_dim, lm_dim),
        self_attn_v_proj=_vl_linear(lm_dim, lm_dim),
        self_attn_o_proj=_vl_linear(lm_dim, lm_dim),
        # Cross-attn: Q is lm_dim (latents), KV is in_features (vision).
        cross_attn_q_proj=_vl_linear(lm_dim, lm_dim),
        cross_attn_k_proj=_vl_linear(in_features, lm_dim),
        cross_attn_v_proj=_vl_linear(in_features, lm_dim),
        cross_attn_o_proj=_vl_linear(lm_dim, lm_dim),
        # FFN over latents.
        ffn_fc1=_vl_linear(lm_dim, ffn_hidden),
        ffn_fc2=_vl_linear(ffn_hidden, lm_dim),
        param_init=_RESAMPLER_PARAM_INIT,
    )


def _build_qwen3_vl_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3-VL models with depth-scaled inits."""
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_vl_norm(dim),
                ffn_norm=_qwen3_vl_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=FlexAttention.Config(),
                    mask_type="block_causal",
                    rope_backend="cos_sin",
                    qk_norm=_qwen3_vl_q_norm(head_dim),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _build_qwen3_vl_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3-VL models with depth-scaled inits."""
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_vl_norm(dim),
                ffn_norm=_qwen3_vl_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=FlexAttention.Config(),
                    mask_type="block_causal",
                    rope_backend="cos_sin",
                    qk_norm=_qwen3_vl_q_norm(head_dim),
                ),
                moe=make_moe_config(
                    num_experts=num_experts,
                    score_before_experts=False,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_depth_init(layer_id),
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=True,
                    ),
                    experts=make_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        param_init=_depth_experts_init(layer_id),
                    ),
                ),
            )
        )
    return layers


def _debugmodel() -> Qwen3VLModel.Config:
    dim = 256
    head_dim = 64
    n_layers = 4
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_vl_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            hidden_dim=512,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indices=[1, 2, 3],
        ),
        mrope_section=[8, 8, 8],
    )


def _debugmodel_moe() -> Qwen3VLModel.Config:
    dim = 256
    head_dim = 64
    n_layers = 1
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_vl_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=64,
            top_k=8,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indices=[1, 2, 3],
        ),
        mrope_section=[8, 8, 8],
    )


def _2b() -> Qwen3VLModel.Config:
    dim = 2048
    head_dim = 128
    n_layers = 28
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_vl_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=6144,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=1024,
            ffn_dim=4096,
            n_layers=24,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indices=[5, 11, 17],
        ),
        mrope_section=[24, 20, 20],
    )


def _8b() -> Qwen3VLModel.Config:
    dim = 4096
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_vl_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=12288,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
            deepstack_visual_indices=[8, 16, 24],
        ),
        mrope_section=[24, 20, 20],
    )


def _8b_perceiver_resampler() -> Qwen3VLModel.Config:
    """Qwen3-VL-8B variant with a Perceiver Resampler projector.

    Flamingo-style temporal-aware resampler with 64 latents, 6 layers,
    and a per-frame learnable temporal positional encoding (T_max=32).
    Identical to ``_8b`` except for ``projector_type="perceiver_resampler"``
    and the populated ``perceiver_resampler`` field.

    The resampler's KV input dim defaults to the LM dim (4096) -- i.e.
    the resampler consumes the *post-merger* encoder output. See
    ``docs/upstream_prs/007_torchtitan_qwen3_vl_resampler.md`` for the
    open question on switching to pre-merger features
    (``in_features=1152``) to avoid the in-encoder 2x2 spatial-merge
    compression on top of the resampler's ~26x token-budget compression.
    """
    base = _8b()
    resampler_cfg = _vl_resampler_config(
        in_features=base.vision_encoder.out_hidden_size,  # 4096 (post-merger)
        lm_dim=base.dim,  # 4096
        num_latents=64,
        num_layers=6,
        n_heads=16,
        ffn_mult=2,
        t_max=32,
    )
    return dataclasses.replace(
        base,
        projector_type="perceiver_resampler",
        perceiver_resampler=resampler_cfg,
    )


# Qwen3-VL MoE models


def _30b_a3b() -> Qwen3VLModel.Config:
    dim = 2048
    head_dim = 128
    n_layers = 48
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_vl_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=128,
            top_k=8,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indices=[8, 16, 24],
        ),
        mrope_section=[24, 20, 20],
    )


def _235b_a22b() -> Qwen3VLModel.Config:
    dim = 4096
    head_dim = 128
    n_layers = 94
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_vl_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=1536,
            num_experts=128,
            top_k=8,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
            deepstack_visual_indices=[8, 16, 24],
        ),
        mrope_section=[24, 20, 20],
    )


qwen3_vl_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_moe": _debugmodel_moe,
    "2B": _2b,
    "8B": _8b,
    "8B-perceiver-resampler": _8b_perceiver_resampler,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = qwen3_vl_configs[flavor]()
    return ModelSpec(
        name="qwen3_vl",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=pipeline_qwen3_vl,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3VLStateDictAdapter,
    )
