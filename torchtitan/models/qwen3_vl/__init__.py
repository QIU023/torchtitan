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
from .pixelshuffle_projector import Qwen3VLPixelShufflePlusLinearProjector
from .qformer_projector import Qwen3VLQFormerProjector
from .state_dict_adapter import Qwen3VLStateDictAdapter
from .vision_encoder import Qwen3VLVisionEncoder

__all__ = [
    "parallelize_qwen3_vl",
    "pipeline_qwen3_vl",
    "Qwen3VLModel",
    "Qwen3VLPixelShufflePlusLinearProjector",
    "Qwen3VLQFormerProjector",
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


_QFORMER_QUERY_INIT = {
    "queries": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)
}


def _vl_qformer_config(
    *,
    in_features: int,
    lm_dim: int,
    num_queries: int = 64,
    num_layers: int = 6,
    n_heads: int = 16,
    ffn_mult: int = 4,
) -> Qwen3VLQFormerProjector.Config:
    """Build a fully-specified Q-Former projector config.

    Args:
        in_features: KV input dim. Set to ViT hidden dim (e.g. 1152 for
            Qwen3-VL-8B) when bypassing the in-encoder spatial merger;
            set to ``lm_dim`` (e.g. 4096) when consuming post-merger
            features.
        lm_dim: LM hidden dim (output dim). 4096 for Qwen3-VL-8B.
        num_queries: Fixed-length output token count. 64 is the target
            for TRT-friendly inference; ~26x compression over the stock
            3-cam x 4-frame visual budget.
        num_layers: Number of cross-attn + FFN blocks.
        n_heads: Number of attention heads in cross-attn.
        ffn_mult: FFN hidden-dim multiplier (FFN hidden = ffn_mult * lm_dim).
    """
    ffn_hidden = ffn_mult * lm_dim
    return Qwen3VLQFormerProjector.Config(
        in_features=in_features,
        lm_dim=lm_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        n_heads=n_heads,
        ffn_mult=ffn_mult,
        q_proj=_vl_linear(lm_dim, lm_dim),
        k_proj=_vl_linear(in_features, lm_dim),
        v_proj=_vl_linear(in_features, lm_dim),
        o_proj=_vl_linear(lm_dim, lm_dim),
        ffn_fc1=_vl_linear(lm_dim, ffn_hidden),
        ffn_fc2=_vl_linear(ffn_hidden, lm_dim),
        param_init=_QFORMER_QUERY_INIT,
    )


def _vl_pixelshuffle_config(
    *,
    in_features: int,
    lm_dim: int,
    shuffle_ratio: int = 2,
) -> Qwen3VLPixelShufflePlusLinearProjector.Config:
    """Build a fully-specified PixelShuffle + Linear projector config.

    Args:
        in_features: Channel dim of the visual features fed into the
            projector. Set to ``out_hidden_size`` (e.g. 4096 for
            Qwen3-VL-8B) when the projector consumes the in-encoder
            PatchMerger output (default / minimum-invasive wiring). Set to
            ``vision_encoder.dim`` (e.g. 1152 for Qwen3-VL-8B) when
            bypassing the in-encoder merger (pre-merger path; requires a
            follow-up wiring change, see
            ``docs/upstream_prs/006_torchtitan_qwen3_vl_pixelshuffle.md``).
        lm_dim: LM hidden dim (output channel dim). 4096 for Qwen3-VL-8B.
        shuffle_ratio: Spatial compression ratio. Default 2 (LLaVA-NeXT
            standard).

    The Linear sits at ``Linear(in_features * shuffle_ratio**2 -> lm_dim)``.
    """
    proj_in = in_features * (shuffle_ratio ** 2)
    return Qwen3VLPixelShufflePlusLinearProjector.Config(
        in_features=in_features,
        lm_dim=lm_dim,
        shuffle_ratio=shuffle_ratio,
        proj=_vl_linear(proj_in, lm_dim),
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


def _8b_qformer() -> Qwen3VLModel.Config:
    """Qwen3-VL-8B variant with a Q-Former projector (64 queries, 6 layers).

    Identical to ``_8b`` except for ``projector_type="qformer"`` and the
    populated ``qformer`` field. The Q-Former's KV input dim is the LM
    dim (4096) by default — i.e. the Q-Former consumes the *post-merger*
    encoder output. See ``docs/upstream_prs/005_*.md`` for the open
    question on switching to pre-merger features (``in_features=1152``)
    to avoid the in-encoder 2x2 spatial-merge compression on top of
    Q-Former's 26x token-budget compression.
    """
    base = _8b()
    qformer_cfg = _vl_qformer_config(
        in_features=base.vision_encoder.out_hidden_size,  # 4096 (post-merger)
        lm_dim=base.dim,  # 4096
        num_queries=64,
        num_layers=6,
        n_heads=16,
    )
    return dataclasses.replace(
        base,
        projector_type="qformer",
        qformer=qformer_cfg,
    )


def _8b_pixelshuffle() -> Qwen3VLModel.Config:
    """Qwen3-VL-8B variant with a PixelShuffle 2x + Linear projector.

    Identical to ``_8b`` except for ``projector_type="pixelshuffle"`` and
    the populated ``pixelshuffle`` field.

    The projector consumes the **post-merger** encoder output by default
    (``in_features=4096``). With ``shuffle_ratio=2`` this gives a TOTAL
    spatial compression of (in-encoder merger 2x2) * (PixelShuffle 2x2) =
    16x raw ViT patches, NOT the 4x stated in the LLaVA-NeXT-style
    deterministic-compression spec. Reaching the canonical 4x ratio
    requires bypassing the in-encoder PatchMerger and feeding raw ViT
    features into the projector (``in_features=1152``) — a follow-up
    plumbing change tracked in
    ``docs/upstream_prs/006_torchtitan_qwen3_vl_pixelshuffle.md``.

    Param count (lm_dim=4096, in_features=4096, shuffle_ratio=2):
    ``Linear(4 * 4096, 4096)`` ~ 67M params (bias-included).

    Param count (lm_dim=4096, in_features=1152, shuffle_ratio=2) — i.e.
    the pre-merger path the 17M target in the spec assumes:
    ``Linear(4 * 1152, 4096)`` ~ 19M params.
    """
    base = _8b()
    ps_cfg = _vl_pixelshuffle_config(
        in_features=base.vision_encoder.out_hidden_size,  # 4096 (post-merger)
        lm_dim=base.dim,  # 4096
        shuffle_ratio=2,
    )
    return dataclasses.replace(
        base,
        projector_type="pixelshuffle",
        pixelshuffle=ps_cfg,
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
    "8B-qformer": _8b_qformer,
    "8B-pixelshuffle": _8b_pixelshuffle,
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
