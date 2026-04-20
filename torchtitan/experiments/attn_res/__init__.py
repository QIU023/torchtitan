# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Attention Residuals experiment (Kimi Team, 2026).

Registers Llama3 dense model flavors with Block AttnRes enabled and exposes
them through torchtitan's standard ``--module attn_res --config <name>`` path.

Design: the experiment is self-contained under this folder. Core torchtitan
model files (``torchtitan/models/common/decoder.py``,
``torchtitan/models/llama3/model.py``) are not modified; all AttnRes forward
paths live in :mod:`torchtitan.experiments.attn_res.model` as subclasses of
the core blocks.
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

# Total number of transformer layers for the 150M config. Fixed by the
# Llama3 shape; all num_blocks values must divide it (see _150m_attn_res).
_150M_N_LAYERS = 12

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.experiments.attn_res.attn_res import AttnResConfig, AttnResProjection
from torchtitan.experiments.attn_res.model import (
    AttnResLlama3Model,
    AttnResLlama3TransformerBlock,
)
from torchtitan.experiments.attn_res.pipeline_adapter import (
    pipeline_llm_with_cache_adapter,
)
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec


__all__ = [
    "AttnResLlama3Model",
    "AttnResLlama3TransformerBlock",
    "attn_res_configs",
    "model_registry",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
# Pseudo-query vectors for Block AttnRes MUST be zero-initialized so that
# initial softmax weights are uniform -- i.e., the model starts equivalent to
# standard residuals and avoids training volatility in the first steps.
_ATTN_RES_PROJ_INIT = {"weight": nn.init.zeros_}


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


def _build_attn_res_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
) -> list[AttnResLlama3TransformerBlock.Config]:
    layers: list[AttnResLlama3TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        layers.append(
            AttnResLlama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=ScaledDotProductAttention.Config(),
                    mask_type="causal",
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
                attn_res_proj=AttnResProjection.Config(
                    dim=dim, param_init=_ATTN_RES_PROJ_INIT
                ),
                mlp_res_proj=AttnResProjection.Config(
                    dim=dim, param_init=_ATTN_RES_PROJ_INIT
                ),
                attn_res_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                mlp_res_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
            )
        )
    return layers


def _debugmodel_attn_res() -> AttnResLlama3Model.Config:
    """Debug model with Block Attention Residuals enabled.

    6 layers / 3 blocks = 2 layers per block, so every even-indexed layer
    is a block start. Intentionally small for unit/smoke testing.
    """
    dim = 256
    n_heads = 16
    n_layers = 6
    num_blocks = 3
    return AttnResLlama3Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_attn_res_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
        ),
        attn_res=AttnResConfig(enabled=True, num_blocks=num_blocks),
        final_attn_res_proj=AttnResProjection.Config(
            dim=dim, param_init=_ATTN_RES_PROJ_INIT
        ),
        final_attn_res_norm=RMSNorm.Config(
            normalized_shape=dim, param_init=_NORM_INIT
        ),
    )


def _150m_attn_res(
    num_blocks: int = 6,
    n_layers: int = _150M_N_LAYERS,
    enable_weight_tying: bool = True,
) -> AttnResLlama3Model.Config:
    """~150M dense Llama3 with Block AttnRes enabled.

    Args:
        num_blocks: Number of attention-residual blocks. Must divide
            ``n_layers``. Paper sweet spot is ``N=8``; default is 6
            (closest divisor below 8 for n_layers=12). N=1 is equivalent
            to standard residuals and is disallowed.
        n_layers: Total transformer layers. Default 12 (original 150M
            shape). Pass 16 to align with the Phase-3 8-GPU PP layout
            (PP=8, layers_per_stage=1 -> 16 virtual stages, 2 chunks
            per rank under Interleaved1F1B).
        enable_weight_tying: Tie input embedding with output projection.
            Must be False under Pipeline Parallel — torchtitan's
            parallelize_llama explicitly raises on tying + PP.

    All per-layer and final pseudo-queries are zero-initialized so
    training begins numerically equivalent to standard residuals (uniform
    softmax over sources). Uses GQA (n_kv_heads=4) and tied embeddings to
    keep parameter count mostly in the transformer stack.
    """
    if num_blocks < 2 or n_layers % num_blocks != 0:
        raise ValueError(
            f"num_blocks={num_blocks} must be >=2 and divide "
            f"n_layers={n_layers}. Valid: "
            f"{sorted(d for d in range(2, n_layers + 1) if n_layers % d == 0)}"
        )
    dim = 768
    n_heads = 12
    n_kv_heads = 4
    vocab_size = 128256
    return AttnResLlama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_attn_res_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=1.0
            ),
        ),
        attn_res=AttnResConfig(enabled=True, num_blocks=num_blocks),
        final_attn_res_proj=AttnResProjection.Config(
            dim=dim, param_init=_ATTN_RES_PROJ_INIT
        ),
        final_attn_res_norm=RMSNorm.Config(
            normalized_shape=dim, param_init=_NORM_INIT
        ),
    )


attn_res_configs = {
    "debugmodel_attn_res": _debugmodel_attn_res,
    "150M_attn_res": _150m_attn_res,  # default N=6
    # Ablation flavors over num_blocks. n_layers=12, so divisors are
    # {2, 3, 4, 6, 12}. N=1 is degenerate (= standard residuals).
    "150M_attn_res_n2": partial(_150m_attn_res, num_blocks=2),
    "150M_attn_res_n3": partial(_150m_attn_res, num_blocks=3),
    "150M_attn_res_n4": partial(_150m_attn_res, num_blocks=4),
    "150M_attn_res_n12": partial(_150m_attn_res, num_blocks=12),
    # 16-layer variant sized to align with 8-GPU PP: n_layers=16,
    # num_blocks=8 gives 2 layers per block (paper sweet spot N=8).
    # The Phase-3 launchers (phase3/launch_8gpu_{naive,adapter}.sh) pass
    # --parallelism.pipeline_parallel_layers_per_stage=1, with
    # first_stage_less_layers=0 and last_stage_less_layers=0, yielding
    # (n_layers=16 + first_less=0 + last_less=0) / layers_per_stage=1
    # = 16 virtual stages. With PP=8 that gives 16 / 8 = 2 chunks per
    # rank, which satisfies Interleaved1F1B's ">=2 chunks per rank"
    # requirement and preserves steady-state overlap. Every other
    # virtual stage boundary also coincides with a block boundary, so
    # the cross-stage caching adapter's "send only new blocks"
    # invariant is exercised at half the stage transitions.
    "150M_attn_res_L16_n8": partial(
        _150m_attn_res, num_blocks=8, n_layers=16, enable_weight_tying=False
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    config = attn_res_configs[flavor]()
    return ModelSpec(
        name="attn_res",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_llama,
        # Custom pipelining_fn wraps core pipeline_llm with the AttnRes
        # cross-stage caching adapter (opt-in via TORCHTITAN_ATTNRES_CACHE=1).
        # When the flag is unset, this is a thin passthrough over
        # torchtitan.distributed.pipeline_parallel.pipeline_llm.
        pipelining_fn=pipeline_llm_with_cache_adapter,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
