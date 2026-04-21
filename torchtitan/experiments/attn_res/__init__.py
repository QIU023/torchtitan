# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Attention Residuals experiment (Kimi Team, 2026).

Registers two families of AttnRes-native model flavors and exposes them
through torchtitan's standard ``--module attn_res --config <name>`` path:

1. **Dense + GQA** (``175M_attn_res`` and ablation variants) — the
   single-GPU reference used for the RFC evidence runs.
2. **MoE + MLA** (``dsv3_*_attn_res``) — DeepSeek-V3-shaped Mixture-of-
   Experts variants with AttnRes wired in. Closest open architectural
   match to Kimi's production line (Kimi Linear / K2 follow the
   "Moonlight / DeepSeek-V3 design" per paper §5). Reuses torchtitan's
   shared MoE (``models/common/moe.py``) and DSv3's MLA ``Attention``
   class unchanged — no MoE / MLA code duplicated here.

Design: the experiment is self-contained under this folder. Model classes
(``AttnResModel``, ``AttnResTransformerBlock``) inherit only from the shared
``torchtitan.models.common.decoder`` bases — no coupling to Llama3 or to
DSv3 model classes. The block's FFN branch is chosen per-layer
(``moe`` OR ``feed_forward``, DSv3 pattern), so a single model can mix
first-N-dense-then-MoE layers without a separate block class. AttnRes is
free to pivot further (e.g. add KDA when Kimi open-sources it) without
reaching back into core torchtitan.
"""

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch.nn as nn

# Total number of transformer layers for the 175M dense AttnRes config.
# All num_blocks values must divide n_layers (see
# _175m_attn_res). Name reflects total parameter count (~174M with tied
# embedding counted once); torchtitan's size-log convention reports this
# as 75.5M non-embedding.
_175M_N_LAYERS = 12

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.attn_res.attn_res import AttnResConfig, AttnResProjection
from torchtitan.experiments.attn_res.model import (
    AttnResModel,
    AttnResTransformerBlock,
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
from torchtitan.models.common.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
# DSv3-family imports: MLA attention (private to DSv3, reused per
# experiments/ one-way dependency rule), the MoE parallelize function,
# and the HF state-dict adapter used by MoE flavors. The Attention-config
# builder ``_make_dsv3_attn_config`` is nominally private but is the
# single source of truth for DSv3 MLA config assembly; duplicating it
# would drift as DSv3's config surface evolves.
from torchtitan.models.deepseek_v3 import _make_dsv3_attn_config
from torchtitan.models.deepseek_v3.model import Attention as DSv3MLAAttention
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec


__all__ = [
    "AttnResModel",
    "AttnResTransformerBlock",
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
) -> list[AttnResTransformerBlock.Config]:
    layers: list[AttnResTransformerBlock.Config] = []
    for layer_id in range(n_layers):
        layers.append(
            AttnResTransformerBlock.Config(
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


def _debugmodel_attn_res() -> AttnResModel.Config:
    """Debug model with Block Attention Residuals enabled.

    6 layers / 3 blocks = 2 layers per block, so every even-indexed layer
    is a block start. Intentionally small for unit/smoke testing.
    """
    dim = 256
    n_heads = 16
    n_layers = 6
    num_blocks = 3
    return AttnResModel.Config(
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


def _175m_attn_res(
    num_blocks: int = 6,
    n_layers: int = _175M_N_LAYERS,
    enable_weight_tying: bool = True,
) -> AttnResModel.Config:
    """~175M dense AttnRes-native model with Block AttnRes enabled.

    Parameter count: 174,017,280 total (tied embedding counted once) /
    75,555,072 as reported by torchtitan's size-log under the tied-
    embedding convention (excludes the 98.5M shared embed/output).

    Args:
        num_blocks: Number of attention-residual blocks. Must divide
            ``n_layers``. Paper sweet spot is ``N=8``; default is 6
            (closest divisor below 8 for n_layers=12). N=1 is equivalent
            to standard residuals and is disallowed.
        n_layers: Total transformer layers. Default 12 (original 175M
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
    return AttnResModel.Config(
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


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    """DSv3 depth-scaled init for GroupedExperts w1/w2/w3 weights."""
    return {
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _build_dsv3_attn_res_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    inner_attention=None,
    mask_type: str = "causal",
) -> list[AttnResTransformerBlock.Config]:
    """Build DSv3-shaped layers (MLA attention + mixed dense/MoE FFN)
    with AttnRes wiring on every layer.

    Structural equivalent of ``_build_dsv3_layers`` in
    ``torchtitan/models/deepseek_v3/__init__.py``, but emits
    :class:`AttnResTransformerBlock.Config` and appends the four AttnRes
    fields (pre-attn + pre-MLP pseudo-queries and their norms) to every
    layer. The MLA attention config is built by DSv3's own
    ``_make_dsv3_attn_config`` (reused verbatim) so our MLA layer config
    cannot drift from DSv3's canonical shape.

    Layers ``0..n_dense_layers-1`` get a dense FeedForward; layers
    ``n_dense_layers..n_layers-1`` get a MoE. This matches DSv3's
    first-N-dense-then-MoE convention. AttnRes is orthogonal: every
    layer has the four AttnRes params regardless of FFN type.
    """
    layers: list[AttnResTransformerBlock.Config] = []
    for layer_id in range(n_layers):
        attn_cfg = _make_dsv3_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            inner_attention=inner_attention,
            mask_type=mask_type,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                score_before_experts=score_before_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    param_init=_depth_experts_init(layer_id),
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )

        layers.append(
            AttnResTransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
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


def _dsv3_debugmodel_attn_res() -> AttnResModel.Config:
    """Small DSv3-shaped AttnRes model for unit / smoke testing.

    6 layers (1 dense + 5 MoE), 8 experts, N=3 AttnRes blocks (2 layers
    per block). Dim 256, 16 heads. Same attention + routing shape as
    DSv3's own ``debugmodel``, plus AttnRes on every layer. Meant for
    CPU tests and sub-second GPU smoke runs; not a training target.
    """
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    rope_dim = 64
    num_blocks = 3

    layers = _build_dsv3_attn_res_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=1024,
        moe_hidden_dim=256,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return AttnResModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
        attn_res=AttnResConfig(enabled=True, num_blocks=num_blocks),
        final_attn_res_proj=AttnResProjection.Config(
            dim=dim, param_init=_ATTN_RES_PROJ_INIT
        ),
        final_attn_res_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
    )


def _dsv3_16b_attn_res(num_blocks: int = 9) -> AttnResModel.Config:
    """~16B DSv3 MoE + AttnRes. Training-scale MoE target.

    Mirrors ``deepseek_v3_16b`` shape (27 layers, dim=2048, 64 experts,
    MLA + fine-grained routing) and adds AttnRes. ``n_layers=27`` has
    divisors ``{1, 3, 9, 27}``; N=9 gives 3 layers per block —
    halfway between N=3 (coarse) and N=27 (per-layer, Full-AttnRes).
    Paper prescribes ``N ≈ 8`` at large scale; N=9 is the closest
    divisor of 27 to that target.
    """
    if num_blocks < 2 or 27 % num_blocks != 0:
        raise ValueError(
            f"num_blocks={num_blocks} must be >=2 and divide 27. "
            f"Valid: {sorted(d for d in range(2, 28) if 27 % d == 0)}"
        )

    dim = 2048
    n_layers = 27
    vocab_size = 102400
    n_heads = 16
    rope_dim = 64

    layers = _build_dsv3_attn_res_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=10944,
        moe_hidden_dim=1408,
        num_experts=64,
        num_shared_experts=2,
        router_top_k=6,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
    )
    return AttnResModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
        attn_res=AttnResConfig(enabled=True, num_blocks=num_blocks),
        final_attn_res_proj=AttnResProjection.Config(
            dim=dim, param_init=_ATTN_RES_PROJ_INIT
        ),
        final_attn_res_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
    )


attn_res_configs = {
    "debugmodel_attn_res": _debugmodel_attn_res,
    "175M_attn_res": _175m_attn_res,  # default N=6
    # Dense + GQA ablation flavors over num_blocks. n_layers=12, so
    # divisors are {2, 3, 4, 6, 12}. N=1 is degenerate (= standard
    # residuals).
    "175M_attn_res_n2": partial(_175m_attn_res, num_blocks=2),
    "175M_attn_res_n3": partial(_175m_attn_res, num_blocks=3),
    "175M_attn_res_n4": partial(_175m_attn_res, num_blocks=4),
    "175M_attn_res_n12": partial(_175m_attn_res, num_blocks=12),
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
    "175M_attn_res_L16_n8": partial(
        _175m_attn_res, num_blocks=8, n_layers=16, enable_weight_tying=False
    ),
    # DSv3-shaped MoE + MLA + AttnRes flavors. Architecturally closest
    # open match to Kimi's production design. See _build_dsv3_attn_res_layers.
    "dsv3_debugmodel_attn_res": _dsv3_debugmodel_attn_res,
    "dsv3_16b_attn_res": _dsv3_16b_attn_res,  # N=9 (3 layers/block)
    "dsv3_16b_attn_res_n3": partial(_dsv3_16b_attn_res, num_blocks=3),
    "dsv3_16b_attn_res_n27": partial(_dsv3_16b_attn_res, num_blocks=27),
}


def model_registry(flavor: str) -> ModelSpec:
    """Build a ``ModelSpec`` for ``flavor``.

    Parallelize / post-optimizer / state-dict-adapter are chosen from the
    Config itself:

    - If any layer has ``moe`` set, the flavor is DSv3-shaped → use
      ``parallelize_deepseekv3`` (handles EP + TP + MLA layout),
      ``register_moe_load_balancing_hook`` (load-balance aux grad hook),
      and ``DeepSeekV3StateDictAdapter`` (HF conversion for DSv3 shape).
    - Otherwise it is Llama3-proportioned dense → keep ``parallelize_llama``
      and the Llama3 state-dict adapter.

    The cross-stage caching ``pipelining_fn`` is the same for both because
    it wraps core ``pipeline_llm`` and is agnostic to the model shape.
    """
    config = attn_res_configs[flavor]()
    has_moe = any(layer.moe is not None for layer in config.layers)
    if has_moe:
        parallelize_fn = parallelize_deepseekv3
        post_optimizer_build_fn = register_moe_load_balancing_hook
        state_dict_adapter = DeepSeekV3StateDictAdapter
    else:
        parallelize_fn = parallelize_llama
        post_optimizer_build_fn = None
        state_dict_adapter = Llama3StateDictAdapter
    return ModelSpec(
        name="attn_res",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_fn,
        # Custom pipelining_fn wraps core pipeline_llm with the AttnRes
        # cross-stage caching adapter (opt-in via TORCHTITAN_ATTNRES_CACHE=1).
        # When the flag is unset, this is a thin passthrough over
        # torchtitan.distributed.pipeline_parallel.pipeline_llm.
        pipelining_fn=pipeline_llm_with_cache_adapter,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=post_optimizer_build_fn,
        state_dict_adapter=state_dict_adapter,
    )
