# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Linear experiment — torchtitan-idiom port of
MoonshotAI/Kimi-Linear (MoE Transformer with KDA + MLA + AttnRes).

See ``README.md`` in this directory for scope, and
``../../phase4/README.md`` for the overall plan.

Phase 4c: ModelSpec registration is live. Use the flavors below via::

    torchrun ... --module kimi_linear --config <flavor>

where ``<flavor>`` is one of the 15 scaling-law flavors exported by
:func:`model_registry` (5 sizes × 3 AttnRes variants). Trainer-level
configuration (optimizer, lr schedule, data) lives in
:mod:`.config_registry`.
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.experiments.kimi_linear.attn_res_model import (
    KimiAttnResDecoderLayer,
    KimiLinearAttnResModel,
)
from torchtitan.experiments.kimi_linear.config_registry import (
    build_kimi_linear_config,
    flavor_names,
    resolve_num_blocks,
    SCALING_LAW_TABLE,
)
from torchtitan.experiments.kimi_linear.model import (
    KimiDecoderLayer,
    KimiDeltaAttention,
    KimiLinearConfig,
    KimiLinearModel,
    KimiLinearSpec,
    KimiMLAAttention,
    KimiMLP,
    KimiMoE,
)
from torchtitan.experiments.kimi_linear.parallelize import parallelize_kimi_linear
from torchtitan.experiments.kimi_linear.pipeline_adapter import (
    pipeline_kimi_linear_with_cache_adapter,
)
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "KimiAttnResDecoderLayer",
    "KimiDecoderLayer",
    "KimiDeltaAttention",
    "KimiLinearAttnResModel",
    "KimiLinearConfig",
    "KimiLinearModel",
    "KimiLinearSpec",
    "KimiMLAAttention",
    "KimiMLP",
    "KimiMoE",
    "build_kimi_linear_config",
    "flavor_names",
    "model_registry",
    "resolve_num_blocks",
]


def _parse_flavor(flavor: str) -> tuple[str, str]:
    """Parse ``kimi_linear_<size>_<variant>`` -> (size, variant)."""
    if not flavor.startswith("kimi_linear_"):
        raise ValueError(
            f"Expected flavor starting with 'kimi_linear_'; got '{flavor}'"
        )
    rest = flavor[len("kimi_linear_"):]
    # variant is one of {baseline, block_attn_res, full_attn_res}
    for variant in ("baseline", "block_attn_res", "full_attn_res"):
        suffix = f"_{variant}"
        if rest.endswith(suffix):
            size = rest[: -len(suffix)]
            return size, variant
    raise ValueError(
        f"Unknown flavor '{flavor}'. Valid: {flavor_names()[:3]} ..."
    )


def model_registry(flavor: str) -> ModelSpec:
    """Return a :class:`ModelSpec` for the requested flavor.

    Flavors: ``kimi_linear_<size>_<variant>`` where ``size`` ∈
    {194m, 241m, 296m, 436m, 528m} and ``variant`` ∈
    {baseline, block_attn_res, full_attn_res}. The ``baseline`` variant
    disables AttnRes (plain Kimi Linear backbone). See
    :func:`flavor_names()` for the full list.
    """
    size, variant = _parse_flavor(flavor)
    kimi_config = build_kimi_linear_config(size)
    num_blocks = resolve_num_blocks(size, variant)

    spec_config = KimiLinearSpec(
        kimi_config=kimi_config,
        num_blocks=num_blocks,
    )

    # PP + cache adapter wiring: always set, even for baseline. When
    # pp=1 the pipelining_fn is never exercised. When pp>1 baseline, the
    # adapter's detection logic sees no num_blocks attr and passes
    # through without wrapping stages; PP still happens via core
    # pipeline_llm, just no cross-stage cache optimization.
    return ModelSpec(
        name="kimi_linear",
        flavor=flavor,
        model=spec_config,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
