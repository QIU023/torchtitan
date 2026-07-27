# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 experiment: KDA + MLA + MoE backbone with Block Attention
Residuals (AttnRes, arXiv:2603.15031), the architecture family Kimi K3
confirmed in production.

Flavors follow ``kimi_linear_<size>_<variant>`` (size from the AttnRes
tech-report Table 2 scaling-law sweep plus the 48B-A3B layout; variant in
{baseline, block_attn_res, full_attn_res}). Trainer-level configuration
lives in :mod:`.config_registry`; architecture-side builders in
:mod:`.model_configs`.

The cross-stage pipeline-parallel cache adapter (``pipeline_adapter.py``)
is private to this experiment by design -- see the AttnRes RFC history.
"""

from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger

# fla-core (triton) is required by the KDA path; guard so environments
# without it (e.g. CPU-only dev boxes) can still import the package and
# fail with a clear error only when a Kimi flavor is requested.
try:
    from torchtitan.experiments.kimi_k3.attn_res_model import (
        KimiAttnResDecoderLayer,
        KimiLinearAttnResModel,
    )
    from torchtitan.experiments.kimi_k3.model import (
        KimiDecoderLayer,
        KimiDeltaAttention,
        KimiLinearConfig,
        KimiLinearModel,
        KimiLinearSpec,
        KimiMLAAttention,
        KimiMLP,
        KimiMoE,
    )
    from torchtitan.experiments.kimi_k3.model_configs import (
        build_kimi_linear_config,
        flavor_names,
        resolve_num_blocks,
        SCALING_LAW_TABLE,
    )
    from torchtitan.experiments.kimi_k3.multimodal_model import (
        KimiLinearMultimodalModel,
        KimiMultimodalConfig,
        KimiVisionProjector,
    )
    from torchtitan.experiments.kimi_k3.parallelize import parallelize_kimi_linear
    from torchtitan.experiments.kimi_k3.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )

    _KIMI_IMPORT_ERROR: ImportError | None = None
except ImportError as _err:
    _KIMI_IMPORT_ERROR = _err

__all__ = [
    "KimiAttnResDecoderLayer",
    "KimiDecoderLayer",
    "KimiDeltaAttention",
    "KimiLinearAttnResModel",
    "KimiLinearConfig",
    "KimiLinearModel",
    "KimiLinearMultimodalModel",
    "KimiLinearSpec",
    "KimiMLAAttention",
    "KimiMLP",
    "KimiMoE",
    "KimiMultimodalConfig",
    "KimiVisionProjector",
    "build_kimi_linear_config",
    "flavor_names",
    "model_registry",
    "resolve_num_blocks",
]


def _parse_flavor(flavor: str) -> tuple[str, str]:
    """Parse ``kimi_linear_<size>_<variant>`` -> (size, variant)."""
    if not flavor.startswith("kimi_linear_"):
        raise ValueError(
            f"Unknown flavor '{flavor}'. Kimi K3 flavors follow "
            "'kimi_linear_<size>_<variant>'; see flavor_names()."
        )
    rest = flavor[len("kimi_linear_") :]
    for variant in ("baseline", "block_attn_res", "full_attn_res"):
        suffix = f"_{variant}"
        if rest.endswith(suffix):
            size = rest[: -len(suffix)]
            return size, variant
    raise ValueError(f"Unknown flavor '{flavor}'.")


def model_registry(flavor: str, attn_backend: str | None = None) -> ModelSpec:
    """Return a :class:`ModelSpec` for a ``kimi_linear_<size>_<variant>``
    flavor. The ``baseline`` variant disables AttnRes (plain backbone);
    the cache-adapter ``pipelining_fn`` is always wired and passes
    through untouched for baseline / pp=1 runs."""
    if _KIMI_IMPORT_ERROR is not None:
        raise ImportError(
            "Kimi K3 flavors require fla-core (KDA kernels)."
        ) from _KIMI_IMPORT_ERROR
    # attn_backend is accepted for registry-interface compatibility
    # (veRL's torchtitan engine passes it): KDA runs on fla kernels and
    # MLA on SDPA here, so backend selection does not apply yet.
    if attn_backend is not None:
        logger.warning(
            "kimi_k3.model_registry ignores attn_backend=%r (KDA=fla, "
            "MLA=SDPA are fixed in this implementation).",
            attn_backend,
        )
    # Graft-variant suffixes (post-train flavors): strip and record.
    gated = False
    lora_rank = None
    base_flavor = flavor
    if base_flavor.endswith("_gated_lora"):
        base_flavor = base_flavor[: -len("_gated_lora")]
        gated = True
        lora_rank = 16
    elif base_flavor.endswith("_gated"):
        base_flavor = base_flavor[: -len("_gated")]
        gated = True
    size, variant = _parse_flavor(base_flavor)
    kimi_config = build_kimi_linear_config(size)
    num_blocks = resolve_num_blocks(size, variant)
    spec_config = KimiLinearSpec(
        kimi_config=kimi_config,
        num_blocks=num_blocks,
        attn_res_gated=gated,
        lora_rank=lora_rank,
    )
    from torchtitan.experiments.kimi_k3.state_dict_adapter import (
        KimiLinearStateDictAdapter,
    )

    return ModelSpec(
        name="kimi_linear",
        flavor=flavor,
        model=spec_config,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )


# Flavor-name dict for registry-discovery consumers (veRL's torchtitan
# engine looks for a module-level ``*_configs`` dict and uses its KEYS
# with ``model_registry``). Values are unused.
kimi_linear_configs: dict[str, None] = {name: None for name in flavor_names()}
