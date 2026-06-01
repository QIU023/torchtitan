# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline-parallel plumbing for Kimi Linear AttnRes.

Reuses Phase 3's :class:`torchtitan.experiments.attention_residual.pipeline_adapter.CrossStageCacheAdapter`
verbatim (the adapter duck-types on ``wrapped.layers`` /
``wrapped._return_only_new_blocks`` / ``wrapped.forward(tokens, blocks)``
and doesn't care whether the backbone is Llama3 or Kimi Linear).

What this module adds:

1. **FQN injection for Kimi module names.** Core
   ``generate_llm_fqn_per_model_part`` hardcodes ``tok_embeddings`` /
   ``output`` (Llama3 convention). Kimi uses ``embed_tokens`` /
   ``lm_head``. We substitute on the result, then extend the last stage
   with ``final_attn_res_proj`` / ``final_attn_res_norm`` if AttnRes.

2. **Kimi-specific AttnRes detection.** attn_res's adapter looks at
   ``stage.submod.config.attn_res.enabled``; Kimi's model exposes
   ``num_blocks`` and ``layers_per_block`` attributes directly.

3. **Custom ``pipelining_fn``** that wires everything together:
   ``pipeline_kimi_linear_with_cache_adapter``. Registered in Kimi's
   ``ModelSpec`` for AttnRes variants.

For multi-node 48B-A3B where PP is mandatory, this is the path that
delivers the cross-stage cache adapter's comm-savings benefit to Kimi
Linear. At single-node FSDP (Phase 4c) ``pipelining_fn=None`` is fine
and this module isn't exercised.
"""

from __future__ import annotations

import math
import os
import warnings

import torch.nn as nn

from torch.distributed.pipelining.schedules import PipelineScheduleSingle
from torchtitan.experiments.attention_residual.layout import (
    _infer_block_layout_tables_from_stages,
)
from torchtitan.experiments.attention_residual.pipeline_adapter import (
    _INTERLEAVED_1F1B_CLASS,
    _install_mb_index_patch,
    _install_step_drop_patch,
    _iter_schedule_stages,
    adapter_enabled,
    CrossStageCacheAdapter,
)


# Kimi-specific FQNs injected into the last PP stage when AttnRes is enabled.
_KIMI_ATTN_RES_LAST_STAGE_FQNS = ("final_attn_res_proj", "final_attn_res_norm")


def _kimi_llm_fqns(
    num_stages: int, num_layers: int,
    input_weight: int = 1, output_weight: int = 1,
) -> list[list[str]]:
    """Kimi-named version of ``generate_llm_fqn_per_model_part``.

    Substitutes ``tok_embeddings``→``embed_tokens`` and
    ``output``→``lm_head``. Keeps the layer distribution logic
    (delegated to core's function, then re-mapped) so any future
    tweaks there apply to us automatically.
    """
    from torchtitan.distributed.pipeline_parallel import (
        generate_llm_fqn_per_model_part,
    )
    raw = generate_llm_fqn_per_model_part(
        num_stages, num_layers, input_weight, output_weight
    )
    rename = {"tok_embeddings": "embed_tokens", "output": "lm_head"}
    return [[rename.get(n, n) for n in stage] for stage in raw]


def _inject_kimi_linear_fqns(model: nn.Module, kwargs: dict) -> None:
    """Populate ``parallelism.module_fqns_per_model_part`` so the PP
    split uses Kimi module names and the last stage includes the
    AttnRes final-aggregation modules.
    """
    if not any(hasattr(model, n) for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS) \
            and not hasattr(model, "embed_tokens"):
        return  # Not a Kimi model; pass through
    parallelism = kwargs.get("parallelism")
    if parallelism is None or parallelism.module_fqns_per_model_part is not None:
        return
    model_config = kwargs.get("model_config")
    pp = kwargs["parallel_dims"].pp
    if pp <= 1 or model_config is None:
        return

    # Layer count: kimi's config stores it at ``num_hidden_layers``.
    num_layers = getattr(model_config, "num_hidden_layers", None)
    if num_layers is None:
        return
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
    else:
        from torchtitan.distributed.pipeline_parallel import get_schedule_class
        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
        stages_per_rank = 1 if issubclass(schedule_class, PipelineScheduleSingle) else 2
        num_virtual_stages = pp * stages_per_rank

    fqns = _kimi_llm_fqns(
        num_virtual_stages, num_layers, input_weight, output_weight
    )
    # Append AttnRes tail modules if present (last stage only).
    extras = [n for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS if hasattr(model, n)]
    if extras:
        fqns[-1].extend(extras)
    parallelism.module_fqns_per_model_part = fqns


def pipeline_kimi_linear_with_cache_adapter(model: nn.Module, **kwargs):
    """``pipelining_fn`` for Kimi Linear (baseline + AttnRes variants).

    Behavior:

    * Always: patch ``parallelism.module_fqns_per_model_part`` to use
      Kimi names and include final AttnRes modules on the last stage,
      then delegate to core ``pipeline_llm`` for the actual PP setup.
    * When ``TORCHTITAN_ATTNRES_CACHE=1`` AND the schedule is
      Interleaved1F1B AND the wrapped model is AttnRes (has
      ``num_blocks`` + ``layers_per_block`` attrs): wrap each stage's
      ``submod`` in ``CrossStageCacheAdapter`` (Phase 3's
      implementation, reused unchanged — it duck-types the wrapped
      model's forward signature).
    * Otherwise: pass through (plain PP, no cache adapter).
    """
    from torchtitan.distributed.pipeline_parallel import pipeline_llm

    _inject_kimi_linear_fqns(model, kwargs)
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )
    passthrough = (pp_schedule, model_parts, has_first_stage, has_last_stage)

    if not adapter_enabled():
        return passthrough

    if _INTERLEAVED_1F1B_CLASS is None or not isinstance(
        pp_schedule, _INTERLEAVED_1F1B_CLASS
    ):
        warnings.warn(
            "Kimi Linear cross-stage caching supports only Interleaved1F1B; "
            "running without the adapter."
        )
        return passthrough

    stages = list(_iter_schedule_stages(pp_schedule))
    parallel_dims = kwargs.get("parallel_dims")
    pp_size = parallel_dims.pp if parallel_dims is not None else len(stages)
    num_stages = pp_size * len(stages)
    stage_to_rank = {s: s % pp_size for s in range(num_stages)}

    # Detect AttnRes by Kimi-specific marker attributes on the wrapped model.
    inner0 = getattr(stages[0], "submod", None)
    num_blocks = getattr(inner0, "num_blocks", None)
    layers_per_block = getattr(inner0, "layers_per_block", None)
    if num_blocks is None or layers_per_block is None:
        warnings.warn(
            "Stage 0 model has no 'num_blocks'/'layers_per_block' — "
            "this is a baseline (non-AttnRes) Kimi Linear run; the "
            "cross-stage cache adapter only applies to AttnRes variants. "
            "Running without the adapter."
        )
        return passthrough

    # Layout tables: same math as attn_res, just with Kimi's layer count.
    model_config = kwargs.get("model_config")
    n_layers_total = getattr(model_config, "num_hidden_layers", None)
    if n_layers_total is None:
        warnings.warn(
            "Cannot determine total layer count; cache adapter falls back to passthrough."
        )
        return passthrough

    try:
        layout_tables = _infer_block_layout_tables_from_stages(
            stages, pp_size=pp_size, num_blocks=num_blocks,
            n_layers=n_layers_total, layers_per_block=layers_per_block,
        )
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(
            f"Failed to build Kimi Linear block-layout tables ({e!r}); "
            "falling back to passthrough."
        )
        return passthrough

    installed_adapters: list[CrossStageCacheAdapter] = []
    for i, stage in enumerate(stages):
        adapter = CrossStageCacheAdapter(
            stage.submod,
            stage_id=stage.stage_index,
            num_stages=num_stages,
            group=getattr(stage, "group", None),
            stage_to_rank=stage_to_rank,
            pp_rank=getattr(stage, "group_rank", None),
            layout_tables=layout_tables,
        )
        stage.submod = adapter
        _install_mb_index_patch(stage, adapter)
        installed_adapters.append(adapter)
        if i < len(model_parts):
            model_parts[i] = adapter

    _install_step_drop_patch(pp_schedule, installed_adapters)

    return pp_schedule, model_parts, has_first_stage, has_last_stage
