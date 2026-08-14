# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for the upstream K3 model.

Their registry sets ``pipelining_fn=None``, so PP cannot run on that tree at all.
Two things are needed and only one of them is new work.

The carrier. `KimiK3Model.forward` rebuilds `block_residual_TND` as a local, so a
pipelined run restarts the attention residual on every stage -- measured at 1.321
relative error with nothing raised. `pp_model.KimiK3PipelineModel` puts it in the
signature; two stages built from it reproduce the unsplit model bitwise.

The rest. Core's `pipeline_vlm` already handles a model whose `vision_encoder` is a
direct child, which is exactly this model's shape (ours needed
`_unwrap_multimodal_for_pp` because our vision tower sits in a wrapper). And
`CrossStageCacheAdapter` duck-types the wrapped model's forward, so it is reused
unchanged rather than ported -- it was verified bitwise against the naive path on
twelve cells after the carrier migration.

What differs from our `pipelining_fn` is the stage FQN names: core's generator emits
`tok_embeddings` and `output`, this model has `tok_embeddings` and `lm_head`, so only
the second needs renaming (ours needed both).
"""

from __future__ import annotations

import torch.nn as nn

# The tail modules that must land on the stage owning lm_head: the output
# attention residual runs once over the whole block stack, so a stage that
# holds only some of them would aggregate a partial stack.
_LAST_STAGE_FQNS = ("output_res_norm", "output_res_proj", "norm", "lm_head")


def _stage_fqns(num_stages: int, num_layers: int) -> list[list[str]]:
    """Core's LLM stage split, with this model's names."""
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part as generate,
    )

    raw = generate(num_stages, num_layers)
    parts = [[n for n in stage if n != "output"] for stage in raw]
    for name in _LAST_STAGE_FQNS:
        if name not in parts[-1]:
            parts[-1].append(name)
    return parts


def pipeline_kimi_k3_up(model: nn.Module, **kwargs):
    """``pipelining_fn`` for the upstream K3 model.

    Re-classes the model to the carrier-threading subclass. That subclass adds a
    forward override and no state, so re-classing is equivalent to having built it
    -- and it keeps the change out of their config tree, which is what makes a
    rebase onto the upstream PR cheap.
    """
    from torchtitan.distributed.pipeline_parallel import pipeline_vlm
    from torchtitan.models.kimi_k3_up.model import KimiK3Model
    from torchtitan.models.kimi_k3_up.pp_model import KimiK3PipelineModel

    if isinstance(model, KimiK3Model) and not isinstance(model, KimiK3PipelineModel):
        model.__class__ = KimiK3PipelineModel

    parallelism = kwargs.get("parallelism")
    if parallelism is not None and parallelism.module_fqns_per_model_part is None:
        # Virtual stages come from layers_per_stage when it is set, which is how
        # core derives the count; there is no separate virtual-stage field.
        num_layers = len(model.layers)
        per_stage = parallelism.pipeline_parallel_layers_per_stage
        num_stages = (
            -(-num_layers // per_stage)
            if per_stage
            else parallelism.pipeline_parallel_degree
        )
        parallelism.module_fqns_per_model_part = _stage_fqns(num_stages, num_layers)

    return pipeline_vlm(model, **kwargs)
