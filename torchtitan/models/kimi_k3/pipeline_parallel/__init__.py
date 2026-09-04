# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables."""

import logging
from typing import Any

import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

from torchtitan.distributed.pipeline_parallel import (
    _neighbor_p2p_stage_class,
    _NeighborP2PTransportMixin,
    pipeline_with_first_last_stage_modules,
)

from ..pp_balance import install_pp_balance, PPBalanceKnobs
from .layout import infer_block_layout_tables, layer_to_stage_from_split
from .stage import AttnResPipelineStage, PPRankLocalCache

__all__ = ["pipeline_kimi_k3"]

logger = logging.getLogger(__name__)


_KIMI_K3_FIRST_STAGE_FQNS = ("vision_encoder",)
_KIMI_K3_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


def _as_attn_res_stage(stage: _PipelineStageBase) -> AttnResPipelineStage:
    assert isinstance(stage, PipelineStage)
    # Keep the opt-in neighbor transport if core composed it into the stage.
    stage_class: type[AttnResPipelineStage] = AttnResPipelineStage
    extra: dict[str, Any] = {}
    if isinstance(stage, _NeighborP2PTransportMixin):
        stage_class = _neighbor_p2p_stage_class(AttnResPipelineStage)
        extra["transport"] = stage._transport
    rebuilt = stage_class(
        stage.submod,
        stage.stage_index,
        stage.num_stages,
        stage.device,
        group=stage.group,
        dw_builder=stage.dw_builder,
        get_mesh=stage._mesh_cache._get_mesh_cb,
        **extra,
    )
    # The schedule wrote its stage-to-rank map onto the stage it was handed.
    rebuilt.stage_index_to_group_rank = stage.stage_index_to_group_rank
    return rebuilt


def _swap_in_attn_res_stages(
    schedule: _PipelineSchedule,
) -> list[AttnResPipelineStage]:
    if isinstance(schedule, PipelineScheduleSingle):
        rebuilt = _as_attn_res_stage(schedule._stage)
        schedule._stage = rebuilt
        return [rebuilt]
    if isinstance(schedule, PipelineScheduleMulti):
        rebuilt_stages = [_as_attn_res_stage(s) for s in schedule._stages]
        held: list[_PipelineStageBase] = list(rebuilt_stages)
        schedule._stages = held
        return rebuilt_stages
    raise RuntimeError(f"Unexpected pipeline schedule class {type(schedule).__name__}.")


def pipeline_kimi_k3(
    model: nn.Module,
    *,
    attn_res_cache: bool = True,
    attn_res_cache_offload: bool = False,
    pp_balance: PPBalanceKnobs | None = None,
    **kwargs,
):
    """pipelining_fn for Kimi K3; with attn_res_cache a hop carries only the blocks the
    receiving rank lacks, without it the whole stack, and every rank must agree;
    attn_res_cache_offload parks the stored blocks on pinned host memory; pp_balance names
    the PP ranks that park autograd's saved tensors on a peer through the Mooncake Transfer Engine."""
    (
        pp_schedule,
        model_parts,
        has_first_stage,
        has_last_stage,
        split,
    ) = pipeline_with_first_last_stage_modules(
        model,
        first_stage_module_fqns=_KIMI_K3_FIRST_STAGE_FQNS,
        last_stage_module_fqns=_KIMI_K3_LAST_STAGE_FQNS,
        return_split=True,
        **kwargs,
    )

    stages = _swap_in_attn_res_stages(pp_schedule)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    num_blocks = -(-n_layers // layers_per_block)
    layer_to_stage = layer_to_stage_from_split(split)
    layout = infer_block_layout_tables(
        stage_to_rank=dict(stages[0].stage_index_to_group_rank),
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
        cache=attn_res_cache,
    )
    store = PPRankLocalCache(offload=attn_res_cache and attn_res_cache_offload)
    for stage in stages:
        stage.set_routing(layout, store)
    if pp_balance is not None and pp_balance.pp_balance_source_ranks:
        # The engine owns the registered pool and staging buffers; it lives as
        # long as the schedule does.
        # pyrefly: ignore[missing-attribute]
        pp_schedule._pp_balance_engine = install_pp_balance(
            pp_schedule, stages[0].group, pp_balance
        )
    logger.info(
        "Kimi K3 pipeline: %d stage(s) on this rank %s, block transport %s",
        len(stages),
        [s.stage_index for s in stages],
        "delta with rank store"
        + (" on pinned host memory" if attn_res_cache_offload else "")
        if attn_res_cache
        else "whole stack every hop",
    )
    return pp_schedule, model_parts, has_first_stage, has_last_stage
