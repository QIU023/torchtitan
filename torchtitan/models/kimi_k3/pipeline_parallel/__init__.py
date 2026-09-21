# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables."""

import dataclasses
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
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
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
_KIMI_K3_VIT_DEP_STAGE_FQNS = ("tok_embeddings", "vision_encoder")


def vit_dep_split(
    model: nn.Module, *, parallel_dims, parallelism, model_config
) -> list[list[str]]:
    """The split with the vision tower and the embedding on the first stage alone and
    the layers over the remaining stages; the stage count the schedule sees is unchanged."""
    if getattr(model, "vision_encoder", None) is None:
        raise ValueError(
            "vit_dep gives the vision tower a pipeline stage of its own, "
            "and this model has no vision encoder."
        )
    num_stages, num_layers, _, output_weight = _get_pipeline_metadata(
        parallel_dims, parallelism, model_config
    )
    if num_stages < 2:
        raise ValueError(
            "vit_dep needs at least two pipeline stages: one for the tower and "
            "the embedding, one for the layers."
        )
    # The embedding rides with the tower: the splice needs the token ids, which
    # only the first stage receives; the text split carries no embedding.
    text = _generate_llm_fqn_per_model_part(
        num_stages - 1, num_layers, 0, output_weight
    )
    text[-1].extend(
        n for n in _KIMI_K3_LAST_STAGE_FQNS if getattr(model, n, None) is not None
    )
    return [list(_KIMI_K3_VIT_DEP_STAGE_FQNS)] + [
        [n for n in stage if n != "tok_embeddings"] for stage in text
    ]


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
    vit_dep: bool = False,
    vit_prefetch: int = 0,
    vit_bubble: bool = False,
    vit_bubble_cost_ratio: float = 1.0,
    vit_bubble_max_pending: int = 0,
    **kwargs,
):
    """pipelining_fn for Kimi K3; with attn_res_cache a hop carries only the blocks the
    receiving rank lacks, without it the whole stack, and every rank must agree;
    attn_res_cache_offload parks the stored blocks on pinned host memory; pp_balance names
    the PP ranks that park autograd's saved tensors on a peer through the Mooncake Transfer Engine;
    vit_dep gives the vision tower and the embedding the first stage alone, with the encodes run
    vit_prefetch micro-batches ahead or, with vit_bubble, in the schedule's idle intervals."""
    parallelism = kwargs["parallelism"]
    configured = parallelism.pipeline_parallel_module_fqns_per_model_part
    if vit_dep and configured is None:
        kwargs["parallelism"] = dataclasses.replace(
            parallelism,
            pipeline_parallel_module_fqns_per_model_part=vit_dep_split(
                model,
                parallel_dims=kwargs["parallel_dims"],
                parallelism=parallelism,
                model_config=kwargs["model_config"],
            ),
        )
    elif vit_dep and set(configured[0]) != set(_KIMI_K3_VIT_DEP_STAGE_FQNS):
        raise ValueError(
            "vit_dep puts the vision tower and the embedding on the first stage "
            f"alone; the configured split starts with {configured[0]}."
        )
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
    if vit_prefetch or vit_bubble:
        if not vit_dep:
            raise ValueError(
                "vit_prefetch and vit_bubble place the tower's encodes around the "
                "pipeline's actions, which needs the tower on its own stage: set "
                "vit_dep."
            )
        if vit_prefetch and vit_bubble:
            raise ValueError(
                "vit_prefetch and vit_bubble are alternatives; set exactly one."
            )
        _install_vision_dep(
            pp_schedule,
            stages,
            rank=kwargs["parallel_dims"].get_mesh("pp").get_local_rank(),
            pp_size=kwargs["parallel_dims"].pp,
            prefetch=int(vit_prefetch),
            bubble=bool(vit_bubble),
            cost_ratio=float(vit_bubble_cost_ratio),
            max_pending=int(vit_bubble_max_pending),
        )
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


def _install_vision_dep(
    pp_schedule,
    stages: list[AttnResPipelineStage],
    *,
    rank: int,
    pp_size: int,
    prefetch: int,
    bubble: bool,
    cost_ratio: float,
    max_pending: int,
) -> None:
    """Wire the tower stage's encodes to the schedule (report sec 5.2.3).

    The stage holding the tower serves each micro-batch's forward from a
    per-step feature cache when the encode already ran, and the cache is
    filled either ahead of the consumer (``prefetch`` micro-batches ahead) or
    in the schedule's idle intervals (``bubble``: the plan is read off the
    schedule's own action order, so every rank derives the same placements).
    A rank without the tower stage is left alone.
    """
    from torchtitan.models.kimi_k3.dep_bubble_backward import (
        cut_for_deferred_backward,
        GradQueue,
        install_backward_slots,
    )
    from torchtitan.models.kimi_k3.dep_bubble_plan import plan_for_rank
    from torchtitan.models.kimi_k3.dep_bubble_runtime import install_bubble_runtime
    from torchtitan.models.kimi_k3.vit_prefetch import VisionPrefetcher

    tower_stages = [
        s for s in stages if getattr(s.submod, "vision_encoder", None) is not None
    ]
    if not tower_stages:
        return
    if len(tower_stages) != 1:
        raise RuntimeError(
            f"rank {rank} holds {len(tower_stages)} stages with a vision tower; "
            "vit_dep places it on one."
        )
    tower_stage = tower_stages[0]
    module = tower_stage.submod
    prefetcher = VisionPrefetcher(module)
    queue = GradQueue(max_pending=max_pending) if bubble else None
    # FSDP2 initializes its state lazily in the root module's first forward; an
    # encode issued before that makes the tower a root of its own and the
    # stage's forward then refuses. So the first step runs its encodes inline,
    # and the run-ahead or the plan starts with the second.
    warm = {"done": False}

    inner_forward = tower_stage.forward_one_chunk

    def forward_one_chunk(fwd_chunk_id, args, kwargs=None, *rest, **more):
        feats = prefetcher.take(int(fwd_chunk_id))
        if feats is not None:
            feats = feats[0] if isinstance(feats, (list, tuple)) else feats
            if queue is not None:
                feats = cut_for_deferred_backward(feats, queue, int(fwd_chunk_id))
            kwargs = {**(kwargs or {}), "vision_embeds": feats}
        out = inner_forward(fwd_chunk_id, args, kwargs, *rest, **more)
        warm["done"] = True
        if prefetch:
            prefetcher.advance(int(fwd_chunk_id), prefetch)
        return out

    tower_stage.forward_one_chunk = forward_one_chunk  # type: ignore[method-assign]

    if bubble:
        pipeline_order = getattr(pp_schedule, "pipeline_order", None)
        if not pipeline_order:
            raise ValueError(
                "vit_bubble reads the schedule's action order, which only the looped "
                "schedules (Interleaved1F1B and kin) expose; use vit_prefetch with a "
                "single-stage schedule."
            )

        def plan_for_step():
            n = prefetcher._num_mbs
            if n == 0 or not warm["done"]:
                return None
            return plan_for_rank(
                pipeline_order[rank],
                rank=rank,
                vision_microbatches=n,
                cost_ratio=cost_ratio,
                upfront=min(pp_size, n),
                vision_stage=tower_stage.stage_index,
            )

        def encode_now(microbatches):
            for mb in microbatches:
                prefetcher.ensure_sync(mb)

        # Installed before the step wrapper below, so begin_step runs first
        # and the plan sees this step's micro-batch count.
        install_bubble_runtime(
            pp_schedule,
            plan_for_step=plan_for_step,
            encode_now=encode_now,
            upfront_encode=encode_now,
        )
        assert queue is not None
        install_backward_slots(pp_schedule, queue)

    original_step = pp_schedule.step

    def step(*args, **kwargs):
        prefetcher.begin_step(kwargs.get("kwarg_mbs"))
        if prefetch and warm["done"]:
            for mb in range(prefetch):
                prefetcher.ensure(mb)
        return original_step(*args, **kwargs)

    pp_schedule.step = step  # type: ignore[method-assign]

    if not bubble:
        logger.info(
            "DEP vision prefetch installed: depth=%d on stage %d",
            prefetch,
            tower_stage.stage_index,
        )
        return
    logger.info(
        "DEP bubble runtime installed on stage %d (cost ratio %.2f, max pending %d)",
        tower_stage.stage_index,
        cost_ratio,
        max_pending,
    )
