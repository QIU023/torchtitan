# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline-parallel plumbing for Qwen3-VL.

This module installs a model-specific ``pipelining_fn`` for the Qwen3-VL
ModelSpec. The convention (matches Megatron and the rest of the VLM PP
ecosystem):

* Vision encoder is **pinned to PP stage 0** together with ``tok_embeddings``
  and the first ``max(deepstack_visual_indices) + 1`` decoder layers. This
  guarantees that vision masked_scatter and DeepStack injection all happen
  BEFORE the first PP send, so downstream stages receive already-fused
  hidden states and never see vision-specific tensors.
* Stages 1..N-1 own contiguous slices of the remaining decoder layers.
* The last stage owns ``norm`` + ``output``.

For Qwen3-VL variants the DeepStack cut locations are:

  * ``debugmodel``: ``deepstack_visual_indices=[1, 2, 3]`` → cut after layer 3
  * ``debugmodel_moe``: ``deepstack_visual_indices=[1, 2, 3]`` → cut after layer 3
    (caveat: only 1 LM layer; not PP-able with >1 stage)
  * ``2B``:        ``deepstack_visual_indices=[5, 11, 17]``  → cut after layer 17
  * ``8B``:        ``deepstack_visual_indices=[8, 16, 24]``  → cut after layer 24
  * ``30B-A3B``:   ``deepstack_visual_indices=[8, 16, 24]``  → cut after layer 24
  * ``235B-A22B``: ``deepstack_visual_indices=[8, 16, 24]``  → cut after layer 24

PP combined with Context Parallel is **not** supported yet (Megatron has
the same restriction); the model's ``parallelize_qwen3_vl`` raises
``NotImplementedError`` when ``cp_enabled``. This file inherits that
restriction implicitly via the same ``parallelize_fn`` callback.

This module uses pure ``torch.distributed.pipelining`` primitives
(``PipelineStage``, ``Schedule1F1B``, etc.) reused via
``torchtitan.distributed.pipeline_parallel.pipeline_module_split`` /
``build_pipeline_schedule``. No custom collectives.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import (
    build_pipeline_schedule,
    generate_vlm_fqn_per_model_part,
    pipeline_module_split,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.tools.logging import logger


__all__ = ["pipeline_qwen3_vl"]


def _resolve_last_vision_consumer_layer(model_config: BaseModel.Config) -> int:
    """Find the LM layer index past which no vision/DeepStack feature is consumed.

    For Qwen3-VL, DeepStack injects intermediate vision-encoder features
    into the LM at the layer indices stored on the vision encoder config
    as ``deepstack_visual_indices``. The PP cut must be AFTER the max of
    those indices so the scatter happens on stage 0 before the first send.

    Raises:
        AttributeError: if ``model_config.vision_encoder.deepstack_visual_indices``
            is missing (i.e. this is not a Qwen3-VL config).
    """
    vision_cfg = getattr(model_config, "vision_encoder", None)
    if vision_cfg is None:
        raise AttributeError(
            "model_config has no ``vision_encoder`` attribute; "
            "pipeline_qwen3_vl only supports Qwen3-VL configs."
        )
    indices = getattr(vision_cfg, "deepstack_visual_indices", None)
    if not indices:
        raise AttributeError(
            "vision_encoder config has no ``deepstack_visual_indices``."
        )
    return int(max(indices))


def _compute_num_virtual_stages(
    *,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
    num_layers: int,
    input_weight: int,
    output_weight: int,
) -> int:
    """Match ``pipeline_llm`` virtual-stage math for VLM cuts."""
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be "
                f"divisible by pipeline_parallel_degree ({parallel_dims.pp}). "
                f"Model has {num_layers} layers with "
                f"pipeline_parallel_layers_per_stage={layers_per_stage}."
            )
        stages_per_rank = num_virtual_stages // parallel_dims.pp
        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single-stage schedule requires exactly 1 stage per rank; "
                f"got {stages_per_rank}."
            )
        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank; "
                f"got {stages_per_rank}."
            )
        return num_virtual_stages
    # Default: 1 stage per rank (single-stage) or 2 stages per rank (multi).
    stages_per_rank = 1 if is_single_stage_schedule else 2
    return parallel_dims.pp * stages_per_rank


def pipeline_qwen3_vl(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    """``pipelining_fn`` for Qwen3-VL.

    Implements the "vision-pinned stage 0" cut described in this module's
    docstring. Stage 0 owns ``vision_encoder`` + ``tok_embeddings`` + the
    first ``max(deepstack_visual_indices) + 1`` decoder layers; downstream
    stages own contiguous slices of the remaining decoder layers; the last
    stage owns ``norm`` + ``output``.

    Raises:
        NotImplementedError: if Context Parallel is requested alongside PP
            for now (Megatron parity). Combined PP+CP is its own follow-up.
    """
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Pipeline Parallel + Context Parallel is not yet supported for "
            "Qwen3-VL. Set context_parallel_degree=1 or "
            "pipeline_parallel_degree=1."
        )

    pp_mesh = parallel_dims.get_mesh("pp")

    if not hasattr(model_config, "layers"):
        raise ValueError("Qwen3-VL model_config must expose ``layers``.")
    num_layers = len(model_config.layers)

    last_vision_consumer_layer = _resolve_last_vision_consumer_layer(model_config)
    if last_vision_consumer_layer >= num_layers:
        # Shouldn't happen with a sane config, but guard anyway.
        raise ValueError(
            f"max(deepstack_visual_indices)={last_vision_consumer_layer} is "
            f">= num_layers={num_layers}; PP cut would have no LM layers "
            f"left for downstream stages."
        )

    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    num_virtual_stages = _compute_num_virtual_stages(
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        num_layers=num_layers,
        input_weight=input_weight,
        output_weight=output_weight,
    )

    # Honor any user-provided FQN-per-stage override exactly like pipeline_llm.
    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_vlm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            last_vision_consumer_layer=last_vision_consumer_layer,
            input_weight=input_weight,
            output_weight=output_weight,
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Qwen3-VL PP stage {i}: {stage_ms}")

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # Apply SPMD parallelisms (TP / FSDP / AC / compile) per chunk via the
    # model's parallelize_fn — same pattern as pipeline_llm.
    for i, m in enumerate(model_parts):
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        # NOTE: refresh ``stage.submod`` in case ``m`` was wrapped/recompiled.
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=loss_fn,
    )

    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage
