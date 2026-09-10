# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking


def kimi_k3_debugmodel_mm() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def kimi_k3_debugmodel_pp4_vp4() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    # 16 stages: one layer per stage from layer 5 on, the head alone on the last.
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [
        ["vision_encoder", "tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3", "layers.4"],
        *[[f"layers.{i}"] for i in range(5, 17)],
        ["norm", "lm_head", "output_res_proj", "output_res_norm"],
    ]
    return config


def kimi_k3_debugmodel_mm_allgather_kv_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.kimi_k3.kda import InnerKDA

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer.load_balancer_type = "headtail"
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention={
                    FlexInnerAttention.Config: KVAllGatherCPFlexInnerAttention,
                    InnerKDA.Config: ContextParallelInnerKDA,
                }
            ),
        ],
    )


def kimi_k3_debugmodel_mm_ulysses_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.cp_attention import UlyssesCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.kimi_k3.kda import InnerKDA

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer.load_balancer_type = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention={
                    FlexInnerAttention.Config: UlyssesCPFlexInnerAttention,
                    InnerKDA.Config: ContextParallelInnerKDA,
                }
            ),
        ],
    )


def kimi_k3_debugmodel_pp8_vp4_vit_dep() -> Trainer.Config:
    # pp8 x vp4 with the vision tower and the embedding on a stage of their own
    # (vit_dep): they take one of the 32 stages, and the 33 layers, the head and
    # the AttnRes aggregation spread over the other 31, spelled out like the
    # pp8 x vp4 split.
    import dataclasses
    from functools import partial

    from torchtitan.models.kimi_k3.parallelize import (
        kimi_k3_module_fqns_per_model_part,
        pipeline_kimi_k3,
    )

    config = kimi_k3_debugmodel_pp8_vp4()
    text = kimi_k3_module_fqns_per_model_part(8 * 4 - 1, 33, 0, first_stage_modules=())
    config.parallelism.module_fqns_per_model_part = [
        ["tok_embeddings", "vision_encoder"]
    ] + [[n for n in stage if n != "tok_embeddings"] for stage in text]
    assert config.model_spec is not None
    config.model_spec = dataclasses.replace(
        config.model_spec, pipelining_fn=partial(pipeline_kimi_k3, vit_dep=True)
    )
    return config
