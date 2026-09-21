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
