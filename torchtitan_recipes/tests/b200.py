# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.trainer import Trainer


def kimi_k3_debugmodel_mm_fsdp2() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def kimi_k3_debugmodel_mm_fsdp2_ep2() -> Trainer.Config:
    """The multimodal debug model with the experts sharded over the two ranks."""
    config = kimi_k3_debugmodel_mm_fsdp2()
    config.parallelism.expert_parallel_degree = 2
    return config
