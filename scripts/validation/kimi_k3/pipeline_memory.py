# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment policies for memory-bounded pipeline microbatch accumulation.

Use existing SAC and public FSDP hooks. Keep FP32 gradient accumulation and
communication unchanged; trade extra dense recomputation/all-gathers for memory.
"""
from dataclasses import dataclass, field

from torch.distributed.fsdp import FSDPModule
from torchtitan.distributed.activation_checkpoint import RegionAC


class RecomputeMatmulsSelectiveAC(RegionAC):
    """SelectiveAC's regions plus every dense linear recomputed; communication regions stay saved."""

    @dataclass(kw_only=True, slots=True)
    class Config(RegionAC.Config):
        save_regions: list[str] = field(default_factory=lambda: ["*"])
        recompute_regions: list[str] = field(
            default_factory=lambda: ["*routed_experts.w13.*", "*.linear"]
        )


def enable_backward_resharding(model_parts):
    """Restore resharding after pipeline's per-microbatch no-sync setup.

    The pinned PipelineStage.backward_maybe_with_nosync disables resharding
    immediately before autograd. A backward pre-hook runs after that setup,
    before parameter gradients, and restores only the resharding policy.
    Return removable handles so tests/experiments can restore baseline behavior.
    """
    parts = list(model_parts)
    if not parts or not all(isinstance(m, FSDPModule) for m in parts):
        raise TypeError("Backward resharding requires FSDP pipeline model parts")

    def reshard(module, grad_output):
        module.set_reshard_after_backward(True)

    return [m.register_full_backward_pre_hook(reshard) for m in parts]
