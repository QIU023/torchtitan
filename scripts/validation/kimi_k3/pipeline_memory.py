# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment policies for memory-bounded pipeline microbatch accumulation.

Use existing SAC and public FSDP hooks. Keep FP32 gradient accumulation and
communication unchanged; trade extra dense recomputation/all-gathers for memory.
"""
from dataclasses import dataclass

import torch
from torch.distributed.fsdp import FSDPModule
from torchtitan.distributed.activation_checkpoint import SelectiveAC


class RecomputeMatmulsSelectiveAC(SelectiveAC):
    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        pass

    def get_save_ops(self):
        return super().get_save_ops() - {
            torch.ops.aten.mm.default,
            torch.ops.aten.mm.dtype,
            torch.ops.aten.bmm.default,
            torch.ops.aten.linear.default,
        }


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
