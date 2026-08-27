# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Leaving DTensor land for the fla-core triton kernels.

KDA calls kernels that do not dispatch through DTensor, so it unwraps at the
kernel call site and re-wraps the result to the layout it received.
"""

from torch.distributed.tensor import DTensor


__all__ = ["to_local_if_dtensor"]


def to_local_if_dtensor(t):
    """Strip DTensor wrapping for fla-core triton kernels.

    fla-core's chunk_kda / fused_kda_gate / ShortConvolution are Triton
    kernels that don't dispatch through DTensor. Under TP, KDA's parameters
    are declared Replicate on the TP axis and the incoming activations are
    DTensors, so the kernel call site strips both, runs on plain tensors
    (each rank computes redundantly under Replicate), and re-wraps the
    result so the declared module boundary composes.

    isinstance(t, DTensor) is the safe check that dynamo's fake-tensor
    mode honors (``hasattr(t, "to_local")`` is unreliable: dynamo's
    type tracking can elide attribute lookups on DTensor parameters).

    ``grad_placements`` is stated explicitly rather than left to default.
    It is the forward placement, which is also what the default would pick --
    and that is the right answer only because every rank does the SAME work
    with the unwrapped value, so each local gradient already IS the full
    gradient of its shard.
    """
    if isinstance(t, DTensor):
        return t.to_local(grad_placements=list(t.placements))
    return t
