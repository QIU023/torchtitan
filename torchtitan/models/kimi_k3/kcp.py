# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KCP: KDA Context Parallelism (report sec 5.1.2).

KDA under context parallelism has two pieces, and only one of them was solved.

The recurrence. K3 does NOT use Ulysses head-sharding for KDA, and does not use
LASP-style state summation either -- plain summation is wrong because the delta
rule applies a token-dependent transition to the incoming state. KCP instead
decomposes each rank's segment into two locally computable fragments, a
cumulative transition and a zero-started state, which compose associatively; a
prefix scan over them recovers each rank's true incoming state in one
fixed-size all-gather, independent of sequence length. fla-core 0.5.1 ships this
(``chunk_kda(cp_context=...)``) and this repo validated it bit-exact, forward
and backward, against a single-rank reference.

The short convolution. KDA runs a causal depthwise conv of width
``W = short_conv_kernel_size`` on q, k and v before the recurrence. Shard the
sequence and each rank's first ``W - 1`` outputs get computed against zero
padding instead of the previous rank's tail. fla ships this too:
``causal_conv1d_cp`` is a real ``autograd.Function`` that exchanges the tail in
the forward and the matching ``dx`` in the backward. Pass
``conv1d_kernel_size`` to ``build_cp_context`` and it is wired for you.

This repo previously carried a hand-rolled halo that fetched the neighbour's
conv state with ``dist.all_gather``. Its forward was bit-exact, but
``all_gather`` is not autograd-aware, so the gradient each rank owed its left
neighbour's tail was silently dropped: rank 0's last ``W - 1`` tokens came out
~60% wrong while every interior token was exact
(``phase13_k3like_48b_posttrain/conv_halo_grad_probe.py``). An error confined
to ``W - 1`` boundary tokens per rank does not move a loss curve, which is why
a forward-only bit-exactness check passed it.

KCP vs the Ulysses path already in this repo: Ulysses all-to-alls the head axis
and gives every rank the full sequence for its head subset, so activation memory
per rank stays O(T/cp x D) only for the projections and the recurrence sees the
whole sequence. KCP keeps the sequence sharded end to end, which is what makes
it composable with a sharded-sequence pipeline and what the 1M-token context
needs. Both are kept: Ulysses is the A/B.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def conv_with_halo(conv, x_local: torch.Tensor, cp_context) -> torch.Tensor:
    """Run a ``ShortConvolution`` on a sequence-sharded input, exactly.

    Thin adapter over fla's ``causal_conv1d_cp``: unpack the depthwise weight
    the way ``ShortConvolution.forward`` does and hand over the CP context,
    which must have been built with ``conv1d_kernel_size`` set.
    """
    from einops import rearrange
    from fla.modules.conv.cp.ops import causal_conv1d_cp

    return causal_conv1d_cp(
        x=x_local,
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        activation=conv.activation,
        cp_context=cp_context,
    )


def build_kcp_context(
    seq_len_local: int, group, device, conv1d_kernel_size: int | None = None
) -> object:
    """fla CP context for a single evenly-split sequence.

    ``chunk_kda`` needs the GLOBAL cu_seqlens of the packed sequence plus the
    process group; ``build_cp_context`` derives each rank's slice from them.
    ``conv1d_kernel_size`` is required by ``causal_conv1d_cp`` and otherwise
    unused, so it is optional here.
    """
    from fla.ops.cp.context import build_cp_context

    world = dist.get_world_size(group)
    total = seq_len_local * world
    cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
    return build_cp_context(
        cu_seqlens, group=group, conv1d_kernel_size=conv1d_kernel_size
    )
