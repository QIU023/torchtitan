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

The short convolution. This was the open blocker. KDA runs a causal depthwise
conv of width ``W = short_conv_kernel_size`` on q, k and v before the
recurrence. Shard the sequence and each rank's first ``W - 1`` outputs get
computed against zero padding instead of the previous rank's tail. fla's
convolution has no CP support, but ``ShortConvolution`` already accepts a
``cache`` -- an ``[N, D, W]`` left-context state built for incremental decoding
-- which is exactly a halo. :func:`conv_with_halo` uses it.

Why the halo is cheap: the conv's support is finite, so rank r needs only rank
r-1's tail, not a scan. One fixed-size exchange, no dependency chain, no
dependence on sequence length. Measured bit-exact (rel 0.0) against the
unsharded convolution at cp2, cp4 and cp8; without the halo the boundary tokens
are ~67% wrong while the sequence-averaged error is only ~5-10%, which is
precisely why this would have survived a loss-curve comparison
(``phase13_k3like_48b_posttrain/conv_halo_probe.py``).

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


def halo_states(state: torch.Tensor, group=None) -> torch.Tensor | None:
    """Shift each rank's conv final state one rank to the right.

    Args:
        state: this rank's ``[N, D, W]`` conv final state, i.e. the left context
            its RIGHT neighbour needs.
        group: the CP process group.

    Returns:
        The preceding rank's state, or None on rank 0 -- a true sequence start,
        where zero left padding is what the unsharded conv also does.

    An all_gather rather than a send/recv pair: the payload is ``[N, D, W]``,
    independent of sequence length, and a ring of point-to-point calls would
    need ordering care for no measurable gain at this size.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return None
    rank = dist.get_rank(group)
    gathered = [torch.empty_like(state) for _ in range(world)]
    dist.all_gather(gathered, state.contiguous(), group=group)
    return None if rank == 0 else gathered[rank - 1]


def conv_with_halo(
    conv,
    x_local: torch.Tensor,
    group=None,
) -> torch.Tensor:
    """Run a ``ShortConvolution`` on a sequence-sharded input, exactly.

    Two passes over the local segment: one to produce the state this rank owes
    its neighbour, one to consume the neighbour's. The first pass discards its
    output, which costs a second depthwise conv -- cheap next to the attention
    and the expert GEMMs, and it keeps the halo entirely inside fla's own
    cache semantics rather than reimplementing the convolution's edge handling.

    Bit-exact against the unsharded convolution; see the module docstring.
    """
    _, my_state = conv(x_local, cache=None, output_final_state=True)
    halo = halo_states(my_state, group)
    # fla updates `cache` IN PLACE, and this buffer is also the neighbour's
    # state in the gathered list, so hand over a copy.
    out, _ = conv(
        x_local,
        cache=None if halo is None else halo.clone(),
        output_final_state=False,
    )
    return out


def build_kcp_context(seq_len_local: int, group, device) -> object:
    """fla CP context for a single evenly-split sequence.

    ``chunk_kda`` needs the GLOBAL cu_seqlens of the packed sequence plus the
    process group; ``build_cp_context`` derives each rank's slice from them.
    """
    from fla.ops.cp.context import build_cp_context

    world = dist.get_world_size(group)
    total = seq_len_local * world
    cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
    return build_cp_context(cu_seqlens, group=group)
