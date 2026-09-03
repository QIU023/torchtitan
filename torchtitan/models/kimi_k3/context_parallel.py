# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism for the KDA layers.

MLA's context parallelism is a declaration (``_set_ulysses_inner_attention``
in ``sharding.py``). The delta rule cannot be: its recurrence hands state from
rank to rank, a sequential dependency no placement pair describes, so the
KDA body keeps the cp axis token-sharded on both sides and runs Attention
Gym's context-parallel recipe inside; this module builds the plan it needs.
"""

import torch.distributed as dist
from attn_gym.linear.context_parallel import ContextParallelPlan

__all__ = ["kcp_plan"]


def kcp_plan(seq_len_local: int, group) -> ContextParallelPlan:
    """attn-gym routing plan for one sequence split into equal contiguous shards.

    Every rank owns ``[rank * L, (rank + 1) * L)`` of one document; the config
    rejects a load balancer under CP so this table is the sharding the trainer
    actually applied. Host-only, so it costs nothing to rebuild per call.
    """
    world = dist.get_world_size(group)
    ranges = [[(r * seq_len_local, (r + 1) * seq_len_local)] for r in range(world)]
    return ContextParallelPlan.from_token_ranges(
        [0, seq_len_local * world], ranges, dist.get_rank(group)
    )
