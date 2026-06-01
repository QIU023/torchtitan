# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Attention Residuals (AttnRes).

Implements Block AttnRes from "Attention Residuals" (Kimi Team, 2026),
https://arxiv.org/abs/2603.15031. AttnRes replaces fixed residual accumulation
with softmax attention over preceding layer outputs, using a per-layer learned
pseudo-query vector. Block AttnRes partitions layers into N blocks, applies
standard residuals within a block, and uses attention only across block
boundaries to keep memory and cross-stage communication at O(Nd).

Pseudocode reference: paper Figure 2.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial
from torch.nn import functional as F

from torchtitan.models.common.linear import Linear as _TTLinear
from torchtitan.protocols.module import Module


@dataclass(kw_only=True, slots=True)
class AttnResConfig:
    """Configuration for Block Attention Residuals.

    Attributes:
        enabled: Master switch. When False, the model uses standard residuals
            and all AttnRes code paths are skipped.
        num_blocks: Number of blocks to partition layers into (N in the paper).
            Sweet spot is ~8; N=2,4,8 all perform similarly, N>=16 degrades.
        norm_eps: Epsilon for the RMSNorm applied to keys.
    """

    enabled: bool = False
    num_blocks: int = 8
    norm_eps: float = 1e-5


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Inter-block attention: attend over completed blocks + current partial.

    Follows paper Figure 2. Pseudo-query is ``proj.weight`` (shape [1, D]),
    values are the stacked blocks (including the current partial). Keys are
    RMSNorm-ed values. Softmax over the block axis produces mixing weights.

    Args:
        blocks: List of completed block representations, each [B, T, D].
        partial_block: Current intra-block partial sum [B, T, D].
        proj: Linear(D, 1, bias=False). Its weight vector is the pseudo-query
            w_l. MUST be zero-initialized so softmax weights start uniform.
        norm: RMSNorm over D, applied to keys.

    Returns:
        Aggregated hidden state [B, T, D].
    """
    V = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
    K = norm(V)
    # proj.weight is [1, D]; squeeze to [D] and contract with K's channel dim.
    # Under TP, proj is wrapped with NoParallel, which makes proj.weight a
    # DTensor(Replicate) on the tp mesh dim. The downstream einsum mixes
    # ``query`` with the plain Tensor ``K``, which would raise "mixed Tensor
    # and DTensor". We strip the DTensor wrapping here at the use-site (every
    # TP rank has the full local copy under Replicate placement, so to_local
    # is a no-op data-wise but unwraps the DTensor for the einsum dispatcher).
    #
    # CRITICAL — backward gradient placement on the tp dim. ``K``/``V`` are
    # produced replicated on the tp mesh in the forward, but they carry
    # gradients back through rowwise-sharded ``o_proj`` / ``down_proj`` whose
    # backward emits Partial gradients. So the einsum's gradient w.r.t.
    # ``query`` is a *per-tp-rank partial contribution* that must be
    # all-reduced across the tp dim. A bare ``to_local()`` defaults the
    # backward grad placement to the DTensor's own placement (Replicate on
    # the tp dim), which tells DTensor the gradient is already consistent and
    # SKIPS the all-reduce — leaving ``proj.weight.grad`` with only one rank's
    # share. ``clip_grad_norm_`` then stacks this mis-reduced grad across the
    # (fsdp, tp) mesh and the norm blows up (observed grad_norm 40k-80k under
    # 4D mesh). Mirror ``models/common/moe.py``: request Partial on the tp
    # dim so the backward all-reduces. Non-tp dims (e.g. the fsdp Shard dim
    # under pure 1D FSDP, or the fsdp dim of the 4D mesh) keep their natural
    # placement so the pure-FSDP path is byte-for-byte unchanged.
    weight = proj.weight
    if isinstance(weight, DTensor):
        mesh = weight.device_mesh
        dim_names = mesh.mesh_dim_names
        if dim_names is not None and "tp" in dim_names:
            # Per-mesh-dim grad placements: Partial on tp (force all-reduce
            # of the replicated-compute gradient), unchanged elsewhere.
            grad_placements = [
                Partial() if name == "tp" else weight.placements[i]
                for i, name in enumerate(dim_names)
            ]
            weight = weight.to_local(grad_placements=grad_placements)
        else:
            # No tp dim (pure FSDP / pure DP): default to_local backward
            # placement already matches the param's Shard/Replicate — leave
            # this path exactly as before.
            weight = weight.to_local()
    query = weight.squeeze(0)
    logits = torch.einsum("d,nbtd->nbt", query, K)
    weights = F.softmax(logits, dim=0)
    h = torch.einsum("nbt,nbtd->btd", weights, V)
    return h


class AttnResProjection(_TTLinear):
    """Pseudo-query projection for AttnRes (D -> 1, no bias).

    Inherits from ``torchtitan.models.common.linear.Linear`` (which is
    ``nn.Linear + Module``) so instances satisfy
    ``Float8LinearConverter.verify_module_protocol``. The weight IS the
    per-layer pseudo-query vector ``w_l`` from the paper.
    ``param_init`` must zero-initialize the weight for training stability.

    NOTE: filter via ``filter_fqns`` to keep AttnRes pseudo-queries in
    high precision — the zero-init carrier story relies on small
    deltas accumulating, which rowwise FP8 quantization noise would
    destroy.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int

    def __init__(self, config: Config):
        nn.Linear.__init__(self, config.dim, 1, bias=False)


def stack_blocks(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of [B, T, D] blocks into a [N, B, T, D] tensor.

    Used when crossing a pipeline parallel stage boundary: the list becomes
    a tensor so PipelineStage can send it via P2P send/recv.
    """
    return torch.stack(blocks, dim=0)


def unstack_blocks(blocks_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Inverse of ``stack_blocks``.

    Returns a list of [B, T, D] views into the stacked tensor. Views share
    storage with the input so autograd gradients flow back correctly.
    """
    return [blocks_tensor[i] for i in range(blocks_tensor.shape[0])]
