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
from torch.nn import functional as F

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


@torch.compiler.disable(recursive=True)
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
    # Under FSDP+TP, inputs may mix plain torch.Tensor (PP send/recv
    # boundary drops DTensor placement) and DTensor (layer outputs in
    # the same forward). We force the entire AttnRes computation to
    # local tensors: the math is per-token and per-rank-replicated under
    # our TP plan (proj.weight Replicate on TP, layer outputs Replicate
    # after RowwiseParallel all-reduce), so doing it on local tensors is
    # bit-identical and side-steps the DTensor-stack / DTensor-norm /
    # DTensor-einsum mixed-type bugs.
    #
    # We also inline the RMSNorm with a local-tensor weight rather than
    # calling ``norm(V)`` directly, because ``norm.weight`` is itself a
    # DTensor under our TP plan and ``rms_norm(V_local, weight=DTensor)``
    # raises "aten.bmm got mixed Tensor and DTensor". The eps and weight
    # values are still pulled from the passed-in module so callers don't
    # have to thread them separately.
    def _to_local(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if hasattr(t, "to_local") else t

    blocks_local = [_to_local(b) for b in blocks]
    partial_local = _to_local(partial_block)
    V = torch.stack(blocks_local + [partial_local], dim=0)  # [N+1, B, T, D]
    norm_weight = _to_local(norm.weight)
    norm_eps = norm.eps if hasattr(norm, "eps") else 1e-5
    K = F.rms_norm(V, normalized_shape=(V.shape[-1],),
                   weight=norm_weight, eps=norm_eps)
    # proj.weight is [1, D]; squeeze to [D] and contract with K's channel dim.
    query = _to_local(proj.weight).squeeze(0)
    logits = torch.einsum("d,nbtd->nbt", query, K)
    weights = F.softmax(logits, dim=0)
    h = torch.einsum("nbt,nbtd->btd", weights, V)
    return h


class AttnResProjection(nn.Linear, Module):
    """Pseudo-query projection for AttnRes (D -> 1, no bias).

    Thin Linear subclass that plugs into torchtitan's Module protocol.
    Its weight IS the per-layer pseudo-query vector ``w_l`` from the paper.
    ``param_init`` must zero-initialize the weight for training stability.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int

    def __init__(self, config: Config):
        super().__init__(config.dim, 1, bias=False)


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
