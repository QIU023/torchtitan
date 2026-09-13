# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel kernels specialised to Kimi K3's MLA key.

MLA expands one rotary vector per token onto every head before the inner
attention, so the key the generic kernels move carries ``H`` copies of it.
These kernels split the key back into its per-head nope part and the headless
rope slice, move the nope part packed with what the algorithm already moves
(q and v for Ulysses, v for the K/V all-gather), move the rope slice once, and
expand it after the exchange. Input sharding (``cp_shard``) is the generic
kernels'.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K = N + R`` the qk head dim (nope
and rope), ``V`` the v head dim, ``W`` a packed width.
"""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)

__all__ = [
    "MLAKVAllGatherCPFlexInnerAttention",
    "MLAUlyssesCPFlexInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1


def _cp_group() -> dist.ProcessGroup:
    cp_group = spmd_mesh_group(MeshAxisName.CP)
    if cp_group is None:
        raise RuntimeError("CP attention requires an active multi-rank CP mesh axis.")
    return cp_group


def _check_rope_head_dim(rope_head_dim: int) -> int:
    if rope_head_dim <= 0:
        raise ValueError(
            "The packed MLA context-parallel kernels need the rope width; "
            "install them with MLAContextParallelTransform."
        )
    return rope_head_dim


def _split_rope(
    k_THK: torch.Tensor, rope_head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The nope part ``[T, H, N]`` and the headless rope slice ``[T, R]``."""
    nope_head_dim = k_THK.shape[-1] - rope_head_dim
    k_nope_THN, k_rope_THR = k_THK.split([nope_head_dim, rope_head_dim], dim=-1)
    # MLA expands one rope vector per token onto every head; head 0 carries it.
    return k_nope_THN, k_rope_THR[:, 0]


def _expand_rope(k_nope_THN: torch.Tensor, k_rope_TR: torch.Tensor) -> torch.Tensor:
    """Rebuild the expanded key from its nope part and the rope slice."""
    k_rope_THR = k_rope_TR.unsqueeze(1).expand(-1, k_nope_THN.shape[1], -1)
    return torch.cat((k_nope_THN, k_rope_THR), dim=-1)


class MLAKVAllGatherCPFlexInnerAttention(KVAllGatherCPFlexInnerAttention):
    """K/V all-gather for MLA: the nope key and v travel packed in one gather,
    the rope slice in a second, once; q and the mask stay token-sharded."""

    @dataclass(kw_only=True, slots=True)
    class Config(KVAllGatherCPFlexInnerAttention.Config):
        rope_head_dim: int = 0
        """Width of the rope slice at the end of the qk head dim; set per layer
        by ``MLAContextParallelTransform``."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rope_head_dim = _check_rope_head_dim(config.rope_head_dim)

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = _cp_group()
        k_nope_THN, k_rope_TR = _split_rope(k_THK, self.rope_head_dim)
        widths = (k_nope_THN.shape[-1], v_THV.shape[-1])
        packed_THW, k_rope_TR = (
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.R,
                backward_options={"op_dtype": self.reduce_dtype},
            )
            for x in (
                torch.cat((k_nope_THN, v_THV), dim=-1),
                k_rope_TR.contiguous(),
            )
        )
        k_nope_THN, v_THV = packed_THW.split(widths, dim=-1)
        return FlexInnerAttention.forward(
            self, q_THK, _expand_rope(k_nope_THN, k_rope_TR), v_THV, **kwargs
        )


class MLAUlyssesCPFlexInnerAttention(UlyssesCPFlexInnerAttention):
    """Ulysses for MLA: one all-to-all for q, the nope key and v; the rope
    slice is all-gathered along the sequence and expanded on the local heads."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPFlexInnerAttention.Config):
        rope_head_dim: int = 0
        """Width of the rope slice at the end of the qk head dim; set per layer
        by ``MLAContextParallelTransform``."""

        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the rope slice's backward reduce-scatter. The generic kernel
        sums the heads' rope gradients on one rank; here each rank sums its
        heads and the reduce-scatter adds the ranks, so it runs in float32 like
        the K/V all-gather's."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rope_head_dim = _check_rope_head_dim(config.rope_head_dim)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = _cp_group()
        k_nope_THN, k_rope_TR = _split_rope(k_THK, self.rope_head_dim)
        widths = (q_THK.shape[-1], k_nope_THN.shape[-1], v_THV.shape[-1])
        # (T/cp, H, W) -> (T, H/cp, W) in one exchange.
        packed_THW = spmd.redistribute(
            torch.cat((q_THK, k_nope_THN, v_THV), dim=-1),
            cp_group,
            src=spmd.S(_TOKEN_DIM),
            dst=spmd.S(_HEAD_DIM),
        )
        q_THK, k_nope_THN, v_THV = packed_THW.split(widths, dim=-1)
        # (T/cp, R) -> (T, R): the same vector on every head, so it never
        # travels expanded. Its backward is a reduce-scatter.
        k_rope_TR = spmd.redistribute(
            k_rope_TR.contiguous(),
            cp_group,
            src=spmd.S(_TOKEN_DIM),
            dst=spmd.R,
            backward_options={"op_dtype": self.reduce_dtype},
        )
        out_THV = FlexInnerAttention.forward(
            self, q_THK, _expand_rope(k_nope_THN, k_rope_TR), v_THV, **kwargs
        )
        # (T, H/cp, V) -> (T/cp, H, V).
        return spmd.redistribute(
            out_THV,
            cp_group,
            src=spmd.S(_HEAD_DIM),
            dst=spmd.S(_TOKEN_DIM),
        )
