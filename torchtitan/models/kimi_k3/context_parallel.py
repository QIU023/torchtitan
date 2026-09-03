# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel kernels for the Kimi K3 attention layers.

Each kernel owns its CP collectives: the inner-attention ShardingConfig keeps
the cp axis token-sharded on both sides of the local_map boundary and the
kernel's ``forward`` issues whatever exchange its algorithm needs.

  - MLA runs Ulysses: one all-to-all trades the token shard for a head shard on
    the way in, the kernel attends over the full sequence for its head subset,
    and the output trades back.
  - KDA runs Attention Gym's context-parallel recipe (KCP): the sequence stays
    sharded end to end and the recurrence hands state from rank to rank, a
    sequential dependency no placement pair describes.

``ContextParallelKernel`` and the field-preserving kernel swap are copied from
pytorch/torchtitan PR 4322 (``torchtitan/models/common/cp_attention.py`` at
ed6dba931b), where every CP implementation is to follow this shape.
TODO: delete the copies and import from ``torchtitan.models.common`` once that
PR lands.

Tensor suffix: ``TNH`` = tokens, heads, head dimension.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import spmd_types as spmd

import torch
import torch.distributed as dist
from attn_gym.linear.context_parallel import ContextParallelPlan

from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.attention import FlexAttention
from torchtitan.protocols.module import Module

from .kda import InnerKDA

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model

__all__ = [
    "ContextParallelKernel",
    "ContextParallelInnerKDA",
    "UlyssesCPFlexAttention",
    "kcp_plan",
    "use_kimi_k3_cp_kernels",
]

_SEQ_DIM = 0
_HEAD_DIM = 1


# Copied from pytorch/torchtitan PR 4322 at ed6dba931b. Pending deletion.
class ContextParallelKernel:
    """Mixin for attention kernels that own their CP collectives."""

    @property
    def cp_group(self) -> dist.ProcessGroup:
        """Return the active multi-rank CP process group."""
        mesh = current_spmd_mesh()
        if mesh is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an active SPMD mesh context."
            )
        mesh_axis_names = mesh.mesh_dim_names or ()
        cp_group = mesh.get_group("cp") if "cp" in mesh_axis_names else None
        if cp_group is None or cp_group.size() == 1:
            raise RuntimeError(f"{type(self).__name__} requires an active CP mesh.")
        return cp_group


# Copied from pytorch/torchtitan PR 4322 at ed6dba931b (``use_cp_kernel``),
# minus the model traversal: K3 picks a kernel per layer kind. Pending deletion.
def _swap_kernel(existing: Module.Config, kernel: type[Module]) -> Module.Config:
    """Replace a kernel config with ``kernel``'s, preserving its fields."""
    config_cls = kernel.Config
    if not issubclass(kernel, ContextParallelKernel):
        raise ValueError(f"{kernel.__qualname__} must inherit ContextParallelKernel.")
    if not issubclass(config_cls, type(existing)):
        raise ValueError(
            f"{kernel.__qualname__}.Config must inherit "
            f"{type(existing).__qualname__}."
        )
    return config_cls(**{f.name: getattr(existing, f.name) for f in fields(existing)})


class UlyssesCPFlexAttention(ContextParallelKernel, FlexAttention):
    """FlexAttention over the full sequence for this rank's head subset.

    q/k/v arrive token-sharded on the cp axis. One all-to-all per tensor
    trades that shard for a head shard, the kernel runs over the full sequence
    with the whole packed-document mask (``preprocess_inputs`` keeps the masks
    out of the CP input sharding), and the output trades back. The heads a
    rank sees are its TP slice split again by cp, so ``n_heads`` must divide
    by tp x cp.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        pass

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        group = self.cp_group
        q_TNH, k_TNH, v_TNH = (
            _all_to_all(x, group, src_dim=_SEQ_DIM, dst_dim=_HEAD_DIM)
            for x in (q_TNH, k_TNH, v_TNH)
        )
        out_TNH = super().forward(q_TNH, k_TNH, v_TNH, **kwargs)
        return _all_to_all(out_TNH, group, src_dim=_HEAD_DIM, dst_dim=_SEQ_DIM)


def _all_to_all(
    x: torch.Tensor, group: dist.ProcessGroup, *, src_dim: int, dst_dim: int
) -> torch.Tensor:
    """Move the cp shard of ``x`` from ``src_dim`` to ``dst_dim``.

    The same redistribution the module boundary would issue for a
    ``S(src_dim) -> S(dst_dim)`` declaration on the cp axis, so the backward is
    the reverse all-to-all in the activation dtype.
    """
    return spmd.redistribute(
        x,
        group,
        src=spmd.S(src_dim),
        dst=spmd.S(dst_dim),
        backward_options={"op_dtype": x.dtype},
    )


def kcp_plan(seq_len_local: int, group: dist.ProcessGroup) -> ContextParallelPlan:
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


class ContextParallelInnerKDA(ContextParallelKernel, InnerKDA):
    """Short convolution and KDA over one sequence shard per rank.

    The causal conv takes the previous rank's tail as history and the delta
    rule runs Attention Gym's context-parallel recipe, which exchanges
    per-fragment affine state summaries so each rank scans from its true
    entry state. The sequence stays sharded on both sides.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerKDA.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Checked at build time so the message is actionable, rather than an
        # ImportError from inside a layer's first forward.
        try:
            from attn_gym.linear.context_parallel import (  # noqa: F401
                context_parallel_conv_history,
            )
            from attn_gym.linear.kda import context_parallel_kda  # noqa: F401
        except ImportError as err:
            raise ValueError(
                "KDA context parallelism needs attn-gym's context-parallel "
                "recipe (attn_gym.linear.kda.context_parallel_kda and "
                "attn_gym.linear.context_parallel.context_parallel_conv_history); "
                f"import failed with: {err}."
            ) from err

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_TNK: torch.Tensor,
        raw_beta_TN: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        from attn_gym.linear.context_parallel import context_parallel_conv_history

        if cu_seqlens is not None:
            raise NotImplementedError(
                "Kimi K3 KDA context parallel runs one document per batch; "
                "packed-document boundaries under CP are not supported yet."
            )
        group = self.cp_group
        mixed_qkv_BTC, conv_weight_C1W = self._pack_inputs(
            query_TC,
            key_TC,
            value_TC,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
        )
        cp_plan = kcp_plan(mixed_qkv_BTC.shape[1], group)
        cu_seqlens = torch.tensor(
            cp_plan.cu_seqlens, dtype=torch.int32, device=mixed_qkv_BTC.device
        )
        # The causal conv needs the previous rank's tail as history.
        conv_state = context_parallel_conv_history(
            mixed_qkv_BTC, cp_plan, group, conv_weight_C1W.shape[-1] - 1
        )
        return self._conv_and_scan(
            mixed_qkv_BTC,
            conv_weight_C1W,
            raw_gate_TNK,
            raw_beta_TN,
            A_log_N,
            dt_bias_NK,
            cu_seqlens=cu_seqlens,
            conv_state=conv_state,
            cp_plan=cp_plan,
            cp_group=group,
        )


def use_kimi_k3_cp_kernels(config: "KimiK3Model.Config") -> None:
    """Give every attention layer its context-parallel kernel.

    The CP method is fixed by the layer kind, so no option selects it: MLA
    layers take Ulysses, KDA layers take KCP. Ulysses is FlexAttention over
    the full sequence, so any other inner attention is rejected here rather
    than by a shape error inside the kernel.
    """
    for layer in config.layers:
        if layer.attention is not None:
            inner = layer.attention.inner_attention
            if not isinstance(inner, FlexAttention.Config):
                raise ValueError(
                    "Kimi K3 context parallel runs Ulysses on FlexAttention; "
                    f"got {type(inner).__qualname__}."
                )
            layer.attention.inner_attention = _swap_kernel(
                inner, UlyssesCPFlexAttention
            )
        if layer.delta_attention is not None:
            layer.delta_attention.inner_kda = _swap_kernel(
                layer.delta_attention.inner_kda, ContextParallelInnerKDA
            )
