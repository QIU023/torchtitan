# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative CP contracts for the K3 attention layers.

Two CP algorithms run at once on disjoint layer kinds: Ulysses on the MLA
layers, KCP on the KDA layers. Each is stated here as a placement pair on the
CP mesh axis plus the preconditions that pair implies, so ``apply_cp_kimi_k3``
resolves a contract per module instead of branching per algorithm.

Only the CP axis is declared. The CP collectives run on plain local tensors
after the TP-wrapped projections, at the same gap the TP plan already strips
DTensor, so TP's own head sharding is not this contract's to describe -- and
declaring both here would be two mesh axes on tensor dim 2, which SpmdLayout
rejects without an explicit partition_spec.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.attention import (
    create_attention_mask,
    get_efficient_causal_mask_mod_for_packed_document,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config

from torchtitan.models.kimi_k3.dtensor_ops import (
    to_local_if_dtensor,
    to_local_partial_grad,
)


__all__ = [
    "CPContract",
    "KCP",
    "ULYSSES",
    "contract_for_mode",
    "cp_all_to_all_headseq",
]

CP = MeshAxisName.CP

# Tensor dims of the [T, H, K] activations the contracts talk about. This model
# carries a folded token stream with no batch axis, so the sequence is dim 0.
SEQ_DIM = 0
HEAD_DIM = 1


def _cp(axis_type: spmd.PerMeshAxisSpmdType) -> SpmdLayout:
    return SpmdLayout(axis_types={CP: axis_type})


@dataclass(frozen=True, slots=True)
class CPContract:
    """What one CP algorithm does to the [B, T, H, K] activations.

    Attributes:
        name: ``kda_cp_mode`` spelling, and what the wiring log reports.
        in_src: Placement entering the attention body.
        in_dst: Placement the body computes at.
        out_src: Placement leaving the body.
        out_dst: Placement at the module boundary.
        head_sharded: Whether the body splits heads across CP, i.e. whether
            the head-divisibility precondition applies.
    """

    name: str
    in_src: SpmdLayout
    in_dst: SpmdLayout
    out_src: SpmdLayout
    out_dst: SpmdLayout
    head_sharded: bool

    def redistributes(self) -> bool:
        """False when in_dst == in_src, i.e. the boundary moves no data."""
        return self.in_src.axis_types != self.in_dst.axis_types

    def in_dims(self) -> tuple[int, int]:
        """(src, dst) tensor dims the CP axis shards on the way in."""
        return _shard_dim(self.in_src), _shard_dim(self.in_dst)

    def out_dims(self) -> tuple[int, int]:
        """(src, dst) tensor dims the CP axis shards on the way out."""
        return _shard_dim(self.out_src), _shard_dim(self.out_dst)


def _shard_dim(layout: SpmdLayout) -> int:
    axis_type = layout.axis_types[CP]
    if not isinstance(axis_type, spmd.Shard):
        raise ValueError(
            f"CP contract expects a Shard on the CP axis, got {axis_type!r}"
        )
    return axis_type.dim


# Ulysses: projections run seq-local, then one all-to-all trades the sharded
# axis -- sequence for heads -- so the body sees the full sequence for its head
# subset. The output pair is the same swap reversed.
ULYSSES = CPContract(
    name="ulysses",
    in_src=_cp(spmd.S(SEQ_DIM)),
    in_dst=_cp(spmd.S(HEAD_DIM)),
    out_src=_cp(spmd.S(HEAD_DIM)),
    out_dst=_cp(spmd.S(SEQ_DIM)),
    head_sharded=True,
)

# KCP: the sequence stays sharded end to end (report sec 5.1.2). The delta-rule
# recurrence carries state rank to rank -- a sequential dependency no placement
# pair describes -- so the contract is an identity, declared to keep one shape.
KCP = CPContract(
    name="kcp",
    in_src=_cp(spmd.S(SEQ_DIM)),
    in_dst=_cp(spmd.S(SEQ_DIM)),
    out_src=_cp(spmd.S(SEQ_DIM)),
    out_dst=_cp(spmd.S(SEQ_DIM)),
    head_sharded=False,
)

_BY_MODE = {c.name: c for c in (ULYSSES, KCP)}


def contract_for_mode(mode: str) -> CPContract:
    if mode not in _BY_MODE:
        raise ValueError(f"kda_cp_mode must be one of {sorted(_BY_MODE)}, got {mode!r}")
    return _BY_MODE[mode]


def cp_all_to_all_headseq(
    x: torch.Tensor, cp_group, *, src_dim: int, dst_dim: int
) -> torch.Tensor:
    """Differentiable Ulysses all-to-all moving the CP shard between tensor dims.

    ``(0, 1)``: ``[T/cp, H, K]`` (seq-sharded) -> ``[T, H/cp, K]``.
    ``(1, 0)``: ``[T, H/cp, K]`` -> ``[T/cp, H, K]``.

    The dims come from the CP contract's placement pair rather than a flag, so a
    contract that names a pair with no implementation raises here instead of being
    quietly ignored.

    Numerics (round-trip and per-head chunk_kda parity) validated
    bit-exact against a single-rank reference; backward is the
    transposed all-to-all via torch.distributed.nn.functional.
    """
    import torch.distributed.nn.functional as dist_nn

    if (src_dim, dst_dim) not in ((SEQ_DIM, HEAD_DIM), (HEAD_DIM, SEQ_DIM)):
        raise ValueError(
            f"no Ulysses all-to-all for CP shard dims {src_dim} -> {dst_dim}; "
            f"implemented pairs are {SEQ_DIM} <-> {HEAD_DIM}"
        )
    cp = dist.get_world_size(cp_group)
    d0, d1, K = x.shape
    if (src_dim, dst_dim) == (SEQ_DIM, HEAD_DIM):
        t_loc, num_heads = d0, d1
        # [T/cp, H, K] -> [cp, T/cp, H/cp, K] (split heads by destination rank)
        x_split = x.reshape(t_loc, cp, num_heads // cp, K).permute(1, 0, 2, 3)
        out = dist_nn.all_to_all_single(
            torch.empty_like(x_split.contiguous()), x_split.contiguous(), group=cp_group
        )
        # recv[s] holds source s's T/cp for THIS rank's head subset, and s is
        # already the sequence-chunk order, so the reshape stacks the sequence.
        return out.reshape(cp * t_loc, num_heads // cp, K).contiguous()
    t_full, h_loc = d0, d1
    t_loc = t_full // cp
    # dim 0 is the destination rank: which sequence chunk each rank receives.
    x_split = x.reshape(cp, t_loc, h_loc, K).contiguous()
    out = dist_nn.all_to_all_single(torch.empty_like(x_split), x_split, group=cp_group)
    # out[s] = source s's head subset for THIS rank's sequence chunk; put T/cp
    # first so the reshape stacks heads in ascending source order.
    return out.permute(1, 0, 2, 3).reshape(t_loc, cp * h_loc, K).contiguous()


def full_sequence_document_mask(attn, positions_L, cp_group):
    """Document-aware mask for the sequence Ulysses reassembles.

    Each rank holds the full sequence after the all-to-all, so the global
    packed-document mask applies as-is; only the positions must be gathered
    (contiguous shards, no load balancer under CP). Rebuilt per call: the
    mask follows the data, not the shape.
    """
    if positions_L is None:
        raise ValueError(
            "context parallel needs positions to rebuild the packed-document "
            "mask for the reassembled sequence, but the attention layer "
            "received None."
        )
    gathered = [torch.empty_like(positions_L) for _ in range(dist.get_world_size(cp_group))]
    dist.all_gather(gathered, positions_L.contiguous(), group=cp_group)
    positions_full = torch.cat(gathered, dim=0)
    num_tokens = positions_full.shape[0]
    return create_attention_mask(
        get_efficient_causal_mask_mod_for_packed_document(positions_full),
        None,
        None,
        num_tokens,
        num_tokens,
        device=positions_full.device,
    )


def mla_ulysses_attention(
    attn,
    q_LHQ: torch.Tensor,
    kv_LHC: torch.Tensor,
    k_rope_LR: torch.Tensor,
    cp_group,
    positions_L: torch.Tensor | None,
) -> torch.Tensor:
    """MLA attention over the full sequence for this rank's head subset.

    * One fused all-to-all trades the sharded axis, sequence for heads; the
      attention backend runs unchanged; a second all-to-all trades back.
    * The rotary slice stays OUT of the exchange: it is headless (one vector
      per token), so it is all-gathered along the sequence and expanded onto
      local heads. Packing the expanded key instead reassembles it against the
      wrong head subset.
    * Shape suffixes beyond the legend: L local sequence (T/cp), G this rank's
      head count, W packed channel width, R rotary width.
    """
    # The two all-to-alls are raw functional collectives with no DTensor
    # sharding strategy, so under TP the stream drops to locals here and the
    # output re-wraps on the way out. The activations are head-sharded on the
    # TP axis: each rank's local gradient is its own shard's gradient, so
    # ``to_local``'s default grad placement (same as forward) is correct.
    # Unwrap BEFORE the shape reads -- a DTensor reports its GLOBAL head
    # count, which would inflate ``h_cp``.
    stream_placements = None
    if isinstance(q_LHQ, DTensor):
        stream_mesh = q_LHQ.device_mesh
        stream_placements = q_LHQ.placements
        q_LHQ = q_LHQ.to_local()
        kv_LHC = to_local_if_dtensor(kv_LHC)

    cp_size = dist.get_world_size(cp_group)
    t_loc = q_LHQ.shape[0]
    t_full = t_loc * cp_size
    # q_LHQ already carries this rank's TP-local heads, so cp splits those.
    h_cp = q_LHQ.shape[1] // cp_size

    packed_LHW = torch.cat([q_LHQ, kv_LHC], dim=-1)
    src_dim, dst_dim = ULYSSES.in_dims()
    packed_TGW = cp_all_to_all_headseq(
        packed_LHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
    )
    q_TGQ, k_nope_TGN, v_TGV = torch.split(
        packed_TGW,
        [attn.q_head_dim, attn.qk_nope_head_dim, attn.v_head_dim],
        dim=-1,
    )

    # wkv_a is TP-replicated, so its gradient is the SUM across TP ranks:
    # Partial, not the Replicate the default would keep. A no-op at tp=1.
    k_rope_LR = to_local_partial_grad(k_rope_LR)
    # Differentiable all-gather: backward is the reduce-scatter a value every
    # rank consumed needs.
    k_rope_TR = torch.cat(
        dist_nn.all_gather(k_rope_LR.contiguous(), group=cp_group), dim=0
    )
    k_TGQ = torch.cat(
        [
            k_nope_TGN,
            k_rope_TR.view(t_full, 1, attn.qk_rope_head_dim).expand(
                t_full, h_cp, attn.qk_rope_head_dim
            ),
        ],
        dim=-1,
    )

    out_TGV = attn.inner_attention(
        q_TGQ,
        k_TGQ,
        v_TGV,
        attention_masks=full_sequence_document_mask(attn, positions_L, cp_group),
        scale=attn.scale,
    )
    out_src_dim, out_dst_dim = ULYSSES.out_dims()
    # inner_attention runs under local_map, which re-wraps its output as a
    # DTensor -- drop it back to locals for the return all-to-all.
    out_LHV = cp_all_to_all_headseq(
        to_local_if_dtensor(out_TGV).contiguous(),
        cp_group,
        src_dim=out_src_dim,
        dst_dim=out_dst_dim,
    )
    if stream_placements is not None:
        out_LHV = DTensor.from_local(
            out_LHV, stream_mesh, stream_placements, run_check=False
        )
    return out_LHV


def set_expert_parallel_sharding_config(
    config, *, enable_sp: bool = False
) -> None:
    """Declare the sharding expert parallel acts on, with TP off.

    Shared with the EP review branch: the routed experts shard on the expert
    axis, and set_moe_sharding_config's input boundary lifts the plain
    incoming activations itself -- no decoder-level declaration is needed
    (ablated: removing it changes nothing to every printed digit). The
    combined ep+tp declaration stays inline in model.py -- with TP on, tp
    becomes a token axis inside the MoE region and the two cannot be
    declared independently.
    """
    for layer in config.layers:
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=True,
                # TODO: flip to True from the caller once the
                # tensor-parallel PR lands; with EP alone the internals run
                # without sequence parallel.
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )
