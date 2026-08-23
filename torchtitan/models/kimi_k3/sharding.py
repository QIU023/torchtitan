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

See CP_DECLARATIVE.md in the logbook for why KCP is an identity pair.

Also the two TP declaration helpers the K3 modules share. They live here rather
than in ``model.py`` so that ``kda.py``, ``vision_encoder.py`` and
``attn_res_model.py`` can reach them without importing the model module, and so
that replacing ``tp_replicate`` with core's invariant shape is one edit.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.distributed as dist

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.sharding import ShardingConfig


__all__ = [
    "CPContract",
    "KCP",
    "ULYSSES",
    "contract_for_mode",
    "cp_all_to_all_headseq",
    "tp_replicate",
    "tp_shard",
]

CP = MeshAxisName.CP

# Tensor dims of the [B, T, H, K] activations the contracts talk about.
SEQ_DIM = 1
HEAD_DIM = 2


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
# recurrence carries state rank to rank, which is a sequential dependency, not a
# redistribution -- no placement pair describes it, so it stays inside the op and
# the contract is an identity. Declared anyway to keep one shape for both modes.
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

    ``(1, 2)``: ``[B, T/cp, H, K]`` (seq-sharded) -> ``[B, T, H/cp, K]``.
    ``(2, 1)``: ``[B, T, H/cp, K]`` -> ``[B, T/cp, H, K]``.

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
    B, d1, d2, K = x.shape
    if (src_dim, dst_dim) == (SEQ_DIM, HEAD_DIM):
        t_loc, num_heads = d1, d2
        # [B, T/cp, H, K] -> [cp, B, T/cp, H/cp, K] (split heads by dest)
        x_split = (
            x.reshape(B, t_loc, cp, num_heads // cp, K)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        out = dist_nn.all_to_all_single(
            torch.empty_like(x_split), x_split, group=cp_group
        )
        # recv[s] holds src s's T/cp for THIS rank's head subset -> stack seq
        return (
            out.permute(1, 0, 2, 3, 4)
            .reshape(B, cp * t_loc, num_heads // cp, K)
            .contiguous()
        )
    t_full, h_loc = d1, d2
    t_loc = t_full // cp
    x_split = x.reshape(B, cp, t_loc, h_loc, K).permute(1, 0, 2, 3, 4).contiguous()
    out = dist_nn.all_to_all_single(torch.empty_like(x_split), x_split, group=cp_group)
    # out[s] = src s's head subset for THIS rank's seq shard; put T/cp
    # before the src(cp) axis so reshape stacks heads in ascending order.
    return out.permute(1, 2, 0, 3, 4).reshape(B, t_loc, cp * h_loc, K).contiguous()


def tp_shard(dim: int) -> ShardingConfig:
    """Weight sharded on ``dim`` of the tp axis; colwise is 0, rowwise is 1."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(dim))}
    )


def tp_replicate() -> ShardingConfig:
    """Weight replicated on the tp axis (the NoParallel case)."""
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


# ---------------------------------------------------------------------------
# Parameter declarations for the spmd_types backend.
#
# spmd_types needs every parameter to already be a DTensor on the full SPMD mesh
# before fully_shard. This model declares none today -- 537 parameter-owning
# modules, zero sharding_config -- so the backend cannot start at all. See
# SPMD_TYPES_GAP_2026-08-20.md in the logbook for the inventory.
#
# Filled in slices, norms first, because they are the placement-simplest 117 of
# the 590 and prove the mechanism end to end before the colwise/rowwise mapping
# for the 280 Linears has to be got right.
# ---------------------------------------------------------------------------


def declare_norm_sharding(model, *, enable_sp: bool) -> int:
    """Attach norm parameter placements to BUILT RMSNorm modules. Returns the count.

    Upstream models declare on ``Module.Config`` before ``build()``. That route does
    not reach this model: KimiK3AttnResModel -- what the flavors actually construct --
    calls ``nn.Module.__init__`` and builds its layers straight from the flat
    KimiK3Config, so there is no config tree carrying ``.norm`` or
    ``.layers[i].input_layernorm`` to declare on. Declaring on the instances is the
    same contract applied one step later.

    Only RMSNorm, which this model owns via torchtitan's class. FusedRMSNormGated is
    fla's and ShortConvolution likewise; those need their own answer.
    """
    from torchtitan.models.common.decoder_sharding import norm_config
    from torchtitan.models.common.nn_modules import RMSNorm

    count = 0
    already = 0
    for module in model.modules():
        if not isinstance(module, RMSNorm):
            continue
        if getattr(module, "_sharding_config", None) is not None:
            continue
        # A module already marked parallelized will never be revisited, so a
        # declaration attached now can only be dead weight. Counted rather than
        # assumed: "declared N" and "N of them will be acted on" are different
        # numbers, and the parameter table cannot tell them apart.
        if getattr(module, "_parallelized", False):
            already += 1
        module._sharding_config = norm_config(enable_sp=enable_sp)
        count += 1
    if already:
        from torchtitan.tools.logging import logger

        logger.warning(
            "declare_norm_sharding: %d of %d norms were already parallelized; "
            "their declarations will not be applied.",
            already,
            count,
        )
    return count


def annotate_untyped_params(model, parallel_dims) -> int:
    """Give every parameter still lacking an spmd type a replicated one. Returns the count.

    Under spmd_types FSDP needs each parameter to carry a type annotation, and
    ``Module.parallelize`` only annotates modules that declare a sharding_config. Three
    kinds of parameter are left over here and none can be reached by declaring on a
    Module:

    * fla's ``ShortConvolution`` and ``FusedRMSNormGated`` are not torchtitan Modules at
      all, so ``parallelize()`` never visits them;
    * the grouped-expert weights are distributed by the EP path, which predates this;
    * a handful of Linears and the embedding sit outside any declared subtree.

    Replicated is the right default and not a placeholder: an unsharded parameter IS
    replicated on every axis, so the annotation states what is already true. Anything
    genuinely sharded is skipped -- a DTensor carries its own layout, and a parameter
    that already has a type was annotated by whoever distributed it.
    """
    from spmd_types.runtime import has_local_type
    from torch.distributed.tensor import DTensor

    from torchtitan.distributed.spmd_types import set_current_spmd_mesh
    from torchtitan.models.common.decoder_sharding import dense_param_placement

    layout = dense_param_placement(tp=spmd.R)
    mesh = parallel_dims.get_optional_mesh(
        [axis.value for axis in layout.axes()], include_singleton_axes=True
    )
    if mesh is None:
        return 0

    count = 0
    with set_current_spmd_mesh(mesh):
        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if isinstance(param, DTensor) or has_local_type(param):
                    continue
                spmd.assert_type(param, layout.axis_types, layout.partition_spec)
                count += 1
    return count


def drop_declarations_on_distributed(model) -> int:
    """Remove declarations from modules the imperative plan already distributed.

    Returns how many were dropped.

    ``_drive_declarative_sharding`` already refuses to enter a subtree whose root the
    imperative plan covered, so that the driver activates exactly the declarations the
    plan does NOT. But that guard only holds at the subtree root: once
    ``Module.parallelize()`` is called it recurses through everything below without it,
    and under spmd_types the first TP-distributed weight it reaches raises
    ``assert_type() does not support DTensor``. Under partial_dtensor the same recursion
    is harmless, because ``_distribute_states`` has a branch that merely verifies an
    existing DTensor's placements.

    So the policy the driver implements per subtree is applied here per module. Dropping
    the declaration costs those parameters nothing: they are DTensors, and FSDP accepts a
    DTensor directly -- it is the LOCAL tensors that need an spmd type annotation.

    Temporary in the same sense as the driver's guard: both exist because the imperative
    TP plan and the declarations are live at once, and both go away when TP becomes
    declarative.
    """
    from torch.distributed.tensor import DTensor

    dropped = 0
    for module in model.modules():
        if getattr(module, "_sharding_config", None) is None:
            continue
        if any(isinstance(p, DTensor) for p in module.parameters(recurse=False)):
            module._sharding_config = None
            dropped += 1
    return dropped


def declare_tp_sharding(model, *, enable_sp: bool) -> tuple[int, int]:
    """Declare TP placements on the MLA projections.

    Returns ``(newly declared, already declared)``.

    The imperative plan cannot run under spmd_types: it makes DTensors on the tp mesh
    while FSDP there wants the full SPMD storage mesh ("Expected param's DTensor mesh to
    be the same mesh passed to fully_shard"). Declaring instead leaves local tensors
    sliced on the tp group and annotated.

    Structured as a walk over layers rather than a match on module names, because the
    names do not separate the two attention kinds: a KDA layer's projection is
    ``delta_attention.q_proj``, which ENDS WITH ``attention.q_proj``. KDA layers take no
    TP at all in the imperative plan -- they stay replicated -- so name matching would
    shard them and be wrong in a way nothing else would catch.

    Only the projections the imperative plan actually shards are declared. The rest of
    that plan is NoParallel, i.e. replicated, which is what annotate_untyped_params
    already provides.
    """
    from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config

    # Counted apart because they mean opposite things. "Declared 0" reads as "no TP
    # was set up", but it also happens when every projection ALREADY carried a
    # declaration -- a correct state. Conflating them turned a working model into a
    # hard failure once.
    declared = 0
    already = 0

    def declare(parent, attr, config) -> None:
        nonlocal declared, already
        module = getattr(parent, attr, None)
        if module is None:
            return
        # LoRA wraps the projection; the placements belong on the nn.Linear inside,
        # the same redirect the imperative plan performs.
        target = getattr(module, "base", module)
        if getattr(target, "_sharding_config", None) is not None:
            already += 1
            return
        target._sharding_config = config
        declared += 1

    layers = getattr(model, "layers", None)
    if layers is None:
        return 0, 0
    for layer in layers.values() if hasattr(layers, "values") else layers:
        if bool(getattr(layer, "is_linear_attn", False)):
            continue
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for attr in ("q_proj", "q_b_proj", "kv_b_proj", "attn_gate_proj"):
            declare(attn, attr, colwise_config())
        declare(attn, "o_proj", rowwise_config(output_sp=enable_sp))

    declare(model, "lm_head", colwise_config())
    return declared, already
