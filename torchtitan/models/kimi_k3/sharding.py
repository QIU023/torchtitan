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
"""

from dataclasses import dataclass

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout


__all__ = ["CPContract", "KCP", "ULYSSES", "contract_for_mode"]

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
