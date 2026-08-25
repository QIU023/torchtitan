# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
import torch.nn as nn

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
)
from torchtitan.tools.logging import logger

from .kda import KimiDeltaAttention
from .sharding import contract_for_mode, ULYSSES
from .model import KimiK3Model, KimiMLAAttention


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """Apply FSDP2 to the Kimi K3 decoder and vision encoder."""

    unsupported_parallelisms = [
        name
        for name, enabled in (
            # Tensor parallel: declarations are in place but the forward does
            # not yet run end to end -- the latent MoE's second input path is
            # not named by the declaration. Paused on maintainer request.
            ("tensor parallel", parallel_dims.tp_enabled),


        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            "Kimi K3 currently supports FSDP2 data parallelism "
            f"only; disable {', '.join(unsupported_parallelisms)}."
        )
    if parallelism.spmd_backend != "partial_dtensor":
        raise NotImplementedError(
            "Kimi K3 FSDP2 currently supports the partial_dtensor SPMD backend "
            "only; the config registry pins it."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    # The routed experts shard on their own data-parallel mesh, which excludes
    # the expert axis; the same shape deepseek_v3 resolves.
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )

    assert isinstance(model, KimiK3Model)
    if (
        parallelism.spmd_backend == "spmd_types"
        or parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
    ):
        model.parallelize(parallel_dims)

    if parallel_dims.cp_enabled:
        apply_cp_kimi_k3(model, parallel_dims, training.max_context_length)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    vision_encoder = model.vision_encoder
    if vision_encoder is not None:
        # TODO: An image batch on one DP rank and a text-only batch on another
        # execute different FSDP collectives, deadlock, and hit a 90-second
        # timeout. A general solution is needed.
        apply_fsdp_to_vision_encoder(
            vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


def _check_head_divisibility(
    contract, num_heads: int, divisor: int, divisor_expr: str, kind: str, field: str
) -> None:
    """Enforce the head split a contract asks for, if it asks for one."""
    if not contract.head_sharded:
        return
    if num_heads % divisor != 0:
        raise ValueError(
            f"{kind} {field}={num_heads} must be divisible by "
            f"{divisor_expr}={divisor} for {contract.name} CP head sharding"
        )


def apply_cp_kimi_k3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    max_context_length: int | None = None,
) -> None:
    """Wire context parallelism: KCP on the KDA layers, Ulysses on the MLA layers.

    Both at once, on disjoint layer kinds. KCP decomposes the delta-rule
    recurrence and says nothing about softmax attention, so it does not replace
    Ulysses; ``cp_mode="ulysses"`` runs the KDA layers the second way and is
    kept as an A/B.

    Imperative rather than declared because KDA's kernels are fla triton and
    never see a DTensor; see ``cp_via_sharding_config`` on the model config for
    why the declarative path cannot serve them.
    """
    cp_group = parallel_dims.get_mesh("cp").get_group()
    cp_degree, tp_degree = parallel_dims.cp, parallel_dims.tp
    model._cp_group = cp_group
    model._cp_subgroups = _build_cp_subgroups(cp_group)
    # The CP mask rebuild is causal-only; hand the layers the context window
    # so they can reject a folded stream that holds several documents.
    # The window comes from the training config: K3's MLA is nope, so the
    # decoder's RoPE-derived max_context_length raises rather than returning
    # one. A folded stream longer than this holds more than one document, which
    # the causal-only CP mask cannot represent.
    max_ctx = max_context_length

    num_mla = 0
    kda_modules = []
    for module in model.modules():
        if isinstance(module, KimiMLAAttention):
            module._cp_max_context_length = max_ctx
            # Under TP the head axis is already tp-sharded, so Ulysses splits
            # what TP left: heads must divide by tp*cp, not by cp.
            _check_head_divisibility(
                ULYSSES,
                module.n_heads,
                tp_degree * cp_degree,
                "tp*cp",
                "MLA",
                "n_heads",
            )
            module._cp_group = cp_group
            num_mla += 1
        elif isinstance(module, KimiDeltaAttention):
            kda_modules.append(module)

    modes = {m.cp_mode for m in kda_modules}
    for mode in modes:
        contract = contract_for_mode(mode)
        for module in kda_modules:
            if module.cp_mode == mode:
                # cp alone, not tp*cp: KDA is TP-replicated (_set_kda_sharding
                # replicates every projection), so TP does not split its heads
                # and _forward_ulysses splits by cp_size only. Requiring tp*cp
                # rejects configurations that run -- tp=2, cp=2, 6 KDA heads is
                # the reference tree's example.
                _check_head_divisibility(
                    contract,
                    module.num_heads,
                    cp_degree,
                    "cp",
                    "KDA",
                    "num_heads",
                )
    if "kcp" in modes:
        # Checked here rather than at the first forward: the message is
        # actionable at wiring time and the failure is otherwise an ImportError
        # from inside a layer.
        try:
            from fla.modules.conv.cp.ops import causal_conv1d_cp  # noqa: F401
            from fla.ops.cp.context import build_cp_context  # noqa: F401
        except ImportError as err:
            raise ValueError(
                "cp_mode='kcp' needs fla-core's CP ops "
                "(fla.ops.cp.context.build_cp_context and "
                "fla.modules.conv.cp.ops.causal_conv1d_cp), which ship in "
                f"fla-core >= 0.5.1; import failed with: {err}. Install a "
                "newer fla-core or use cp_mode='ulysses'."
            ) from err

    for module in kda_modules:
        module._cp_group = cp_group
    if num_mla + len(kda_modules) == 0:
        raise ValueError(
            "context parallel is enabled but no attention layer was found to "
            "wire it onto."
        )
    logger.info(
        "Applied context parallel to %d MLA and %d KDA layer(s), modes=%s.",
        num_mla,
        len(kda_modules),
        sorted(modes) or ["-"],
    )


def _build_cp_subgroups(cp_group) -> dict[int, object]:
    """Pre-create every sub-CP group layout this CP group could use.

    Report 5.2.3 divides each CP group into sub-CP groups so gather-KV runs inside
    a sub-group instead of across the whole group. Which layout a step wants
    depends on how many large images the BATCH holds, and building a process group
    per batch is not an option: ``new_group`` must be called by every process in
    the default group, with the same rank list, in the same order. A per-batch call
    would have each rank passing its own CP group's ranks, which is exactly the
    mismatch that hangs.

    So every layout is built once here and looked up per batch. The layouts are the
    divisors of ``cp_size`` -- for cp=8 that is 1, 2, 4, 8 sub-groups -- so the set
    is small, and an unused group costs nothing because NCCL creates its
    communicator lazily on first use.

    Uniformity across ranks is achieved by all-gathering the CP rank lists first,
    so every rank iterates the same global list of sub-groups in the same order and
    keeps the one it belongs to. Returns ``{num_subgroups: this rank's group}``.
    """
    if cp_group is None:
        return {}
    cp_ranks = dist.get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)
    if cp_size <= 1:
        return {}

    # Every rank needs every CP group's membership, or the new_group calls below
    # would differ between ranks.
    world = dist.get_world_size()
    all_cp: list[list[int] | None] = [None] * world
    dist.all_gather_object(all_cp, cp_ranks)
    # Deduplicate while keeping a deterministic order: identical CP groups appear
    # once per member rank.
    seen: list[list[int]] = []
    for entry in all_cp:
        if entry and list(entry) not in seen:
            seen.append(list(entry))
    seen.sort()

    my_rank = dist.get_rank()
    out: dict[int, object] = {}
    for n_sub in [d for d in range(1, cp_size + 1) if cp_size % d == 0]:
        g = cp_size // n_sub
        mine = None
        for ranks in seen:
            for s in range(n_sub):
                members = ranks[s * g : (s + 1) * g]
                # Called on every rank, same order, same lists.
                pg = dist.new_group(ranks=members)
                if my_rank in members:
                    mine = pg
        if mine is not None:
            out[n_sub] = mine
    return out
