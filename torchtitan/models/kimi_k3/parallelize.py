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
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from .model import KimiK3Model
from .vision_encoder import enable_vision_cp_typecheck_rules


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig | None,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
    """Apply FSDP2 to the Kimi K3 decoder and vision encoder."""

    unsupported_parallelisms = [
        name
        for name, enabled in (("pipeline parallel", parallel_dims.pp_enabled),)
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            f"Kimi K3 does not support {', '.join(unsupported_parallelisms)}."
        )
    if compile_config is not None and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    assert isinstance(model, KimiK3Model)
    # Seed replicated layouts for parameters outside the explicit expert
    # declarations. Vision buffers declare their DP layouts separately.
    annotate_replicated_parameters(model, parallel_dims)

    # model_registry's moe_comm_backend picks the dispatcher: standard
    # (default) and deepep run on this model; hybridep
    # needs GB200-class hardware.
    model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    # Skip FSDP wrapper for inference. FSDP's forward hooks
    # are incompatible with torch.inference_mode() used by vLLM.
    # AC and compile are disabled via config (AC mode="none", compile is None).
    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    if parallel_dims.cp_enabled:
        # Building a process group is collective, so every rank runs this,
        # including a pipeline stage that holds no tower and never uses it.
        enable_vision_cp_typecheck_rules()
        model.set_vision_cp_subgroups(
            _build_cp_subgroups(parallel_dims.get_mesh(MeshAxisName.CP).get_group())
        )

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
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            dp_mesh_dims=dp_mesh_dims,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
    )

    return model


def _build_cp_subgroups(cp_group) -> dict[int, dist.ProcessGroup]:
    """One sub-CP group layout per divisor of the CP size: ``{sub-group count: this rank's group}``.

    Which layout a step wants depends on how many large images its batch holds,
    and a group cannot be built per batch, so every layout is built here. The CP
    rank lists are all-gathered first because the enumeration each call takes has
    to cover the world and be identical on every rank.
    """
    if cp_group is None:
        return {}
    cp_ranks = dist.get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)
    if cp_size <= 1:
        return {}
    gathered: list[list[int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, cp_ranks)
    every_cp_group = sorted({tuple(entry) for entry in gathered if entry})
    out: dict[int, dist.ProcessGroup] = {1: cp_group}
    for n_sub in (d for d in range(2, cp_size + 1) if cp_size % d == 0):
        size = cp_size // n_sub
        mine, _ = dist.new_subgroups_by_enumeration(
            [
                list(ranks[s * size : (s + 1) * size])
                for ranks in every_cp_group
                for s in range(n_sub)
            ]
        )
        assert isinstance(mine, dist.ProcessGroup)
        out[n_sub] = mine
    return out
