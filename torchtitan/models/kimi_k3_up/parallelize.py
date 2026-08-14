# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 and context parallelism for the eager Kimi K3 reference model."""

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
from torchtitan.models.kimi_k3_up.cp_kcp import apply_kcp
from torchtitan.models.kimi_k3_up.cp_ulysses import apply_ulysses_cp
from torchtitan.models.kimi_k3_up.model import KimiK3Model


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
    """Apply FSDP2 and CP while keeping the model's eager reference forward path.

    CP is Ulysses, not core's ring path. Core's ``apply_cp_to_forward`` routes
    an SDPA inner attention onto the CP dispatcher, and measured there K3's MLA
    fails inside the dispatcher's accumulation while llama3 passes the same cell
    on the same tree. Ulysses also needs nothing from the attention kernel,
    which is what makes it the portable choice here. See ``cp_ulysses``.

    KDA takes KCP (``cp_kcp``): the recurrence gets fla's prefix-scan context
    and the short convolutions exchange their halo. It carries one constraint
    Ulysses does not -- fla's CP path packs the batch into a single sequence, so
    local_batch_size must be 1.
    """
    del dump_folder

    unsupported_parallelisms = [
        name
        for name, enabled in (
            ("tensor parallel", parallel_dims.tp_enabled),
            ("pipeline parallel", parallel_dims.pp_enabled),
            ("expert parallel", parallel_dims.ep_enabled),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            "Kimi K3 eager reference currently supports FSDP2 data parallelism "
            f"only; disable {', '.join(unsupported_parallelisms)}."
        )
    if parallelism.spmd_backend != "default":
        raise NotImplementedError(
            "Kimi K3 eager FSDP2 currently supports the default SPMD backend only."
        )
    if compile_config.enable:
        raise NotImplementedError(
            "Kimi K3 eager reference does not support torch.compile."
        )
    if ac_config is not None:
        raise NotImplementedError(
            "Kimi K3 eager FSDP2 does not support activation checkpointing yet."
        )
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "Kimi K3 eager FSDP2 does not support parameter CPU offload yet."
        )

    if parallel_dims.cp_enabled:
        # Ulysses wraps the ATTENTION module, not its inner attention: the
        # all-to-all has to sit around the projections, which live one level up.
        mla_blocks = [
            block.attention
            for block in model.layers.values()
            if block.attention is not None
        ]
        kda_blocks = [
            b.delta_attention
            for b in model.layers.values()
            if b.delta_attention is not None
        ]
        if kda_blocks and training.local_batch_size != 1:
            # fla's causal_conv1d_cp asserts [1, T, D]: the CP path is built for
            # cu_seqlens packing, where the batch IS one packed sequence. Caught
            # here rather than at the first KDA forward, which is several minutes
            # of startup later.
            raise NotImplementedError(
                "KCP requires training.local_batch_size == 1 (fla's CP path packs "
                f"the batch into one sequence), got {training.local_batch_size}."
            )
        # Before parallelize(), matching core's helper contract.
        cp_mesh = parallel_dims.get_mesh("cp")
        apply_ulysses_cp(mla_blocks, cp_mesh)
        if kda_blocks:
            apply_kcp(kda_blocks, cp_mesh)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    assert isinstance(model, KimiK3Model)
    vision_encoder = model.vision_encoder
    if vision_encoder is not None:
        apply_fsdp_to_vision_encoder(
            vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=1,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model
