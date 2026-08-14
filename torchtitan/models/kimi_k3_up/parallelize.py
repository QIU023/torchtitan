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
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
)
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

    CP uses core's ``apply_cp_to_forward`` rather than a model-local
    implementation. Their MLA builds a ``ScaledDotProductAttention`` as its
    ``inner_attention``, which is one of the two types that helper handles, so
    ring attention over the sequence comes for free -- the same path llama3
    takes.

    KDA is a separate problem and is NOT covered here.
    ``prepare_context_parallel_input`` shards the sequence before the first
    layer, so every layer sees a shard, and KDA's recurrence has to be made
    aware of that (fla's merged KCP). Until that lands, CP is rejected for any
    model that has a KDA layer rather than silently producing a model whose
    linear-attention layers run on a shard as if it were the whole sequence.
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
        mla_inner = [
            block.attention.inner_attention
            for block in model.layers.values()
            if block.attention is not None
        ]
        kda_blocks = [b for b in model.layers.values() if b.delta_attention is not None]
        if kda_blocks:
            raise NotImplementedError(
                f"Context parallel needs KDA-aware linear attention, and this "
                f"model has {len(kda_blocks)} KDA layer(s). Sharding the "
                "sequence would leave their recurrence running on a shard as "
                "if it were the whole sequence, which is silently wrong rather "
                "than an error. Use a full-attention flavor until the KDA path "
                "lands."
            )
        # Before parallelize(), per apply_cp_to_forward's contract.
        apply_cp_to_forward(mla_inner, parallel_dims.get_mesh("cp"))

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
