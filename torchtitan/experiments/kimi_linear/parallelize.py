# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism application for Kimi Linear models.

Phase 4c scope: FSDP2 full-shard only (no TP, no CP, no PP, no AC,
no compile). Targets single-node training where PP is an anti-pattern
(NVLink-class interconnects make FSDP the throughput winner — see
``phase4/launch_fsdp_small.sh`` header).

Adapted from ``torchtitan.models.llama3.parallelize.apply_fsdp`` to
Kimi's module names (``embed_tokens``, ``norm``, ``lm_head``,
``layers`` as ``nn.ModuleList``). The FSDP2 API itself (``fully_shard``
+ ``MixedPrecisionPolicy``) is identical; only the module-layout
traversal differs.

Phase 4d / later may add TP or AC once the decoder layer's sub-module
names are stable and sharded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_kimi_linear(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
) -> nn.Module:
    """Apply FSDP2 to a Kimi Linear model.

    Only Data Parallel Sharding is wired here. Tensor Parallel,
    Context Parallel, activation checkpoint, and torch.compile are
    unsupported in Phase 4c — they'd require per-module layout
    decisions that this minimal port skips. Raise on unsupported knobs.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "TP not supported for Kimi Linear in Phase 4c. "
            "Use parallelism.tensor_parallel_degree=1."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "CP not supported for Kimi Linear in Phase 4c. "
            "Use parallelism.context_parallel_degree=1."
        )
    if ac_config.mode not in ("none", "None", None):
        logger.warning(
            "activation_checkpoint.mode=%s ignored — Kimi Linear Phase 4c "
            "skips AC. Set mode=none to silence this warning.",
            ac_config.mode,
        )
    # torch.compile applied per-decoder-layer BEFORE FSDP wrap (so each
    # FSDP unit wraps a compiled subgraph). MoE for-loop expert path
    # is NOT compiled (torchtitan upstream has the same carve-out: see
    # apply_compile_sparse comment about unbacked symints in for-loop
    # fallback). fla-core ops (chunk_kda, ShortConvolution,
    # FusedRMSNormGated) are wrapped with torch.compiler.disable since
    # they're triton kernels that dynamo can't trace through.
    if compile_config.enable:
        _apply_compile_kimi_linear(model, compile_config)
        logger.info(
            "Compiled each KimiDecoderLayer with torch.compile (backend=%s).",
            compile_config.backend,
        )

    if parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled:
        # Use "fsdp" (shard only) when shard>1 replicate=1; "batch"
        # mesh combines shard + replicate when replicate>1. Fall back
        # to whichever is valid for the current mesh layout.
        if parallel_dims.dp_replicate_enabled and parallel_dims.dp_shard_enabled:
            dp_mesh = parallel_dims.get_mesh("batch")
        elif parallel_dims.dp_shard_enabled:
            dp_mesh = parallel_dims.get_mesh("fsdp")
        else:
            dp_mesh = parallel_dims.get_mesh("dp_replicate")
        param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
        apply_fsdp(
            model,
            dp_mesh=dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=(
                parallelism.fsdp_reshard_after_forward
            ),
        )
        logger.info(
            "Applied FSDP2 to Kimi Linear model (dp_shard=%d, dp_replicate=%d).",
            parallel_dims.dp_shard,
            parallel_dims.dp_replicate,
        )
    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    """FSDP2 fully_shard application tuned for Kimi Linear's module layout.

    Module naming (see ``kimi_linear/model.py``):
      - ``embed_tokens`` (nn.Embedding)
      - ``layers`` (nn.ModuleList of KimiDecoderLayer or AttnRes variant)
      - ``norm`` (nn.RMSNorm)
      - ``lm_head`` (nn.Linear)
      - [AttnRes only] ``final_attn_res_proj`` + ``final_attn_res_norm``

    Sharding layout mirrors Llama3's apply_fsdp: group embed with
    {norm, lm_head} only when tied, else put embed alone and
    {norm, lm_head} together.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    # Collect the "output tail" modules. norm + lm_head ALWAYS run
    # together every forward, so they share an FSDP unit — that
    # amortizes a single all-gather over both. final_attn_res_proj
    # and final_attn_res_norm are AttnRes-only and only fire at the
    # very end of forward (block_attn_res(...) accumulates into
    # h_final), so they get their OWN FSDP unit; bundling them with
    # norm/lm_head triggers FSDP2's "module did not run forward
    # before backward" warning because dynamo / autograd sees the
    # AttnRes call timing as separate from the lm_head pass.
    head_tail: list[nn.Module] = []
    if getattr(model, "norm", None) is not None:
        head_tail.append(model.norm)
    if getattr(model, "lm_head", None) is not None:
        head_tail.append(model.lm_head)

    attn_res_tail: list[nn.Module] = []
    if getattr(model, "final_attn_res_proj", None) is not None:
        attn_res_tail.append(model.final_attn_res_proj)
    if getattr(model, "final_attn_res_norm", None) is not None:
        attn_res_tail.append(model.final_attn_res_norm)

    tied = bool(getattr(model, "config", None)) and getattr(
        model.config, "tie_word_embeddings", False
    )

    if tied:
        # When tied, embed shares storage with lm_head — bundle them
        # so the shared weight isn't all-gathered twice.
        fully_shard(
            [model.embed_tokens, *head_tail],
            **fsdp_config,
            reshard_after_forward=(reshard_after_forward_policy == "always"),
        )
    else:
        fully_shard(
            model.embed_tokens,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        fully_shard(
            head_tail,
            **fsdp_config,
            reshard_after_forward=(reshard_after_forward_policy == "always"),
        )

    if attn_res_tail:
        fully_shard(
            attn_res_tail,
            **fsdp_config,
            reshard_after_forward=(reshard_after_forward_policy == "always"),
        )

    # Shard every decoder layer independently so each layer's forward
    # all-gather / backward reduce-scatter is overlapped with compute.
    # model.layers is a ModuleDict (str→layer); iterate .values() to
    # grab the layer modules.
    for layer in model.layers.values():
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Finally, wrap the top-level model so FSDP2 has a single root
    # module for its pre-/post-forward hook chain. Without this, FSDP
    # errors at forward with "requires a single root module".
    fully_shard(
        model,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )


def _apply_compile_kimi_linear(model: nn.Module, compile_config: CompileConfig) -> None:
    """Wrap each KimiDecoderLayer with torch.compile.

    Carve-outs (must NOT be compiled):
    * fla-core triton kernels (chunk_kda, ShortConvolution,
      FusedRMSNormGated, fused_kda_gate) — dynamo cannot trace through
      arbitrary Triton, and these are already optimized.
    * MoE for-loop expert path (when ``use_grouped_mm=False``) — same
      unbacked-symint issue torchtitan upstream documents in
      ``apply_compile_sparse``.

    The fla carve-outs are applied as ``torch.compiler.disable`` shims
    with ``recursive=True`` so dynamo treats the entire subtree as
    opaque (otherwise the backward pass re-enters dynamo at e.g.
    ``cuda_utils.get_device_properties`` and emits warnings).

    Recompile-limit handling: KimiDecoderLayer alternates between
    KDA and MLA attention (3:1 by layer index). Default dynamo
    recompile_limit=8 is too small — the type check on
    ``self_attn`` triggers a recompile per attention class, and once
    the limit is hit dynamo silently falls back to eager for
    affected frames. We bump recompile_limit + cache_size_limit so
    each layer-flavor compiles cleanly on first hit and stays cached.
    """
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate

    # Mark triton ops as opaque to dynamo. recursive=True so dynamo
    # also stays out on re-entry from autograd backward (otherwise
    # fla's backward kernels trip on cuda_utils.get_device_properties
    # and lru_cache decorators inside fused_norm_gate).
    for op in (chunk_kda, fused_recurrent_kda, fused_kda_gate):
        torch.compiler.disable(op, recursive=True)
    for cls in (ShortConvolution, FusedRMSNormGated):
        cls.forward = torch.compiler.disable(cls.forward, recursive=True)

    # Allow MoE token-choice routing's data-dependent control flow.
    torch._dynamo.config.capture_scalar_outputs = True
    # Eager AC <-> compile divergence acceptance (matches upstream).
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    # KDA + MLA layers each compile separately; we have up to L layer
    # flavors plus permutations. 64 leaves comfortable headroom for
    # all per-layer specializations without thrashing.
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64

    for _, layer in model.layers.named_children():
        layer.compile(backend=compile_config.backend, fullgraph=False)
