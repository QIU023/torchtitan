from typing import Any
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile, raise_dynamo_recompile_limit
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from .model import KimiK3Model


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
    """Apply FSDP2 to the Kimi K3 decoder and vision encoder."""

    unsupported_parallelisms = [
        name
        for name, enabled in (
            (
                "context parallel with tensor parallel",
                parallel_dims.cp_enabled and parallel_dims.tp_enabled,
            ),
            (
                "context parallel with pipeline parallel",
                parallel_dims.cp_enabled and parallel_dims.pp_enabled,
            ),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            f"Kimi K3 does not support {', '.join(unsupported_parallelisms)}."
        )

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

    if compile_config.enable and "model" in compile_config.components:
        assert isinstance(model.config, KimiK3Model.Config)
        raise_dynamo_recompile_limit(_kimi_k3_recompile_limit(model.config))
        apply_compile(model, compile_config=compile_config, parallel_dims=parallel_dims)
        if model.vision_encoder is not None:
            apply_compile(
                model.vision_encoder,  # pyrefly: ignore [bad-argument-type]
                compile_config=compile_config,
                parallel_dims=parallel_dims,
            )

    # Skip FSDP wrapper for inference. FSDP's forward hooks
    # are incompatible with torch.inference_mode() used by vLLM.
    # AC and compile are disabled via config (mode="none", enable=False).
    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

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
            cpu_offload=training.enable_cpu_offload,
            dp_mesh_dims=dp_mesh_dims,
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
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
    )

    return model
_kernels_carved_out = False


def _carve_kernels_out_of_dynamo() -> None:
    """Keep the KDA kernel wrapper out of the compiled graph.

    ``KDAKernel.forward`` is the Attention Gym Triton kernels plus their gate
    and normalisation preprocessing, and under context parallelism the
    per-fragment state exchange; dynamo does not trace through them, so the
    whole forward runs eagerly and the graph breaks around it.
    ``recursive=True`` keeps dynamo out on re-entry from the backward too.
    Class-level state, applied once, not per model part.
    """
    global _kernels_carved_out
    if _kernels_carved_out:
        return
    _kernels_carved_out = True
    from torchtitan.models.kimi_k3.kda import KDAKernel

    KDAKernel.forward = torch.compiler.disable(  # pyrefly: ignore [bad-assignment]
        KDAKernel.forward, recursive=True
    )


def _apply_compile_kimi_k3(
    model: nn.Module, *, compile_config: CompileConfig, parallel_dims: ParallelDims
) -> None:
    """Compile each transformer block with the KDA kernels carved out.

    A block alternates between KDA and MLA attention, so the compiled blocks
    specialise per attention class; the graph breaks around the kernel wrapper
    mean ``fullgraph`` cannot be required.
    """
    _carve_kernels_out_of_dynamo()
    apply_compile(
        model,
        compile_config=compile_config,
        parallel_dims=parallel_dims,
        fullgraph=False,
    )


def _apply_ac_outside_attention(ac_policy, model: nn.Module) -> None:
    """Checkpoint only each block's MoE/feed-forward; attention stays outside.

    The KDA kernel is a custom op outside the selective policy's save set, so a
    whole-block wrap recomputes it in backward. Wrapping just the FFN keeps the
    mm save/recompute balance where the parameter memory is, while attention
    and the residual math keep their activations and are reused in backward.
    """
    layers = model.layers
    assert isinstance(layers, nn.ModuleDict)
    for name, block in layers.named_children():
        inner_name = "moe" if block.moe is not None else "feed_forward"
        inner = getattr(block, inner_name)
        wrapped = ac_policy._wrap_block(inner, base_fqn=f"layers.{name}.{inner_name}")
        block.register_module(inner_name, wrapped)
    logger.info(
        "Applied activation checkpointing to the MoE/feed-forward of %d block(s); "
        "attention and the residual math stay outside (ac_reuse_attention).",
        len(layers),
    )


def _install_vision_dep(
    pp_schedule,
    stages: list[AttnResPipelineStage],
    *,
    rank: int,
    pp_size: int,
    prefetch: int,
    bubble: bool,
    cost_ratio: float,
    max_pending: int,
) -> None:
    """Wire the tower stage's encodes to the schedule (report sec 5.2.3).

    The stage holding the tower serves each micro-batch's forward from a
    per-step feature cache when the encode already ran, and the cache is
    filled either ahead of the consumer (``prefetch`` micro-batches ahead) or
    in the schedule's idle intervals (``bubble``: the plan is read off the
    schedule's own action order, so every rank derives the same placements).
    A rank without the tower stage is left alone.
    """
    from torchtitan.models.kimi_k3.dep_bubble_backward import (
        cut_for_deferred_backward,
        GradQueue,
        install_backward_slots,
    )
    from torchtitan.models.kimi_k3.dep_bubble_plan import plan_for_rank
    from torchtitan.models.kimi_k3.dep_bubble_runtime import install_bubble_runtime
    from torchtitan.models.kimi_k3.vit_prefetch import VisionPrefetcher

    tower_stages = [
        s for s in stages if getattr(s.submod, "vision_encoder", None) is not None
    ]
    if not tower_stages:
        return
    if len(tower_stages) != 1:
        raise RuntimeError(
            f"rank {rank} holds {len(tower_stages)} stages with a vision tower; "
            "vit_dep places it on one."
        )
    tower_stage = tower_stages[0]
    module = tower_stage.submod
    prefetcher = VisionPrefetcher(module)
    queue = GradQueue(max_pending=max_pending) if bubble else None
    # FSDP2 initializes its state lazily in the root module's first forward; an
    # encode issued before that makes the tower a root of its own and the
    # stage's forward then refuses. So the first step runs its encodes inline,
    # and the run-ahead or the plan starts with the second.
    warm = {"done": False}

    inner_forward = tower_stage.forward_one_chunk

    def forward_one_chunk(fwd_chunk_id, args, kwargs=None, *rest, **more):
        feats = prefetcher.take(int(fwd_chunk_id))
        if feats is not None:
            feats = feats[0] if isinstance(feats, (list, tuple)) else feats
            if queue is not None:
                feats = cut_for_deferred_backward(feats, queue, int(fwd_chunk_id))
            kwargs = {**(kwargs or {}), "vision_embeds": feats}
        out = inner_forward(fwd_chunk_id, args, kwargs, *rest, **more)
        warm["done"] = True
        if prefetch:
            prefetcher.advance(int(fwd_chunk_id), prefetch)
        return out

    tower_stage.forward_one_chunk = forward_one_chunk  # type: ignore[method-assign]

    if bubble:
        pipeline_order = getattr(pp_schedule, "pipeline_order", None)
        if not pipeline_order:
            raise ValueError(
                "vit_bubble reads the schedule's action order, which only the looped "
                "schedules (Interleaved1F1B and kin) expose; use vit_prefetch with a "
                "single-stage schedule."
            )

        def plan_for_step():
            n = prefetcher._num_mbs
            if n == 0 or not warm["done"]:
                return None
            return plan_for_rank(
                pipeline_order[rank],
                rank=rank,
                vision_microbatches=n,
                cost_ratio=cost_ratio,
                upfront=min(pp_size, n),
                vision_stage=tower_stage.stage_index,
            )

        def encode_now(microbatches):
            for mb in microbatches:
                prefetcher.ensure_sync(mb)

        # Installed before the step wrapper below, so begin_step runs first
        # and the plan sees this step's micro-batch count.
        install_bubble_runtime(
            pp_schedule,
            plan_for_step=plan_for_step,
            encode_now=encode_now,
            upfront_encode=encode_now,
        )
        assert queue is not None
        install_backward_slots(pp_schedule, queue)

    original_step = pp_schedule.step

    def step(*args, **kwargs):
        prefetcher.begin_step(kwargs.get("kwarg_mbs"))
        if prefetch and warm["done"]:
            for mb in range(prefetch):
                prefetcher.ensure(mb)
        return original_step(*args, **kwargs)

    pp_schedule.step = step  # type: ignore[method-assign]

    if not bubble:
        logger.info(
            "DEP vision prefetch installed: depth=%d on stage %d",
            prefetch,
            tower_stage.stage_index,
        )
        return
    logger.info(
        "DEP bubble runtime installed on stage %d (cost ratio %.2f, max pending %d)",
        tower_stage.stage_index,
        cost_ratio,
        max_pending,
    )


def _kimi_k3_recompile_limit(config: KimiK3Model.Config) -> int:
    # One graph per (MLA or KDA, opens a block, stack width 0/1/>=2); the tower adds 2.
    variants = {
        (
            layer.attention is not None,
            layer.layer_id % layer.attn_res_block_size == 0,
            min(-(-layer.layer_id // layer.attn_res_block_size), 2),
        )
        for layer in config.layers
    }
    return len(variants) + 2
