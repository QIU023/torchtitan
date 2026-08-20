# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism application for Kimi Linear models.

Supported parallelism dimensions:

* **FSDP2 full-shard** -- primary path, modeled on
  ``torchtitan.models.llama3.parallelize.apply_fsdp``. Adapted to Kimi's
  module names (``embed_tokens``, ``norm``, ``lm_head``, ``layers`` as
  ``nn.ModuleList``).
* **torch.compile** -- per-decoder-layer compile via
  ``_apply_compile_kimi_k3``; MoE for-loop and fla-core triton ops
  are carved out.
* **Activation checkpointing** -- applied via the shared
  ``ActivationCheckpointingConfig.build().apply()`` path since the Kimi
  decoder layer registry matches the llama3 ``model.layers`` iteration
  pattern. Honors all upstream modes (``selective``, ``full``,
  ``memory_budget``, ``none``).

TP, CP and EP each raise here. They arrive as one follow-up PR per axis, each
deleting its own ValueError and adding its own ``apply_*`` -- the three do not
call into each other, so the only thing they share is this entry. PP arrives
the same way, through ``pipeline_adapter.py`` rather than through this file.

"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

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
    """Apply the configured parallelism plan to a Kimi Linear model.

    Wires (in order, before FSDP wrap): AC -> compile -> FSDP/HSDP. AC is
    applied before compile so the compiled subgraph is the checkpointed unit
    (matches upstream llama3/qwen3 ordering).

    TP, CP and EP raise; each is added by its own follow-up PR, ahead of AC.
    """

    # Resolve the topology knobs from config ONCE, before anything reads them
    # (finding 32). Both this and the pipelining entry register; first call wins.
    from torchtitan.models.kimi_k3.knobs import register_topology

    if hasattr(model, "config"):
        register_topology(model.config)

    # Enable TF32 tensor cores for fp32 matmuls (loss aggregation,
    # optimizer master weight updates, fp32 RoPE etc.). bf16 path is
    # unaffected. Speedup ~5-10% on fp32 ops, no measurable accuracy
    # impact at our scale.
    torch.set_float32_matmul_precision("high")

    if parallel_dims.tp_enabled:
        raise ValueError("Tensor parallelism is not wired for Kimi K3 in this PR.")
    if parallel_dims.cp_enabled:
        raise ValueError("Context parallelism is not wired for Kimi K3 in this PR.")
    if parallel_dims.ep_enabled:
        raise ValueError("Expert parallelism is not wired for Kimi K3 in this PR.")
    entered = _drive_declarative_sharding(model, parallel_dims)
    if entered:
        from collections import Counter

        logger.info(
            "Declarative sharding: entered parallelize() on %d outermost Modules: %s",
            len(entered),
            dict(Counter(entered)),
        )

    if ac_config is not None:
        # Caveat for KDA layers: ``selective`` mode recomputes ops not
        # marked MUST_SAVE during backward; fla-core's chunk_kda kernel is
        # recomputed (~2x invocations). ``full`` mode is safer if you can
        # spare the recompute (see fla fused_norm_gate crash history).
        ac_config.build(dump_folder=dump_folder).apply(model)
        logger.info("Applied activation checkpointing to KimiDecoderLayer stack.")
    # torch.compile applied per-decoder-layer BEFORE FSDP wrap (so each
    # FSDP unit wraps a compiled subgraph). MoE for-loop expert path
    # is NOT compiled (torchtitan upstream has the same carve-out: see
    # apply_compile_sparse comment about unbacked symints in for-loop
    # fallback). fla-core ops (chunk_kda, ShortConvolution,
    # FusedRMSNormGated) are wrapped with torch.compiler.disable since
    # they're triton kernels that dynamo can't trace through.
    if compile_config.enable:
        _apply_compile_kimi_k3(model, compile_config)
        logger.info(
            "Compiled each KimiDecoderLayer with torch.compile (backend=%s).",
            compile_config.backend,
        )

    # NOTE cp_enabled belongs in this gate: torchtitan's "fsdp" mesh is
    # dp_shard x cp and FSDP is the mechanism that reduces param grads
    # over cp. Gating on dp alone silently skipped FSDP at dp_shard=1,
    # cp>1 -- every cp rank then trained an UNSYNCED replica on its own
    # seq shard (diverging, no error; per-rank grad_norm was the only
    # visible symptom). Upstream llama3 applies FSDP unconditionally.
    if (
        parallel_dims.dp_shard_enabled
        or parallel_dims.dp_replicate_enabled
        or parallel_dims.cp_enabled
    ):
        # The FSDP shard axis must be "fsdp" (= dp_shard x cp), never
        # "batch" (= dp_replicate x dp_shard, EXCLUDES cp): grads only
        # reduce over cp through FSDP's mesh. Mirrors upstream llama3's
        # ["dp_replicate", "fsdp"] selection.
        # veRL builds its own mesh and does not name one "fsdp" -- its axes
        # are ['pp','batch','loss','dp_replicate','cp','tp','ep','efsdp',
        # 'dp','dp_shard']. Fall back to composing the same product from the
        # axes it does have, so the semantics ("fsdp" = dp_shard x cp) are
        # preserved rather than silently narrowed to dp_shard.
        def _fsdp_axis(extra: list[str] | None = None):
            names = list(extra or [])
            try:
                return parallel_dims.get_mesh(names + ["fsdp"])
            except ValueError:
                axes = names + ["dp_shard"]
                if parallel_dims.cp_enabled:
                    axes.append("cp")
                return parallel_dims.get_mesh(axes)

        if parallel_dims.dp_replicate_enabled:
            dp_mesh = _fsdp_axis(["dp_replicate"])
        else:
            dp_mesh = _fsdp_axis()
        # Under EP, MoE expert parameters must shard via the *edp* mesh
        # (= dp_shard with the EP rank dim factored out) so FSDP's
        # mesh does not overlap EP's mesh on the same physical ranks.
        # See ``apply_fsdp`` docstring for the rationale; mirrors the
        # llama4 / deepseek_v3 path.
        edp_mesh = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
        if training.enable_cpu_offload:
            # FSDP CPUOffloadPolicy streams PARAMETERS to GPU per unit
            # but leaves buffers where they materialized (CPU) -- the
            # MoE router's expert_bias_E then meets GPU activations.
            # Lazily hoist CPU buffers to the compute device on first
            # forward (no-op afterwards).
            def _hoist_cpu_buffers(module, args):
                for m in module.modules():
                    for bname, buf in list(m.named_buffers(recurse=False)):
                        if buf is not None and buf.device.type == "cpu":
                            setattr(m, bname, buf.cuda())

            model.register_forward_pre_hook(_hoist_cpu_buffers)

        # Shard the tower before the decoder, as the core helper documents. A
        # local apply_fsdp_vision used to live here and was never called, so
        # the tower rode along inside the root wrap fully replicated. That is
        # invisible at the debug tower's 4 layers / hidden 256, but MoonViT-V2
        # ships at 447.4M against k3mini's 80.9M text side -- 5.5x the model it
        # serves -- so replicating it is not an option at real size.
        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is not None:
            apply_fsdp_to_vision_encoder(
                vision_tower,
                dp_mesh,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
                pp_enabled=parallel_dims.pp_enabled,
            )

        apply_fsdp(
            model,
            dp_mesh=dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=(parallelism.fsdp_reshard_after_forward),
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
        )
        logger.info(
            "Applied FSDP2 to Kimi Linear model (dp_shard=%d, dp_replicate=%d).",
            parallel_dims.dp_shard,
            parallel_dims.dp_replicate,
        )
    return model


def _drive_declarative_sharding(model: nn.Module, parallel_dims: ParallelDims) -> int:
    """Start upstream's declarative sharding from a plain-``nn.Module`` root.

    ``Module.parallelize`` recurses through its own children and looks THROUGH
    non-``Module`` containers, but something has to call it. Our containers
    (``KimiDecoderLayer``, ``KimiK3Model``, ``KimiMoE``) are plain ``nn.Module``, so
    nothing ever did -- which left the 64 modules that already carry a
    ``sharding_config`` declaring into the void. Measured with a probe: after this
    driver they hold DTensors with exactly the declared placements
    (``gate_proj`` Shard(0), ``down_proj`` Shard(1), ``q_a_proj`` Replicate).

    Already-parallelized subtrees are SKIPPED rather than re-entered, because
    ``Module.parallelize`` raises on a second call. Nothing in this PR parallelizes
    ahead of the driver, so the skip is inert here; it is load-bearing once the TP and
    EP PRs land, since each of those calls ``parallelize`` on subtrees of its own.

    Returns the class names entered, so a small count can be READ rather than guessed.
    """
    from torch.distributed.tensor import DTensor as _DTensor

    from torchtitan.protocols.module import Module

    def _already_distributed(m: nn.Module) -> bool:
        """Has the imperative plan (or an earlier pass) already distributed this subtree?

        ``_distribute_states`` raises "already a DTensor with placements ..." on a second
        distribution of the same weight. The TP and EP PRs each add an imperative plan
        covering some of the same modules the declarations do, and skipping what is
        already distributed makes this driver activate exactly the declarations those
        plans do NOT cover -- so imperative pieces can be deleted one at a time and the
        declarations take over as they go.
        """
        # recurse=False: the question is whether THIS module's own parameters
        # are distributed. parallelize() only touches what the module declares,
        # so a parent whose children the imperative plan covered is not done --
        # with recursion it counted as done and its own declared parameters were
        # never distributed.
        return any(isinstance(p, _DTensor) for p in m.parameters(recurse=False))

    entered: list[str] = []
    queue = list(model.children())
    while queue:
        child = queue.pop()
        if isinstance(child, Module) and not getattr(child, "_parallelized", False):
            if getattr(child, "_kimi_ep_parallelized", False):
                continue
            if not _already_distributed(child):
                child.parallelize(parallel_dims)
                entered.append(type(child).__name__)
                continue
            # Partially covered: descend so the children the plan missed still get theirs.
        queue.extend(child.children())
    return entered


def apply_fsdp(
    model: nn.Module,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims=None,
    edp_mesh_dims=None,
    enable_symm_mem: bool = False,
) -> None:
    """FSDP2 for the Kimi models: the shared helper, plus the AttnRes tail.

    This was a 182-line copy of ``distributed.fsdp.apply_fsdp_to_decoder`` and had
    fallen behind it in five ways -- no ``enable_symm_mem``, no ``dp_mesh_dims``
    flattening under full_dtensor, no ``edp_mesh_dims``, no ``Shard(1)`` refinement when
    the FSDP degree exceeds the expert count, and no EP prefetch wiring. It also
    nested-wrapped routed experts on ``edp_mesh`` to work around per-param meshes not
    being expressible in ``shard_placement_fn``; the helper now does that properly via
    ``ShardPlacementResult``, so the workaround is obsolete rather than merely duplicated.

    Delegation is possible without renaming anything: the helper only READS the names it
    needs, so ``UpstreamFSDPNames`` supplies them as properties and no FQN or checkpoint
    key moves. See that class for why aliases rather than a rename.

    What remains ours is the AttnRes output tail.
    """
    # The tail is wrapped BEFORE delegating, for two reasons. FSDP2 requires a child unit
    # to exist before its parent, and the helper's last act is to wrap the root -- which
    # would otherwise absorb these two top-level modules into the root unit.
    #
    # They must share ONE unit, and that is load-bearing rather than an optimization:
    # block_attn_res reads ``output_res_proj.weight`` directly as the pseudo-query
    # instead of calling ``proj(...)``, so no forward hook fires on it and FSDP2 warns that
    # it "did not run forward before backward". Pairing it with output_res_norm is what
    # makes that correct -- norm IS called one line earlier (``K = norm(V)``) and triggers
    # the shared param group's all-gather, so the weight is unsharded by the time it is
    # read. Do not move the weight access above the norm call. Verified on both ranks at
    # dp2.
    attn_res_tail = [
        m
        for m in (
            getattr(model, "output_res_proj", None),
            getattr(model, "output_res_norm", None),
        )
        if m is not None
    ]
    if attn_res_tail:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )
        tail_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
        if dp_mesh_dims is not None:
            tail_config["dp_mesh_dims"] = dp_mesh_dims
        if cpu_offload:
            tail_config["offload_policy"] = CPUOffloadPolicy()
        fully_shard(
            attn_res_tail,
            **tail_config,
            reshard_after_forward=(reshard_after_forward_policy == "always"),
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype,
        reduce_dtype,
        pp_enabled,
        cpu_offload=cpu_offload,
        reshard_after_forward_policy=reshard_after_forward_policy,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=enable_symm_mem,
    )


_fla_dynamo_carveout_done = False


def _disable_dynamo_on_fla_ops() -> None:
    """Make the fla kernels and the AttnRes read opaque to dynamo.

    Split out of :func:`_apply_compile_kimi_k3` for two reasons. This is global
    state -- class attributes and module bindings, nothing owned by the model
    passed in -- so applying it per model part would wrap each function once per
    part under PP. And separating it is what makes the carve-out observable at
    all: a caller can compare a rebound name against fla's original. The version
    of this code that discarded ``torch.compiler.disable``'s return value did
    nothing, and nothing could see that.
    """
    global _fla_dynamo_carveout_done
    if _fla_dynamo_carveout_done:
        return
    _fla_dynamo_carveout_done = True

    from fla.modules import FusedRMSNormGated, ShortConvolution

    # Mark triton ops as opaque to dynamo. recursive=True so dynamo
    # also stays out on re-entry from autograd backward (otherwise
    # fla's backward kernels trip on cuda_utils.get_device_properties
    # and lru_cache decorators inside fused_norm_gate).
    #
    # torch.compiler.disable RETURNS a wrapper; it does not mark the function
    # in place. Discarding the return left all three ops fully traceable, so
    # this carve-out did nothing. Rebinding has to happen on the module that
    # CALLS them -- model.py's own `from fla.ops.kda import ...` bindings --
    # for the same reason spelled out for block_attn_res below. Patching
    # fla.ops.kda alone would not be seen by an already-imported name.
    from torchtitan.models.kimi_k3 import model as _model_mod

    for _name in ("chunk_kda", "fused_recurrent_kda", "fused_kda_gate"):
        setattr(
            _model_mod,
            _name,
            torch.compiler.disable(getattr(_model_mod, _name), recursive=True),
        )
    for cls in (ShortConvolution, FusedRMSNormGated):
        cls.forward = torch.compiler.disable(cls.forward, recursive=True)

    # block_attn_res: TP path requires DTensor.to_local on proj.weight to
    # unmix DTensor and plain Tensor in the einsum. dynamo's fake-tensor
    # mode doesn't trace through the conditional to_local cleanly (it
    # propagates DTensor type past the isinstance branch and the einsum
    # call sees mixed DTensor + plain). Easiest fix: graph-break at the
    # block_attn_res entry, the function runs eagerly. block_attn_res is
    # a single softmax + two einsums, so eager dispatch doesn't lose
    # meaningful compile gains.
    #
    # We patch in-place at every callsite's bound module -- both the
    # source module (attn_res) and its importer (attn_res_model) --
    # because each ``from .attn_res import block_attn_res`` creates an
    # independent binding that wouldn't be touched by patching the
    # source module alone.
    from torchtitan.models.kimi_k3 import (
        attn_res as _src,
        attn_res_model as _kimi_attn_res_mod,
    )

    disabled = torch.compiler.disable(_src.block_attn_res, recursive=True)
    _src.block_attn_res = disabled
    _kimi_attn_res_mod.block_attn_res = disabled

    # KDA forward: also opaque to dynamo. Body is all fla-core triton
    # kernels (already disabled) plus simple linears. Under TP, the
    # forward starts with ``_to_local_if_dtensor(x)`` to strip the
    # incoming DTensor; dynamo's fake-tensor mode doesn't always
    # propagate the type-narrowing of an ``isinstance`` branch through
    # the linear ops that follow, so the q_proj call sees the original
    # DTensor and errors with "mixed Tensor and DTensor". Disabling
    # KDA forward eagerly runs the to_local + the linears, which is
    # negligible compute cost on top of the already-eager triton
    # kernels.
    from torchtitan.models.kimi_k3.model import KimiDeltaAttention

    KimiDeltaAttention.forward = torch.compiler.disable(
        KimiDeltaAttention.forward,
        recursive=True,
    )


def _apply_compile_kimi_k3(model: nn.Module, compile_config: CompileConfig) -> None:
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
    the attention module triggers a recompile per attention class, and once
    the limit is hit dynamo silently falls back to eager for
    affected frames. We bump recompile_limit + cache_size_limit so
    each layer-flavor compiles cleanly on first hit and stays cached.
    """
    _disable_dynamo_on_fla_ops()

    # Allow MoE token-choice routing's data-dependent control flow.
    torch._dynamo.config.capture_scalar_outputs = True
    # Eager AC <-> compile divergence acceptance (matches upstream).
    # Only available in torch nightly; skip silently on stable builds.
    if hasattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint"):
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    # KDA + MLA layers each compile separately; we have up to L layer
    # flavors plus permutations. 64 leaves comfortable headroom for
    # all per-layer specializations without thrashing.
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64

    for _, layer in model.layers.named_children():
        layer.compile(backend=compile_config.backend, fullgraph=False)
