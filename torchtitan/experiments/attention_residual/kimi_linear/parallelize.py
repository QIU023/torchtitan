# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism application for Kimi Linear models.

Supported parallelism dimensions (as of Phase 6):

* **FSDP2 full-shard** — primary path, modeled on
  ``torchtitan.models.llama3.parallelize.apply_fsdp``. Adapted to Kimi's
  module names (``embed_tokens``, ``norm``, ``lm_head``, ``layers`` as
  ``nn.ModuleList``).
* **TP** (Phase 6 A3) — DSv3-style plan in ``apply_tp_kimi_linear``,
  matched to Kimi's MLA + KDA + MoE layout. All boundary tensors
  use_local_output so PP send/recv, AttnRes stacking, and fla-core
  triton kernels see plain tensors.
* **EP** (Phase 6 A6) — Expert Parallel for KimiMoE via
  ``apply_ep_kimi_linear``; all-to-all dispatch/combine on the EP mesh.
* **PP** — via the Phase 3 cache adapter in ``pipeline_adapter.py``;
  PP rank assignment is in torchtitan core, scheduling in the
  ``pipeline_kimi_linear_with_cache_adapter`` wrapper.
* **torch.compile** (Phase 4 onwards) — per-decoder-layer compile via
  ``_apply_compile_kimi_linear``; MoE for-loop and fla-core triton ops
  are carved out.
* **Activation checkpointing** — applied via shared
  ``torchtitan.distributed.activation_checkpoint.apply_ac`` since the
  Kimi decoder layer registry matches the llama3 ``model.layers``
  iteration pattern. Honors all upstream modes (``selective``,
  ``full``, ``memory_budget``, ``none``).

**Not supported**:

* **CP** — blocked on fla-core's ``chunk_kda`` triton kernel lacking
  ring-recurrence over CP shards; see comment near the CP guard below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.distributed.tensor_parallel import NoParallel

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
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
    """Apply the configured parallelism plan to a Kimi Linear model.

    Wires (in order, before FSDP wrap): TP → CP (raises) → EP → AC →
    compile → FSDP/HSDP. AC is applied before compile so the compiled
    subgraph is the checkpointed unit (matches upstream llama3/qwen3
    ordering).

    CP raises ``NotImplementedError`` — see the inline comment near the
    CP guard for the fla-core ``chunk_kda`` ring-recurrence blocker.
    """
    # Enable TF32 tensor cores for fp32 matmuls (loss aggregation,
    # optimizer master weight updates, fp32 RoPE etc.). bf16 path is
    # unaffected. Speedup ~5-10% on fp32 ops, no measurable accuracy
    # impact at our scale.
    torch.set_float32_matmul_precision("high")

    if parallel_dims.tp_enabled:
        # Phase 6 A3: TP plan modeled on ``deepseek_v3/parallelize.py``.
        # Key idea: every module boundary in the forward emits a plain
        # Tensor (use_local_output=True / output_layouts=Replicate())
        # so:
        #   * the stack inside ``block_attn_res`` aggregates plain
        #     Tensors uniformly across MLA-output / KDA-output / partial
        #     blocks (no mixed-dispatch errors);
        #   * fla-core triton kernels inside KDA see plain Tensors and
        #     dispatch normally;
        #   * SDPA in ``KimiMLAInnerAttention`` runs on plain Tensors
        #     thanks to ``prepare_module_input(use_local_output=True)``.
        #
        # The TP collectives still fire — ColwiseParallel produces
        # DTensor(Shard) intermediates internally and RowwiseParallel
        # all-reduces on the way out before to_local. We just keep
        # boundary types plain so PP send/recv, AttnRes block stacking,
        # and triton kernels never see a mixed-mesh tensor.
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp_kimi_linear(
            model, tp_mesh,
            skip_expert_params=parallel_dims.ep_enabled,
        )
        # Stash the TP mesh on the model so AttnRes top-level forward
        # can DTensor-ify PP-received block tensors when they arrive
        # plain (PP P2P uses raw send/recv, so mid-stage receives
        # plain tensors that need to be converted back into the TP
        # mesh's local view before aggregation).
        model._tp_mesh = tp_mesh
        logger.info(
            "Applied DSv3-style TP plan tp_degree=%d.", parallel_dims.tp,
        )
    if parallel_dims.cp_enabled:
        # Phase 6 CP: blocked on fla-core. The kimi_linear backbone
        # alternates KDA (3:1 ratio) and MLA layers. KDA's forward
        # path uses fla-core's chunk_kda triton kernel, which runs a
        # causal recurrence over the seq dim. CP shards the seq dim
        # across ranks; chunk_kda would see only seq_len/cp tokens
        # per rank and the recurrence state across rank boundaries
        # would be lost. Making KDA CP-correct requires a ring-
        # recurrence variant of chunk_kda that exchanges state between
        # adjacent CP ranks at the chunk boundary — which lives in
        # fla-core upstream (https://github.com/fla-org/flash-linear-attention),
        # not in torchtitan or this experiment.
        #
        # MLA + dense MLP would compose with CP via the standard
        # torchtitan path (apply_cp_to_attention_module + SDPA
        # dispatcher), but applying CP only to MLA while replicating
        # the seq across KDA layers requires a per-layer all-gather
        # at the KDA boundary — a non-trivial wrapper that is out of
        # scope here.
        #
        # Status: CP is documented as out-of-scope until fla-core
        # ships ring-attention KDA. Non-AttnRes / MLA-only flavors
        # would compose with CP cleanly.
        raise NotImplementedError(
            "CP is not currently supported for kimi_linear "
            "(KDA layers' fla-core chunk_kda kernel does not "
            "implement ring-recurrence over CP shards). "
            "Track upstream fla-core for ring-KDA support; "
            "until then, run with context_parallel_degree=1."
        )
    if parallel_dims.ep > 1:
        # Phase 6 A6: Expert Parallel for Kimi MoE layers. The
        # KimiMoE module wraps torchtitan.models.common.moe.MoE as
        # self._moe; the expert ModuleList is at self._moe.experts.
        # Apply standard ExpertParallel() to that experts container,
        # which fires all-to-all on the EP mesh for token dispatch +
        # combine. Cache adapter delta accumulation interacts with
        # MoE only at the block boundary (after FFN residual add),
        # so EP routing within the FFN body is transparent to the
        # AttnRes block-commit logic.
        ep_mesh = parallel_dims.get_mesh("ep")
        apply_ep_kimi_linear(model, ep_mesh)
        logger.info(
            "Applied EP plan (per-MoE-layer ExpertParallel) ep_degree=%d.",
            parallel_dims.ep,
        )
    if ac_config.mode != "none":
        # Kimi's decoder layers live at ``model.layers`` as ``nn.ModuleList``,
        # matching llama3's iteration pattern, so the shared apply_ac
        # implementation works without a per-module specialization.
        #
        # Caveat for KDA layers: ``selective`` mode recomputes ops not
        # marked MUST_SAVE during backward. fla-core's chunk_kda triton
        # kernel is recomputed (it's not in the SAVE set). This works
        # but increases the kernel's invocation count by ~2x — if you're
        # hitting the device-side assert in
        # ``fla/modules/fused_norm_gate.py`` (KDA crash, see phase5
        # task #46), AC will trigger it more often. ``full`` mode is
        # safer if you can spare the extra recompute, since it saves
        # only the layer-input and recomputes the whole block linearly.
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=compile_config.enable,
            base_folder=dump_folder,
        )
        logger.info(
            "Applied activation checkpointing mode=%s to KimiDecoderLayer stack.",
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
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
        )
        logger.info(
            "Applied FSDP2 to Kimi Linear model (dp_shard=%d, dp_replicate=%d).",
            parallel_dims.dp_shard,
            parallel_dims.dp_replicate,
        )
    return model


def _patch_fla_for_dtensor() -> None:
    """Patch ShortConvolution + FusedRMSNormGated forwards to be DTensor-safe.

    Both classes wrap fla-core triton kernels (``causal_conv1d``,
    ``fused_norm_gated``) that take raw tensor pointers and don't
    dispatch through DTensor. Under TP, KDA's self_attn is
    NoParallel-wrapped — child params are DTensor(Replicate) on tp_mesh
    and inputs arrive as DTensor too. Calling the triton kernel with
    DTensor crashes.

    The patch wraps each class's ``forward`` to:
    1. ``to_local`` the input ``x`` (preserving DTensor mesh+placements
       for re-wrap at the output) and any positional / keyword args
       that happen to be DTensors.
    2. Run the original forward unchanged. The forward calls the
       kernel with ``self.weight`` (still a DTensor on the module, but
       autograd-traced via to_local under the hood — see below) — so
       we additionally redirect ``self.weight`` access to its local
       view via a transient property override on the instance during
       the call.
    3. Re-wrap the output as DTensor on the same mesh+placements so the
       parent's (NoParallel) prepare_output hook receives a DTensor.

    Critically we do NOT swap ``self._parameters`` to a fresh
    ``nn.Parameter(local)`` — that would break autograd (the new
    parameter has no link back to the DTensor original, so kernel
    backward would write into a temporary local tensor and the
    DTensor's grad would never see the contribution). Instead we mask
    the attribute lookup so the kernel sees a plain ``self.weight`` for
    the duration of forward, but autograd's saved-tensors reference the
    local-view of the DTensor (which is differentiable through to_local).

    Patch is idempotent: ``cls._fla_orig_forward`` is set on first patch.
    """
    from fla.modules import FusedRMSNormGated, ShortConvolution

    def _maybe_local(t):
        if isinstance(t, DTensor):
            return t.to_local()
        return t

    def _make_patch(cls):
        # Idempotent: skip if already patched.
        if getattr(cls, "_fla_orig_forward", None) is not None:
            return
        orig = cls.forward
        cls._fla_orig_forward = orig

        def _patched(self, x, *args, **kwargs):
            in_mesh = None
            in_placements = None
            if isinstance(x, DTensor):
                in_mesh = x.device_mesh
                in_placements = x.placements
                x = x.to_local()
            args = tuple(_maybe_local(a) for a in args)
            kwargs = {k: _maybe_local(v) for k, v in kwargs.items()}

            # Override attribute lookup for ``weight`` (and ``bias`` if
            # present) on this instance for the duration of the forward
            # call. We use a per-call dict that the descriptor reads;
            # restoring on exit is automatic via the finally block.
            saved_attrs: dict[str, object] = {}
            for name in ("weight", "bias"):
                if name in self._parameters:
                    p = self._parameters[name]
                    if p is not None and isinstance(p, DTensor):
                        # to_local() returns a Tensor that is
                        # differentiable w.r.t. the DTensor: backward
                        # propagates the local grad up to the DTensor's
                        # grad through the AsStridedBackward path.
                        saved_attrs[name] = p
                        # Bypass nn.Module.__setattr__'s parameter
                        # handling by writing directly into __dict__.
                        # This makes ``self.weight`` resolve to a plain
                        # Tensor for the lookup chain inside the
                        # original forward, while ``self._parameters``
                        # still references the DTensor (so
                        # named_parameters and FSDP iteration are
                        # unaffected).
                        self.__dict__[name] = p.to_local()
            try:
                out = orig(self, x, *args, **kwargs)
            finally:
                for name in saved_attrs:
                    # Restore the attribute lookup so subsequent
                    # accesses fall back to ``self._parameters[name]``.
                    self.__dict__.pop(name, None)

            def _rewrap(t):
                if (
                    in_mesh is not None
                    and in_placements is not None
                    and isinstance(t, torch.Tensor)
                    and not isinstance(t, DTensor)
                ):
                    return DTensor.from_local(
                        t, in_mesh, in_placements, run_check=False,
                    )
                return t

            if isinstance(out, tuple):
                return tuple(_rewrap(o) for o in out)
            return _rewrap(out)

        cls.forward = _patched

    _make_patch(ShortConvolution)
    _make_patch(FusedRMSNormGated)


def apply_tp_kimi_linear(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    skip_expert_params: bool = False,
) -> None:
    """Phase 6 A3 TP plan for kimi_linear, modeled on
    ``deepseek_v3/parallelize.py``.

    Design constraint: every module-boundary tensor in this model is a
    plain ``torch.Tensor`` (not DTensor). The motivation is composability
    with three subsystems that don't dispatch through DTensor:

    * **fla-core triton kernels** in :class:`KimiDeltaAttention`
      (``causal_conv1d`` inside ``ShortConvolution``, ``fused_kda_gate``,
      ``chunk_kda``, ``FusedRMSNormGated``) — opaque to DTensor.
    * **PP P2P send/recv** at AttnRes block boundaries — only plain
      Tensors are sendable; DTensor wrappers don't survive P2P.
    * **AttnRes ``torch.stack``** in :func:`block_attn_res` — stacking
      mixed plain/DTensor inputs raises ``mixed Tensor and DTensor``.

    To satisfy all three, every Colwise/Rowwise plan emits
    ``use_local_output=True`` (or ``output_layouts=Replicate()`` plus
    ``use_local_output=True``), and every NoParallel call passes
    ``local_output_grad_placements=(Replicate(),)`` so the output is
    converted back to a plain Tensor at the module boundary. The TP
    collectives still fire (Colwise → DTensor(Shard) inside Linear's
    forward, Rowwise → DTensor(Partial) → all-reduce → Replicate →
    to_local).

    Per-module wrapping (DSv3-aligned):

    * **Top-level**:
      - ``embed_tokens``: RowwiseParallel (vocab dim sharded; no SP).
      - ``norm``: NoParallel.
      - ``lm_head``: ColwiseParallel (vocab dim sharded; output local).
      - ``final_attn_res_proj`` / ``final_attn_res_norm`` (AttnRes
        only): NoParallel — output dim 1 / RMSNorm cannot be sharded.

    * **MLA layer (KimiMLAAttention)**:
      - ``q_proj``: ColwiseParallel (out = H * q_head_dim).
      - ``kv_a_proj_with_mqa``: NoParallel (DSv3 pattern — the output
        is split into two halves of different sizes, which is not
        compatible with sharding on the concatenated last dim).
      - ``kv_a_layernorm``: NoParallel (single-half normalization).
      - ``kv_b_proj``: ColwiseParallel (out = H * (qk_nope + v_head_dim)).
      - ``inner_attention``: prepare_module_input(use_local_output=True)
        with sequence-axis placements — strips DTensor wrapping before
        SDPA's mem-efficient cutlass kernel sees q/k/v.
      - ``o_proj``: RowwiseParallel (in_features sharded, output
        all-reduced + Replicate).

    * **KDA layer (KimiDeltaAttention)**: NOT TP-wrapped. KDA's body is
      almost entirely fla-core triton kernels; sharding heads requires
      a ring-recurrence variant of ``chunk_kda`` that fla-core doesn't
      provide. Leaving KDA self_attn unwrapped means its parameters
      remain plain Tensors (FSDP shards them on the FSDP axis,
      replicated across TP ranks — accepting compute redundancy on the
      TP axis for a clean kernel-safety story). The KDA forward
      defensively strips any incoming DTensor (see
      ``_to_local_if_dtensor`` in model.py).

    * **dense MLP**: gate_proj / up_proj ColwiseParallel; down_proj
      RowwiseParallel(output_layouts=Replicate(), use_local_output=True).

    * **MoE FFN**: NoParallel on the ``ffn`` container (EP handles the
      real parallelization on a separate axis).

    * **Per-layer norms** (``input_layernorm``, ``post_attention_layernorm``):
      NoParallel.

    * **AttnRes per-layer modules** (``attn_res_proj``, ``attn_res_norm``,
      ``mlp_res_proj``, ``mlp_res_norm``): NoParallel each.

    Collectives per forward pass (summed across layers):
    - 1 all-reduce per dense-MLP layer (down_proj rowwise)
    - 1 all-reduce per MLA layer (o_proj rowwise)
    - KDA layers fire no TP collectives.
    Plus the symmetric backward all-reduces.
    """
    # Plain-output NoParallel: ``output_layout=Replicate()`` (default)
    # plus ``local_output_grad_placements=(Replicate(),)`` produces a
    # plain torch.Tensor at the module exit. The grad placement is
    # Replicate so the backward all-reduces dY back to Replicate before
    # entering the module's backward — matching the forward-side
    # Replicate placement of the params.
    no_par_local = NoParallel(local_output_grad_placements=(Replicate(),))

    # fla-core triton kernels (causal_conv1d in ShortConvolution,
    # fused_norm_gated in FusedRMSNormGated) do not dispatch through
    # DTensor: they call triton kernels directly on the data pointers
    # of x and weight. Under TP, KDA's self_attn is NoParallel-wrapped,
    # so ShortConvolution and FusedRMSNormGated submodules have DTensor
    # weights and receive DTensor inputs — which would crash inside
    # the triton call. We patch their forward methods to to_local both
    # input and weight at the kernel boundary, then from_local the
    # output back so downstream ops (which expect DTensor under the
    # NoParallel wrap) compose correctly.
    #
    # The patch is applied in-place on the class; the patch is
    # idempotent (re-patching a previously patched class is safe — the
    # original-forward attr is set once at first patch).
    _patch_fla_for_dtensor()

    # Top-level layout: embed, output norm, lm_head.
    # Both embed and lm_head emit plain Tensors (use_local_output=True)
    # so the AttnRes top-level forward composes cleanly with the
    # block-stacking path.
    parallelize_module(
        model,
        tp_mesh,
        {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
            "norm": no_par_local,
            "lm_head": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # AttnRes-only top-level tail modules. Linear(d, 1) and RMSNorm(d):
    # neither is shardable, both must live on the TP mesh.
    if getattr(model, "final_attn_res_proj", None) is not None:
        parallelize_module(
            model,
            tp_mesh,
            {
                "final_attn_res_proj": no_par_local,
                "final_attn_res_norm": no_par_local,
            },
        )

    # MLA inner_attention input plan (DSv3 mirror): q/k/v arrive sharded
    # on the head axis (transposed to dim 1 inside KimiMLAAttention.forward
    # before SDPA), use_local_output=True converts them to plain Tensors
    # before the SDPA kernel dispatcher sees them.
    inner_attn_plan = PrepareModuleInput(
        input_layouts=(Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=True,
    )

    # Per-layer plan. Each layer is a KimiDecoderLayer (or AttnRes
    # subclass with attn_res_proj + attn_res_norm).
    for layer in model.layers.values():
        is_moe = bool(getattr(layer, "is_moe", False))
        is_kda = bool(getattr(layer, "is_linear_attn", False))

        # input_layernorm and post_attention_layernorm: plain NoParallel
        # (DTensor output). Downstream MLA forward consumes DTensor
        # naturally; downstream KDA strips DTensor at entry via
        # _to_local_if_dtensor; downstream dense MLP's prepare_input
        # accepts both. Plain NoParallel is the most natural choice.
        plan: dict[str, object] = {
            "input_layernorm": NoParallel(),
            "post_attention_layernorm": NoParallel(),
        }

        if is_kda:
            # KDA: register self_attn as NoParallel so all child params
            # (q/k/v/o projections, conv1d weights, A_log, dt_bias, etc.)
            # become DTensors on tp_mesh (Replicate). This is required
            # for FSDP+TP composability: ``clip_grad_norm_`` stacks
            # per-param grad norms across the parameter list, and stack
            # fails if some norms live on (fsdp, tp) mesh and others on
            # (fsdp,) mesh only.
            #
            # Inside KDA forward, all linears (q_proj/k_proj/v_proj,
            # f/g/b projections) receive DTensor input from input_layernorm
            # and DTensor weights from this NoParallel wrap → produce
            # DTensor outputs. fla-core triton kernels (causal_conv1d
            # inside ShortConvolution, fused_kda_gate, chunk_kda,
            # FusedRMSNormGated) don't dispatch through DTensor, so we
            # patch ShortConvolution.forward and FusedRMSNormGated.forward
            # below to to_local their weight + input at the kernel
            # boundary, then from_local the output. fused_kda_gate and
            # chunk_kda are called explicitly in KDA forward — those
            # to_local-and-call wrappers live in model.py.
            #
            # local_output_grad_placements=Replicate so the output is
            # to_local'd at the module exit. This matches MLA's o_proj
            # (use_local_output=True / output_layouts=Replicate) and the
            # dense MLP's down_proj (RowwiseParallel use_local_output)
            # so attn_out is plain Tensor everywhere — partial_block
            # accumulation and AttnRes block_attn_res both see uniform
            # plain Tensors.
            plan["self_attn"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
        else:
            # MLA layer: DSv3-style plan.
            # NOTE: ``kv_a_proj_with_mqa`` is NOT sharded — its output
            # is split into ``[kv_lora_rank, qk_rope_head_dim]`` halves
            # of unequal size, and downstream ``kv_a_layernorm`` only
            # sees the kv_lora half. Sharding the concatenated last dim
            # would corrupt the split. NoParallel here matches DSv3's
            # ``wkv_a`` (kv_a_proj_with_mqa). The output is plain Tensor
            # so the inline torch.split runs on a regular tensor.
            # MLA: every submodule except inner_attention/o_proj
            # emits DTensor (Shard or Replicate) — the MLA forward's
            # split/cat/view/transpose/expand operations all dispatch
            # through DTensor. Only at SDPA (inner_attention) we
            # convert to plain via use_local_output=True; o_proj emits
            # plain to match the rest of the model's plain-boundary
            # convention.
            plan.update(
                {
                    "self_attn.q_proj": ColwiseParallel(
                        use_local_output=False,
                    ),
                    # NoParallel (no local_output_grad_placements): output
                    # stays as a DTensor(Replicate) so the downstream
                    # split into [kv_lora, qk_rope] halves and the
                    # subsequent kv_a_layernorm + cat with k_pass_expanded
                    # all run consistently in DTensor space (mirrors DSv3's
                    # ``wkv_a`` registration).
                    "self_attn.kv_a_proj_with_mqa": NoParallel(),
                    "self_attn.kv_a_layernorm": NoParallel(),
                    "self_attn.kv_b_proj": ColwiseParallel(
                        use_local_output=False,
                    ),
                    "self_attn.inner_attention": inner_attn_plan,
                    "self_attn.o_proj": RowwiseParallel(
                        output_layouts=Replicate(),
                        use_local_output=True,
                    ),
                }
            )

        # FFN path.
        if not is_moe:
            ffn = getattr(layer, "ffn", None)
            if ffn is None:
                raise ValueError(
                    f"layer {layer.layer_idx}: missing dense ffn"
                )
            for name in ("gate_proj", "up_proj", "down_proj"):
                if not hasattr(ffn, name):
                    raise ValueError(
                        f"layer {layer.layer_idx} dense ffn missing '{name}'"
                    )
            plan.update(
                {
                    "ffn.gate_proj": ColwiseParallel(use_local_output=False),
                    "ffn.up_proj": ColwiseParallel(use_local_output=False),
                    "ffn.down_proj": RowwiseParallel(
                        output_layouts=Replicate(),
                        use_local_output=True,
                    ),
                }
            )
        else:
            # MoE: register each leaf submodule with NoParallel so all
            # params live on tp_mesh as DTensor(Replicate). The
            # torchtitan common MoE forward (at moe.py:403) to_local's
            # its DTensor input at entry; the gate/experts/shared_experts
            # then receive plain x. NoParallel-wrapping each leaf
            # converts plain x → DTensor at the leaf's prepare_input
            # hook, runs the leaf's forward on DTensor (params are also
            # DTensor → matmul works), and converts the output back to
            # plain via local_output_grad_placements=Replicate.
            #
            # GroupedExperts.forward and KimiMLP gate_proj/up_proj/down_proj
            # all dispatch normally on DTensor inputs because their ops
            # are standard nn.Linear / F.linear (DTensor-friendly).
            # GroupedExperts additionally to_local's its DTensor params
            # before the grouped_mm kernel (see moe.py:100-111), so it's
            # already DTensor-aware.
            #
            # The reason we DON'T NoParallel the whole ``ffn`` container:
            # MoE.forward stripping x via to_local at line 410 happens
            # AFTER the parent's prepare_input. If we NoParallel-wrapped
            # ffn or _moe, the input would be DTensor at MoE.forward
            # entry; to_local would convert to plain; then the gate
            # would receive plain. With gate.weight as DTensor (from
            # NoParallel descent), gate(plain_x) errors. Wrapping the
            # gate ITSELF (instead of the parent) keeps the boundary
            # correct: input arrives plain, gate's prepare_input wraps
            # it back to DTensor, the matmul stays DTensor × DTensor.
            #
            # Required for FSDP+TP composability: ``clip_grad_norm_``
            # stacks per-param grad norms across the parameter list.
            # If MoE params live on (fsdp,) mesh and MLA params live on
            # (fsdp, tp) mesh, the stack fails. NoParallel wrapping
            # promotes MoE params to (fsdp, tp) after FSDP wrap.
            ffn = getattr(layer, "ffn", None)
            if ffn is None or not hasattr(ffn, "_moe"):
                raise ValueError(
                    f"MoE layer {layer.layer_idx}: missing ffn._moe"
                )
            # router.gate: NoParallel boundary — gate(plain x) becomes
            # gate(DTensor x), gate.weight is DTensor, gate forward
            # produces DTensor, exits as plain via local_output.
            plan["ffn._moe.router.gate"] = no_par_local
            # experts (GroupedExperts): the forward already to_local's
            # its DTensor params before the grouped_mm kernel call (see
            # moe.py:100-111). Wrapping the module with NoParallel
            # would also wrap the input, but the kernel needs PLAIN
            # input × PLAIN weight (same as the to_local'd weights).
            # So we don't wrap experts with NoParallel; instead we
            # promote w1/w2/w3 to DTensor(Replicate) manually below
            # (after parallelize_module).
            #
            # shared_experts (KimiMLP): each leaf Linear must be
            # individually wrapped as no_par_local so it accepts the
            # plain input from MoE.forward (post-to_local at line 410)
            # while keeping its weight as DTensor on tp_mesh.
            shared = getattr(ffn._moe, "shared_experts", None)
            if shared is not None:
                # Treat shared_experts as a small dense MLP. Its forward
                # is called as ``self.shared_experts(x)`` from MoE; x is
                # plain (already to_local'd at moe.py:410). Wrapping each
                # leaf Linear individually as no_par_local keeps params
                # on tp_mesh while preserving the plain-Tensor I/O.
                #
                # Note: the FeedForward common module names its leaves
                # ``w1, w2, w3`` (not gate/up/down) — see
                # torchtitan/models/common/feed_forward.py.
                for n in ("w1", "w2", "w3"):
                    if hasattr(shared, n):
                        plan[f"ffn._moe.shared_experts.{n}"] = no_par_local

        # AttnRes per-layer modules: each layer has TWO pseudo-queries
        # + TWO RMSNorms, all NoParallel.
        for name in (
            "attn_res_proj", "attn_res_norm",
            "mlp_res_proj",  "mlp_res_norm",
        ):
            if hasattr(layer, name) and getattr(layer, name) is not None:
                plan[name] = no_par_local

        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

        # MoE experts (GroupedExperts.w1/w2/w3): distribute as
        # DTensor(Replicate) without installing module hooks. The
        # GroupedExperts.forward already to_local's its DTensor params
        # before the grouped_mm kernel; wrapping the module would cause
        # plain × plain mismatch (since the input x is plain too).
        #
        # When ``skip_expert_params=True`` (caller has EP enabled), do
        # NOT touch experts — leave them as plain Tensors so the EP
        # path (apply_ep_kimi_linear) can DTensor-shard them on
        # ``ep_mesh`` without hitting cross-mesh redistribute errors.
        # This mirrors llama4's design: TP plan touches router.gate +
        # shared_experts only; routed experts are EP/ETP territory.
        if is_moe and not skip_expert_params:
            ffn = layer.ffn
            experts = ffn._moe.experts
            for name in ("w1", "w2", "w3"):
                p = getattr(experts, name, None)
                if p is not None and not isinstance(p, DTensor):
                    setattr(
                        experts,
                        name,
                        nn.Parameter(
                            distribute_tensor(
                                p.data, tp_mesh, [Replicate()],
                            ),
                            requires_grad=p.requires_grad,
                        ),
                    )

    # Final sweep: any remaining plain Tensor parameters (typically
    # ``A_log``, ``dt_bias`` on KDA layers' self_attn that NoParallel
    # didn't catch because they're bare ``nn.Parameter``s on the
    # ``self_attn`` module rather than children) — promote them to
    # DTensor(Replicate) on tp_mesh. This is required so that under
    # FSDP+TP all params live on the same (fsdp, tp) 2D mesh, satisfying
    # the cross-param mesh consistency check inside
    # ``clip_grad_norm_``'s ``torch.stack`` call.
    #
    # When ``skip_expert_params=True``, build a set of routed-expert
    # param ids first and skip them — they belong to the EP mesh, not
    # the TP mesh. The clip_grad_norm cross-mesh check still passes
    # because EP-sharded params live on a clean ``ep_mesh`` and the
    # rest live on ``tp_mesh``; both are 1D, so torch.stack handles
    # them via the per-mesh path.
    expert_param_ids: set[int] = set()
    if skip_expert_params:
        for layer in model.layers.values():
            if not bool(getattr(layer, "is_moe", False)):
                continue
            ffn = getattr(layer, "ffn", None)
            if ffn is None or getattr(ffn, "_moe", None) is None:
                continue
            for p in ffn._moe.experts.parameters():
                expert_param_ids.add(id(p))
    for module in model.modules():
        for name, p in list(module._parameters.items()):
            if p is not None and not isinstance(p, DTensor) \
                    and id(p) not in expert_param_ids:
                module._parameters[name] = nn.Parameter(
                    distribute_tensor(p.data, tp_mesh, [Replicate()]),
                    requires_grad=p.requires_grad,
                )


def apply_ep_kimi_linear(model: nn.Module, ep_mesh: DeviceMesh) -> None:
    """Phase 6 A6 Expert Parallel plan for kimi_linear MoE flavors.

    Applies ``ExpertParallel()`` to every MoE layer's expert container.
    The KimiMoE module wraps the torchtitan common MoE as ``self._moe``;
    its ``experts`` field is the ModuleList that EP shards across the
    EP mesh, with token dispatch + combine via all-to-all collectives
    on that mesh.

    Layers without MoE (``layer.is_moe == False``, i.e. dense MLP at
    the first ``first_k_dense_replace`` indices) are skipped — they
    have no experts to shard.
    """
    from torchtitan.distributed.expert_parallel import ExpertParallel

    plan = ExpertParallel()
    moe_layers_wrapped = 0
    for layer in model.layers.values():
        if not bool(getattr(layer, "is_moe", False)):
            continue
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        # KimiMoE wraps the torchtitan common MoE as self._moe; its
        # experts container is the ModuleList of per-expert MLPs.
        moe = getattr(ffn, "_moe", None)
        if moe is None or not hasattr(moe, "experts"):
            raise ValueError(
                f"layer {layer.layer_idx} MoE ffn missing _moe.experts; "
                "EP plan needs the standard torchtitan MoE wrapping."
            )
        parallelize_module(
            module=moe.experts,
            device_mesh=ep_mesh,
            parallelize_plan=plan,
        )
        moe_layers_wrapped += 1
    logger.info(
        "EP plan wrapped %d MoE layer experts.", moe_layers_wrapped
    )


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

    When ``ep_degree > 1``, MoE expert parameters shard via ``edp_mesh``
    while non-expert parameters shard via ``dp_mesh`` — this matches
    the llama4 / deepseek_v3 pattern and avoids the
    "Cannot concatenate overlapping meshes" error that fires when a
    single dp_mesh is used for both expert and non-expert params under
    EP. ``edp_mesh`` is the dp_shard axis with the EP rank dimension
    factored out (``parallel_dims.get_optional_mesh("efsdp")`` or
    ``["dp_replicate", "efsdp"]``).
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

    # Under PP, ``embed_tokens`` is stripped on non-first stages and
    # ``lm_head`` (in head_tail) is stripped on non-last stages; both
    # become None on PP-stripped ranks. Filter Nones before passing to
    # fully_shard so the wrap iterates only over real modules.
    embed = getattr(model, "embed_tokens", None)
    if tied:
        # When tied, embed shares storage with lm_head — bundle them
        # so the shared weight isn't all-gathered twice. Skip the bundle
        # entirely if no embed module is present on this rank.
        bundle = [m for m in [embed, *head_tail] if m is not None]
        if bundle:
            fully_shard(
                bundle,
                **fsdp_config,
                reshard_after_forward=(reshard_after_forward_policy == "always"),
            )
    else:
        if embed is not None:
            fully_shard(
                embed,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        if head_tail:
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
    #
    # When EP > 1 and the layer is MoE, the expert ModuleList must
    # shard via ``edp_mesh`` (the dp_shard axis with the EP dimension
    # factored out). The pytorch version in this repo does NOT support
    # per-param ``shard_placement_fn`` returning per-param meshes
    # (signature is ``Callable[[nn.Parameter], Shard | None]`` only),
    # so we use the nested-fully_shard pattern: wrap the experts
    # container as its own FSDP unit on ``edp_mesh`` first, then wrap
    # the surrounding layer on ``dp_mesh``. FSDP2 treats the inner
    # unit as a sub-module and only shards the non-expert params at
    # the layer level, which keeps the meshes orthogonal and avoids
    # the "Cannot concatenate overlapping meshes" error.
    use_nested_ep = ep_degree > 1 and edp_mesh is not None
    edp_fsdp_config = None
    if use_nested_ep:
        edp_fsdp_config = {"mesh": edp_mesh, "mp_policy": mp_policy}
        if cpu_offload:
            edp_fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer in model.layers.values():
        layer_is_moe = bool(getattr(layer, "is_moe", False))
        if use_nested_ep and layer_is_moe:
            ffn = getattr(layer, "ffn", None)
            assert (
                ffn is not None
                and getattr(ffn, "_moe", None) is not None
                and hasattr(ffn._moe, "experts")
            ), (
                f"layer {getattr(layer, 'layer_idx', '?')} is_moe=True "
                "but ffn._moe.experts missing; EP-aware FSDP needs the "
                "standard KimiMoE wrapping."
            )
            # Inner unit: experts on edp_mesh.
            fully_shard(
                ffn._moe.experts,
                **edp_fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # Outer unit: the whole layer on dp_mesh. FSDP2 sees the
        # already-wrapped experts as a nested unit and only shards
        # non-expert params at this level.
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

    # block_attn_res: TP path requires DTensor.to_local on proj.weight to
    # unmix DTensor and plain Tensor in the einsum. dynamo's fake-tensor
    # mode doesn't trace through the conditional to_local cleanly (it
    # propagates DTensor type past the isinstance branch and the einsum
    # call sees mixed DTensor + plain). Easiest fix: graph-break at the
    # block_attn_res entry, the function runs eagerly. block_attn_res is
    # a single softmax + two einsums, so eager dispatch doesn't lose
    # meaningful compile gains.
    #
    # We patch in-place at every callsite's bound module — both the
    # source module (attn_res.attn_res) and importers (kimi_linear's
    # attn_res_model and the standalone attn_res.model) — because each
    # ``from .attn_res import block_attn_res`` creates an independent
    # binding that wouldn't be touched by patching the source module.
    from torchtitan.experiments.attention_residual import attn_res as _src
    from torchtitan.experiments.attention_residual import model as _llama_attn_res_mod
    from torchtitan.experiments.attention_residual.kimi_linear import attn_res_model as _kimi_attn_res_mod
    disabled = torch.compiler.disable(_src.block_attn_res, recursive=True)
    _src.block_attn_res = disabled
    _llama_attn_res_mod.block_attn_res = disabled
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
    from torchtitan.experiments.attention_residual.kimi_linear.model import KimiDeltaAttention
    KimiDeltaAttention.forward = torch.compiler.disable(
        KimiDeltaAttention.forward, recursive=True,
    )

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
