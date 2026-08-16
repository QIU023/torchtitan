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
* **TP** -- DSv3-style plan in ``apply_tp_kimi_k3``, matched to
  Kimi's MLA + KDA + MoE layout. All boundary tensors use_local_output
  so PP send/recv, AttnRes stacking, and fla-core triton kernels see
  plain tensors.
* **CP** -- Ulysses-style (seq-local projections, one fused all-to-all
  seq<->head, full-sequence attention on the local head subset) inside
  each KDA/MLA module. The module boundary stays a seq-sharded plain
  tensor, which is what keeps CP composable with FSDP/TP/PP/EP. Requires
  ``context_parallel_load_balancer=None`` (enforced with a ValueError).
* **EP** -- Expert Parallel for KimiMoE via ``apply_ep_kimi_k3``;
  all-to-all dispatch/combine on the EP mesh.
* **PP** -- via the cross-stage cache adapter in ``pipeline_adapter.py``;
  PP rank assignment is in torchtitan core, scheduling in the
  ``pipeline_kimi_k3_with_cache_adapter`` wrapper.
* **torch.compile** -- per-decoder-layer compile via
  ``_apply_compile_kimi_k3``; MoE for-loop and fla-core triton ops
  are carved out.
* **Activation checkpointing** -- applied via the shared
  ``ActivationCheckpointingConfig.build().apply()`` path since the Kimi
  decoder layer registry matches the llama3 ``model.layers`` iteration
  pattern. Honors all upstream modes (``selective``, ``full``,
  ``memory_budget``, ``none``).

Why KDA is ``NoParallel`` on the tp axis
----------------------------------------
The whole ``delta_attention`` is registered ``NoParallel``, so every child param
(q/k/v/o projections, conv1d weights, ``A_log``, ``dt_bias``) becomes a
``DTensor(Replicate)`` on ``tp_mesh``. Three consequences worth stating once:

* **It is required for FSDP+TP composability.** ``clip_grad_norm_`` stacks
  per-param grad norms across the parameter list, and the stack fails if some
  norms live on ``(fsdp, tp)`` and others on ``(fsdp,)`` alone.
* **The linears compose, the kernels do not.** Inside the KDA forward every
  linear receives a DTensor input from ``input_layernorm`` and DTensor weights
  from this wrap, so it produces DTensor. fla-core's triton kernels
  (``causal_conv1d`` inside ``ShortConvolution``, ``fused_kda_gate``,
  ``chunk_kda``, ``FusedRMSNormGated``) do not dispatch through DTensor, so
  ``ShortConvolution.forward`` and ``FusedRMSNormGated.forward`` are shimmed to
  ``to_local`` weight and input at the kernel boundary and ``from_local`` the
  output. ``fused_kda_gate`` and ``chunk_kda`` are called explicitly, and their
  to_local-and-call wrappers live in ``model.py``.
* **The exit is plain.** ``local_output_grad_placements=Replicate`` ``to_local``s
  the output at the module exit, matching MLA's ``o_proj`` and the dense MLP's
  ``down_proj``, so ``attn_out`` is a plain tensor everywhere and both
  partial-block accumulation and ``block_attn_res`` see one tensor kind.

Why the MoE TP plan wraps LEAVES with ``NoParallel`` rather than the container
--------------------------------------------------------------------------------
Four facts, all load-bearing, kept here rather than inline at the call site:

1. **The mechanism.** Registering each leaf with ``NoParallel`` puts its params on
   ``tp_mesh`` as ``DTensor(Replicate)``. The shared MoE forward ``to_local``s its
   DTensor input at entry, so gate/experts/shared_experts receive PLAIN x; each
   leaf's ``prepare_input`` hook converts back to DTensor, runs on DTensor
   (params are DTensor too, so the matmul composes), and returns plain via
   ``local_output_grad_placements=Replicate``.
2. **Why the experts need nothing extra.** ``GroupedExperts.forward`` and
   ``KimiMLP``'s gate/up/down projections dispatch normally on DTensor because
   their ops are standard ``F.linear``; ``GroupedExperts`` additionally
   ``to_local``s its DTensor params before the grouped_mm kernel.
3. **Why NOT the ``moe`` container.** The MoE forward strips x with ``to_local``
   AFTER the parent's ``prepare_input``. Wrapping ``moe`` or ``moe._moe`` would make
   the input a DTensor at MoE entry, ``to_local`` would make it plain, and the
   gate would then receive plain x while holding DTensor weights -- which errors.
   Wrapping the gate ITSELF keeps the boundary right.
4. **Required for FSDP+TP composability.** ``clip_grad_norm_`` stacks per-param
   grad norms across the whole parameter list, which fails if MoE params live on
   ``(fsdp,)`` while MLA params live on ``(fsdp, tp)``. ``NoParallel`` promotes
   the MoE params to ``(fsdp, tp)`` after the FSDP wrap.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_module, distribute_tensor, DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

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
from torchtitan.distributed.tensor_parallel import NoParallel
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

    Wires (in order, before FSDP wrap): TP -> CP -> EP -> AC -> compile ->
    FSDP/HSDP. AC is applied before compile so the compiled subgraph is
    the checkpointed unit (matches upstream llama3/qwen3 ordering).

    CP is Ulysses-style (seq-local projections, all-to-all seq<->head,
    full-seq attention on the local head subset) inside each KDA/MLA
    module; the module boundary stays a seq-sharded plain tensor, which
    is what keeps CP composable with FSDP/PP/EP. CP+TP composes too: the
    CP collectives run on plain local tensors AFTER the TP-wrapped
    projections (to_local at the same gap the TP plan already strips
    DTensor), so under TP each rank computes num_heads/(tp*cp) MLA heads.
    Requires context_parallel_load_balancer=None (validated below).
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
        # TP plan modeled on ``deepseek_v3/parallelize.py``.
        # Key idea: every module boundary in the forward emits a plain
        # Tensor (use_local_output=False / output_layouts=Replicate())
        # so:
        #   * the stack inside ``block_attn_res`` aggregates plain
        #     Tensors uniformly across MLA-output / KDA-output / partial
        #     blocks (no mixed-dispatch errors);
        #   * fla-core triton kernels inside KDA see plain Tensors and
        #     dispatch normally;
        #   * SDPA in ``KimiMLAInnerAttention`` runs on plain Tensors
        #     thanks to ``prepare_module_input(use_local_output=False)``.
        #
        # The TP collectives still fire — ColwiseParallel produces
        # DTensor(Shard) intermediates internally and RowwiseParallel
        # all-reduces on the way out before to_local. We just keep
        # boundary types plain so PP send/recv, AttnRes block stacking,
        # and triton kernels never see a mixed-mesh tensor.
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp_kimi_k3(
            model,
            tp_mesh,
            skip_expert_params=(parallel_dims.ep_enabled or _model_has_moe(model)),
            moe_module_parallel=_model_has_moe(model),
        )
        # Stash the TP mesh on the model so AttnRes top-level forward
        # can DTensor-ify PP-received block tensors when they arrive
        # plain (PP P2P uses raw send/recv, so mid-stage receives
        # plain tensors that need to be converted back into the TP
        # mesh's local view before aggregation).
        model._tp_mesh = tp_mesh
        # Also on the language model. The AttnRes layer loop lives there and
        # reads _tp_mesh to lift the stream at its entry; with the mesh only on
        # the multimodal wrapper the getattr returned None, the lift never ran,
        # and the plain carrier met a DTensor prefix_sum in the carrier cat.
        _lm = getattr(model, "language_model", None)
        if _lm is not None:
            _lm._tp_mesh = tp_mesh
        logger.info(
            "Applied DSv3-style TP plan tp_degree=%d.",
            parallel_dims.tp,
        )
    if parallel_dims.cp_enabled:
        # Fail loudly on configs the CP implementation cannot honor.
        # Silent degradation here has already produced plausible-but-wrong
        # runs (headtail-permuted sequences), so these are ValueErrors,
        # not warnings.
        cp_load_balancer = parallelism.context_parallel_load_balancer
        if cp_load_balancer is not None:
            raise ValueError(
                "kimi_linear CP requires context_parallel_load_balancer="
                f"None, got '{cp_load_balancer}'. The KDA/MLA CP path "
                "reassembles the full sequence as contiguous rank-ordered "
                "shards; a load balancer (e.g. headtail) permutes the "
                "sequence before sharding, which silently breaks causal "
                "order inside the attention kernels (future-token leakage). "
                "Load balancing is also unnecessary here: every rank "
                "computes the full sequence for its head subset, so "
                "per-rank work is already symmetric."
            )
        # Ulysses CP for the hybrid KDA/MLA backbone: each attention
        # module runs its projections seq-local, swaps seq<->head
        # sharding with one fused differentiable all-to-all on the cp
        # sub-mesh, and runs conv/scan/SDPA on its head subset over the
        # full sequence (see KimiDeltaAttention/KimiMLAAttention
        # ._forward_cp). chunk_kda is bit-exactly per-head independent
        # (verified bit-exact against a single-rank reference), so
        # head sharding is exact.
        # KDA can't ring (fla-core scan) and the custom MLA
        # inner_attention isn't the torchtitan SDPA type
        # apply_cp_to_forward expects -- hence this module-internal CP
        # rather than the upstream dispatcher.
        from torchtitan.models.kimi_k3.model import KimiDeltaAttention, KimiMLAAttention

        cp_group = parallel_dims.get_mesh("cp").get_group()
        cp_degree = parallel_dims.cp
        tp_degree = parallel_dims.tp
        n_attn = 0
        for m in model.modules():
            if isinstance(m, KimiMLAAttention):
                # Under TP the head axis is already tp-sharded; Ulysses
                # further splits the local heads by cp.
                if m.num_heads % (tp_degree * cp_degree) != 0:
                    raise ValueError(
                        f"MLA num_attention_heads={m.num_heads} must be "
                        f"divisible by tp*cp={tp_degree * cp_degree} for "
                        "Ulysses CP head sharding"
                    )
                m._cp_group = cp_group
                n_attn += 1
            elif isinstance(m, KimiDeltaAttention):
                # KDA is NoParallel under TP (replicated), so only cp
                # splits its heads.
                if m.num_heads % cp_degree != 0:
                    raise ValueError(
                        f"KDA kda_num_heads={m.num_heads} must be "
                        f"divisible by cp={cp_degree} for Ulysses CP "
                        "head sharding"
                    )
                m._cp_group = cp_group
                n_attn += 1
        # The multimodal wrapper needs it too. prepare_context_parallel_input
        # shards inputs/labels/positions but NOT pixel_values, so each CP rank
        # sees a slice of the vision sentinels while still being handed the
        # whole batch of images; the splice then finds a sentinel count that
        # matches neither the image count nor the token count.
        from torchtitan.models.kimi_k3.multimodal_model import KimiK3MultimodalModel

        subgroups = _build_cp_subgroups(cp_group)
        for m in model.modules():
            if isinstance(m, KimiK3MultimodalModel):
                m._cp_group = cp_group
                m._cp_subgroups = subgroups
        logger.info(
            "Applied Ulysses CP cp_degree=%d (all-to-all at %d attn layers).",
            cp_degree,
            n_attn,
        )
    if (parallel_dims.ep > 1 or parallel_dims.tp > 1) and _model_has_moe(model):
        # Expert Parallel for Kimi MoE layers. The
        # KimiMoE module wraps torchtitan.models.common.moe.MoE as
        # self._moe; the expert ModuleList is at self._moe.experts.
        # Apply standard ExpertParallel() to that experts container,
        # which fires all-to-all on the EP mesh for token dispatch +
        # combine. Cache adapter delta accumulation interacts with
        # MoE only at the block boundary (after FFN residual add),
        # so EP routing within the FFN body is transparent to the
        # AttnRes block-commit logic.
        ep_expected = apply_ep_kimi_k3(model, parallel_dims)
        logger.info(
            "Applied EP plan (per-MoE-layer ExpertParallel) ep_degree=%d.",
            parallel_dims.ep,
        )
    # Declarative sharding, after EP so the MoE subtrees are already parallelized and
    # before AC so the checkpointed unit sees the distributed parameters. Step 1 of the
    # migration to upstream's declarative path: this only ACTIVATES declarations that
    # already exist -- no plan is removed here, so the imperative plan and the
    # declarations are both in effect and their agreement is what the matrix checks.
    entered = _drive_declarative_sharding(model, parallel_dims)
    if parallel_dims.ep_enabled:
        # After the driver, because the driver is what carries the ep mesh down.
        verify_ep_applied(ep_expected)
    if parallel_dims.tp_enabled:
        _sweep_remaining_to_replicate(
            model,
            parallel_dims.get_mesh("tp"),
            skip_expert_params=(
                parallel_dims.ep_enabled or _model_has_moe(model)
            ),
        )
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


def _patch_fla_for_dtensor() -> dict:
    """Build DTensor-safe forwards for ShortConvolution + FusedRMSNormGated.

    Returns ``{class: forward_fn}`` for :func:`_bind_fla_dtensor_shims` to bind
    per instance. Nothing here mutates the fla classes.

    Both classes wrap fla-core triton kernels (``causal_conv1d``,
    ``fused_norm_gated``) that take raw tensor pointers and don't
    dispatch through DTensor. Under TP, KDA's delta_attention is
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
                        t,
                        in_mesh,
                        in_placements,
                        run_check=False,
                    )
                return t

            if isinstance(out, tuple):
                return tuple(_rewrap(o) for o in out)
            return _rewrap(out)

        return _patched

    return {
        ShortConvolution: _make_patch(ShortConvolution),
        FusedRMSNormGated: _make_patch(FusedRMSNormGated),
    }


def _bind_fla_dtensor_shims(model: nn.Module) -> int:
    """Bind the DTensor-safe forwards PER INSTANCE, not on the fla classes.

    The previous form assigned ``cls.forward`` on ShortConvolution and
    FusedRMSNormGated, which is a global, irreversible mutation of a third-party
    library: it changes behaviour for every model in the process, including ones
    that never enabled TP, and cannot be undone. torchtitan's own convention
    (qwen3_5) keeps kernel dispatch stateless and does the DTensor conversion at
    the call site.

    Binding to instances keeps the same conversion while touching only the
    modules of the model being parallelized. Idempotent: a module already bound
    is skipped.
    """
    patches = _patch_fla_for_dtensor()
    n = 0
    for m in model.modules():
        fn = patches.get(type(m))
        if fn is None or getattr(m, "_fla_dtensor_bound", False):
            continue
        m.forward = fn.__get__(m, type(m))
        m._fla_dtensor_bound = True
        n += 1
    return n


def _model_has_moe(model: nn.Module) -> bool:
    """True if any layer carries a KimiMoE (module-internal MoE
    parallelization applies)."""
    return any(bool(getattr(layer, "is_moe", False)) for layer in model.layers.values())


def _apply_tp_moonvit_mlp(vision_tower: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Tensor-parallelize the MoonViT encoder MLPs. Returns blocks covered.

    Report sec 5.2.3 asks for a genuinely parallel vision tower, not a replicated
    one. This is the half that is unambiguous: fc0 is hidden -> intermediate and
    fc1 is intermediate -> hidden with an elementwise GELU between them, so
    Colwise/Rowwise is exact and the activation shard commutes with the
    nonlinearity.

    Attention is deliberately NOT sharded here. ``wqkv`` is one fused Linear
    whose flat output is laid out ``[3, A, K]`` with the 3 outermost, so an
    even column split hands rank 0 all of q plus half of k -- which is not
    ``[3, A_local, K]``, and the ``view`` in ``_attend`` would silently
    reinterpret it. Sharding it needs either a permuted weight layout (and a
    matching permutation in the state-dict adapter, i.e. a change to the
    checkpoint contract we just finished aligning to the official
    implementation) or splitting the fused Linear into three. Left for a
    separate change rather than smuggled in here.

    Must run BEFORE ``distribute_module`` replicates the rest of the tower:
    the styles need plain-tensor parameters, and a later ``distribute_module``
    leaves already-distributed ones alone (verified, not assumed).
    """
    encoder = getattr(vision_tower, "encoder", None)
    blocks = getattr(encoder, "blocks", None) if encoder is not None else None
    if not blocks:
        return 0

    # Attention head sharding when the heads divide the ranks. Reported rather
    # than silently skipped: the debug tower has 3 heads, which nothing divides.
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    num_heads = getattr(blocks[0], "num_heads", 0)
    # vit_tp_heads=False forces replicated attention. Kept as a verification
    # affordance: head sharding changes the summation order of the attention
    # output, so the only way to attribute a numerical difference to it is an
    # A/B on one configuration.
    from torchtitan.models.kimi_k3.knobs import topology as _topology

    shard_heads = (
        _topology().vit_tp_heads
        and num_heads >= tp_size
        and num_heads % tp_size == 0
    )
    per_rank = num_heads // tp_size if shard_heads else 0
    if not shard_heads and num_heads:
        logger.warning(
            "MoonViT TP: %d attention heads do not divide %d ranks; attention "
            "stays replicated and only the MLPs are sharded",
            num_heads,
            tp_size,
        )

    plan = {}
    for i in range(len(blocks)):
        # Layouts pinned rather than defaulted. encode_images lifts the
        # tower activations into DTensors at the boundary, so the block
        # residual is a DTensor and fc1 must hand back a DTensor too --
        # use_local_output=False here fails the add on mixed Tensor/DTensor.
        plan[f"encoder.blocks.{i}.mlp.fc0"] = ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        )
        plan[f"encoder.blocks.{i}.mlp.fc1"] = RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=Replicate(),
            use_local_output=False,
        )
    if shard_heads:
        for i in range(len(blocks)):
            # wo receives [L, A_local * K], exactly a Shard(-1) of [L, A * K].
            plan[f"encoder.blocks.{i}.wo"] = RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Replicate(),
                use_local_output=False,
            )

    parallelize_module(vision_tower, tp_mesh, plan)

    if shard_heads:
        for block in blocks:
            block._tp_head_slice = (tp_rank * per_rank, (tp_rank + 1) * per_rank)

    logger.info(
        "MoonViT TP: %d encoder MLPs sharded%s",
        len(blocks),
        f", attention over {per_rank}/{num_heads} heads per rank"
        if shard_heads
        else " (attention replicated)",
    )
    return len(blocks)


def apply_tp_kimi_k3(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    skip_expert_params: bool = False,
    moe_module_parallel: bool = False,
) -> None:
    """TP plan for kimi_linear, modeled on
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
    ``use_local_output=False`` (or ``output_layouts=Replicate()`` plus
    ``use_local_output=False``), and every NoParallel call passes
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
      - ``output_res_proj`` / ``output_res_norm`` (AttnRes
        only): NoParallel — output dim 1 / RMSNorm cannot be sharded.

    * **MLA layer (KimiMLAAttention)**:
      - ``q_proj``: ColwiseParallel (out = H * q_head_dim).
      - ``kv_a_proj_with_mqa``: NoParallel (DSv3 pattern — the output
        is split into two halves of different sizes, which is not
        compatible with sharding on the concatenated last dim).
      - ``kv_a_layernorm``: NoParallel (single-half normalization).
      - ``kv_b_proj``: ColwiseParallel (out = H * (qk_nope + v_head_dim)).
      - ``inner_attention``: prepare_module_input(use_local_output=False)
        with sequence-axis placements — strips DTensor wrapping before
        SDPA's mem-efficient cutlass kernel sees q/k/v.
      - ``o_proj``: RowwiseParallel (in_features sharded, output
        all-reduced + Replicate).

    * **KDA layer (KimiDeltaAttention)**: NOT TP-wrapped. KDA's body is
      almost entirely fla-core triton kernels; sharding heads requires
      a ring-recurrence variant of ``chunk_kda`` that fla-core doesn't
      provide. Leaving KDA delta_attention unwrapped means its parameters
      remain plain Tensors (FSDP shards them on the FSDP axis,
      replicated across TP ranks — accepting compute redundancy on the
      TP axis for a clean kernel-safety story). The KDA forward
      defensively strips any incoming DTensor (see
      ``_to_local_if_dtensor`` in model.py).

    * **dense MLP**: gate_proj / up_proj ColwiseParallel; down_proj
      RowwiseParallel(output_layouts=Replicate(), use_local_output=False).

    * **MoE FFN**: NoParallel on the ``moe`` container (EP handles the
      real parallelization on a separate axis).

    * **Per-layer norms** (``input_layernorm``, ``post_attention_layernorm``):
      NoParallel.

    * **AttnRes per-layer modules** (``attention_res_proj``, ``attention_res_norm``,
      ``ffn_res_proj``, ``ffn_res_norm``): NoParallel each.

    Collectives per forward pass (summed across layers):
    - 1 all-reduce per dense-MLP layer (down_proj rowwise)
    - 1 all-reduce per MLA layer (o_proj rowwise)
    - KDA layers fire no TP collectives.
    Plus the symmetric backward all-reduces.
    """
    # Plain-output NoParallel: ``output_layout=Replicate()`` (default) plus
    # ``use_local_output=False`` produces a plain torch.Tensor at the module
    # exit. ``NoParallel._prepare_output_fn`` ends in a bare ``to_local()``,
    # so the backward placement defaults to the output layout, Replicate:
    # the incoming local gradient is taken to be the same on every tp rank
    # and already complete.
    no_par_local = NoParallel(use_local_output=False)

    # fla-core triton kernels (causal_conv1d in ShortConvolution,
    # fused_norm_gated in FusedRMSNormGated) do not dispatch through
    # DTensor: they call triton kernels directly on the data pointers
    # of x and weight. Under TP, KDA's delta_attention is NoParallel-wrapped,
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
    n_fla = _bind_fla_dtensor_shims(model)
    if n_fla:
        logger.info("Bound DTensor-safe fla forwards on %d modules.", n_fla)

    # Top-level layout: embed, output norm, lm_head.
    # Both embed and lm_head emit plain Tensors (use_local_output=False)
    # so the AttnRes top-level forward composes cleanly with the
    # block-stacking path.
    # The multimodal wrapper keeps the text model at .language_model, so the
    # top-level plan below -- embed_tokens / norm / lm_head, addressed by name --
    # would find none of them and leave the embedding un-sharded, which surfaces
    # as "aten.embedding.default got mixed torch.Tensor and DTensor". Descend to
    # the text model for the top-level names.
    #
    # MoonViT itself wants no TP at this size (no head axis worth sharding), but
    # "leave it alone" is not the same as "replicate it". Untouched, its params
    # stay plain tensors, FSDP wraps them on the dp_mesh alone, and the text
    # params -- TP'd first, then FSDP'd -- land on the 2D (dp, tp) mesh. Nothing
    # in the forward notices; clip_grad_norm_ does, and dies stacking gradient
    # norms across two different meshes. NoParallel replicates on the tp axis so
    # every parameter shares one mesh.
    # distribute_module with no partition_fn replicates the whole subtree's
    # parameters and installs NO boundary hooks. NoParallel would be the obvious
    # choice, but its output hook assumes a single DTensor and MoonViT returns a
    # LIST of per-sample feature blocks. encode_images does the two boundary
    # conversions instead, where the list is in hand.
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is not None:
        _apply_tp_moonvit_mlp(vision_tower, tp_mesh)
        distribute_module(vision_tower, tp_mesh)
        # Record the mesh rather than letting encode_images sniff the weight.
        # Once FSDP also shards the tower its params are DTensors too, so
        # "is the weight a DTensor" no longer distinguishes this replication
        # from that sharding, and lifting the input on an FSDP mesh puts a
        # DTensor up against the plain all-gathered weight inside the conv.
        model._vision_tp_mesh = tp_mesh

    model = getattr(model, "language_model", model)

    parallelize_module(
        model,
        tp_mesh,
        {
            # embed_tokens has NO entry: torchtitan's Embedding runs
            # vocab-parallel in its own forward once parallelize() sets
            # tp_group, and produces an ordinary tensor. RowwiseParallel made
            # DTensor do the split instead, whose MaskPartial cannot be
            # redistributed against the P(sum) the declared AttnRes projections
            # produce. Every upstream model relies on the module, not the style.
            "norm": no_par_local,
            # Shard(-1), not Replicate: core's cross-entropy has a
            # vocab-parallel path for exactly this placement
            # (_LossParallelCrossEntropy), and gathering back to Replicate meant
            # the loss saw a DTensor it had no branch for once the residual
            # stream stopped being unwrapped. This is also what upstream's models
            # do -- their lm_head output is vocab-parallel.
            "lm_head": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
        },
    )

    # Only the NORM stays imperative. Declaring both fails with
    # "aten.mul.Tensor got mixed" -- the norm is CALLED as a module, so a declared
    # weight meets the plain residual stream inside rms_norm, and the declarative
    # vocabulary has no output-side to_local. The proj's weight is read directly at
    # the use site in block_attn_res, which already unwraps a DTensor, so it does
    # not need the plan.
    if getattr(model, "output_res_norm", None) is not None:
        parallelize_module(model, tp_mesh, {"output_res_norm": no_par_local})

    # MLA inner_attention: the ONE place use_local_output=True survives the
    # residual-stream flip. q/k/v arrive head-sharded and SDPA has no DTensor
    # rule, so the kernel must see plain tensors -- the same reason the fla
    # kernels get _to_local_if_dtensor at their call sites. Setting it False with
    # everything else made every tp>1 cell die inside
    # F.scaled_dot_product_attention with the operands at Shard(1).
    inner_attn_plan = PrepareModuleInput(
        input_layouts=(Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=True,
    )

    # Per-layer plan. Each layer is a KimiDecoderLayer (or AttnRes
    # subclass with attention_res_proj + attention_res_norm).
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
            # KDA takes its own declaration now. It used to fail with
            # "aten.cat.default got mixed" because its output stayed a DTensor
            # while AttnRes concatenated it against a plain stream; the stream is
            # a DTensor now, so the mismatch is gone. It already strips DTensor
            # at the fla kernel call sites itself (_to_local_if_dtensor).
            pass
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
            # convert to plain via use_local_output=False; o_proj emits
            # plain to match the rest of the model's plain-boundary
            # convention.
            # Q: either the direct projection or K3's compression pair.
            # The pair registers exactly like the KV pair below -- the
            # compression stays replicated (its output is q_lora_rank, not a
            # head-sharded axis) and only the expansion is Colwise.
            if getattr(layer.attention, "q_lora_rank", None) is None:
                q_plan = {
                    "attention.q_proj": ColwiseParallel(
                        use_local_output=False,
                    ),
                }
            else:
                q_plan = {
                    "attention.q_a_proj": NoParallel(),
                    "attention.q_a_layernorm": NoParallel(),
                    "attention.q_b_proj": ColwiseParallel(
                        use_local_output=False,
                    ),
                }
            plan.update(
                {
                    **q_plan,
                    # NoParallel (no local_output_grad_placements): output
                    # stays as a DTensor(Replicate) so the downstream
                    # split into [kv_lora, qk_rope] halves and the
                    # subsequent kv_a_layernorm + cat with k_pass_expanded
                    # all run consistently in DTensor space (mirrors DSv3's
                    # ``wkv_a`` registration).
                    "attention.kv_a_proj_with_mqa": NoParallel(),
                    "attention.kv_a_layernorm": NoParallel(),
                    "attention.kv_b_proj": ColwiseParallel(
                        use_local_output=False,
                    ),
                    "attention.inner_attention": inner_attn_plan,
                    "attention.o_proj": RowwiseParallel(
                        output_layouts=Replicate(),
                        use_local_output=False,
                    ),
                }
            )
            # Gated MLA (k3faithful flavors): per-head gate projection,
            # out_features = num_heads -> shard on the head axis like
            # q_proj so the local gate matches the local attn heads in
            # both the TP-only and the CP+TP forward. Without this the
            # plain-tensor gate param meets DTensor x (mixed-op crash).
            # Both gate parameterizations shard on the head axis: the
            # per-head variant is [num_heads] and K3's full-rank variant is
            # [num_heads * v_head_dim], so Colwise keeps the local gate width
            # matched to the local attention output in both cases.
            if getattr(layer.attention, "attn_gate_proj", None) is not None:
                plan["attention.attn_gate_proj"] = ColwiseParallel(
                    use_local_output=False,
                )

        # FFN path.
        if not is_moe:
            ffn = getattr(layer, "feed_forward", None)
            if ffn is None:
                raise ValueError(f"layer {layer.layer_idx}: missing dense feed_forward")
            for name in ("gate_proj", "up_proj", "down_proj"):
                if not hasattr(ffn, name):
                    raise ValueError(
                        f"layer {layer.layer_idx} dense feed_forward missing '{name}'"
                    )
            # The dense FFN takes its declarations. Colwise/Rowwise here and
            # Shard(0)/Shard(1) there are the same split written twice, and once
            # the driver stops skipping these modules only one of them can act --
            # "already a DTensor with placements (Replicate(),), but its
            # sharding_config expects (Shard(dim=0),)".
        else:
            # MoE leaves get NoParallel, not the moe container (module docstring).
            ffn = getattr(layer, "moe", None)
            if ffn is None or not hasattr(ffn, "_moe"):
                raise ValueError(f"MoE layer {layer.layer_idx}: missing moe._moe")
            if moe_module_parallel:
                # The post-merge module-internal MoE path owns ALL MoE
                # parallelization (sharding configs declared at config
                # build; _moe.parallelize(parallel_dims) distributes
                # states + wires the dispatcher). Leave every _moe
                # submodule out of the TP plan.
                ffn = None
            if ffn is None:
                pass
            else:
                # router.gate: NoParallel boundary -- gate(plain x) becomes
                # gate(DTensor x), gate.weight is DTensor, gate forward
                # produces DTensor, exits as plain via local_output.
                plan["moe._moe.router.gate"] = no_par_local
            # Stable LatentMoE: the shared down/up pair and the latent
            # RMSNorm are full-width<->latent maps with no head axis, so they
            # are Replicate-on-tp like the router gate. Registering them keeps
            # their params on the tp mesh (clip_grad_norm_ needs one mesh) and
            # keeps the plain-tensor boundary convention -- without this the
            # promoted DTensor weights meet a plain activation inside the
            # RMSNorm (mixed-operand crash).
            # down and up take their declarations (both Replicate -- the MoE's
            # in_src_shardings requires it). The NORM keeps its plan entry: it is
            # on the MoE's output side where the value arrives plain, so a
            # declared weight would meet a plain input inside _fused_rms_norm.
            latent = getattr(layer.moe, "latent", None)
            if latent is not None and getattr(latent, "norm", None) is not None:
                plan["moe.latent.norm"] = no_par_local
            # The shared experts (which under the latent path hang off KimiMoE
            # itself, not off ffn._moe) take their declarations: Shard(0) on the
            # two up-projections, Shard(1) on the down-projection -- the ordinary
            # SwiGLU split. The NoParallel entries that were here replicated them
            # on a premise the residual-stream flip removed, that the MoE gets a
            # plain x, and the declaration probe found this as 36 of 36
            # mismatches: 12 layers x 3 projections, all "has Replicate,
            # declared Shard".
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
            shared = (
                getattr(ffn._moe, "shared_experts", None) if ffn is not None else None
            )
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
                        plan[f"moe._moe.shared_experts.{n}"] = no_par_local

        # AttnRes per-layer modules: each layer has TWO pseudo-queries
        # + TWO RMSNorms, all NoParallel.
        # The two per-layer pseudo-queries move to their declarations; the two
        # NORMS stay imperative. Measured, twice: proj.weight is read directly
        # inside block_attn_res, which already unwraps a DTensor, so a declared
        # Replicate is fine there. A norm is CALLED as a module, so a declared
        # weight meets the plain residual stream inside rms_norm and every tp>1
        # cell dies with "aten.mul.Tensor got mixed". The declarative vocabulary
        # has no output-side to_local, which is what use_local_output=False does
        # here, so the norms cannot move until the whole stream is DTensor.
        for name in ("attention_res_norm", "ffn_res_norm"):
            if hasattr(layer, name) and getattr(layer, name) is not None:
                plan[name] = no_par_local

        # LoRA-aware TP: a Colwise/Rowwise style can't target a
        # KimiLoRALinear (ColwiseParallel needs nn.Linear). Redirect the
        # style to the inner ``.base`` Linear and shard the adapters to
        # match -- Colwise (output-sharded): lora_a Replicate, lora_b
        # Shard(0); Rowwise (input-sharded): lora_a Shard(1), lora_b
        # Replicate. The small adapter matmul then composes with the base's
        # sharded output/input via DTensor dispatch in KimiLoRALinear.forward.
        from torchtitan.models.kimi_k3.lora import KimiLoRALinear

        lora_tp: list[tuple[nn.Module, bool]] = []
        packed_tp: list[tuple[KimiLoRALinear, bool]] = []
        for key in list(plan.keys()):
            style = plan[key]
            if not isinstance(style, (ColwiseParallel, RowwiseParallel)):
                continue
            try:
                target = layer.get_submodule(key)
            except AttributeError:
                continue
            if isinstance(target, KimiLoRALinear):
                del plan[key]
                is_colwise = isinstance(style, ColwiseParallel)
                if target._quantize_base == "mxfp4":
                    # Packed base has no base.weight for a Colwise/Rowwise
                    # style to target; shard the packed qdata/scale
                    # directly (row/whole-block-column sharding is exact
                    # for MX block-32) and let the module's packed-TP
                    # forward do local dequant + matmul + collective.
                    packed_tp.append((target, is_colwise))
                else:
                    plan[f"{key}.base"] = style
                lora_tp.append((target, is_colwise))

        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

        for mod, is_colwise in packed_tp:
            mod.apply_packed_mxfp4_tp(tp_mesh, colwise=is_colwise)

        for mod, is_colwise in lora_tp:
            a_pl = [Replicate()] if is_colwise else [Shard(1)]
            b_pl = [Shard(0)] if is_colwise else [Replicate()]
            mod.lora_a = nn.Parameter(
                distribute_tensor(mod.lora_a, tp_mesh, a_pl),
                requires_grad=mod.lora_a.requires_grad,
            )
            mod.lora_b = nn.Parameter(
                distribute_tensor(mod.lora_b, tp_mesh, b_pl),
                requires_grad=mod.lora_b.requires_grad,
            )

        # Any remaining LoRA adapters (e.g. NoParallel MoE shared experts,
        # which the plan wraps by name so the loop above skips them) must
        # ALSO land on the tp mesh as Replicate -- otherwise clip_grad_norm_
        # stacks per-param grad norms across (fsdp,) and (fsdp,tp) meshes and
        # fails (same rationale as the KDA NoParallel-everything note).
        for m in layer.modules():
            if not isinstance(m, KimiLoRALinear):
                continue
            # base_qdata/base_scale: packed bases NOT hit by a
            # Colwise/Rowwise style above (e.g. MoE shared experts) stay
            # tp-replicated so every param in the FSDP unit lives on the
            # same (fsdp, tp) mesh.
            for nm in ("lora_a", "lora_b", "base_qdata", "base_scale"):
                p = getattr(m, nm, None)
                if p is not None and not isinstance(p, DTensor):
                    setattr(
                        m,
                        nm,
                        nn.Parameter(
                            distribute_tensor(p, tp_mesh, [Replicate()]),
                            requires_grad=p.requires_grad,
                        ),
                    )

        # MoE experts (GroupedExperts.w1/w2/w3): distribute as
        # DTensor(Replicate) without installing module hooks. The
        # GroupedExperts.forward already to_local's its DTensor params
        # before the grouped_mm kernel; wrapping the module would cause
        # plain × plain mismatch (since the input x is plain too).
        #
        # When ``skip_expert_params=True`` (caller has EP enabled), do
        # NOT touch experts — leave them as plain Tensors so the EP
        # path (apply_ep_kimi_k3) can DTensor-shard them on
        # ``ep_mesh`` without hitting cross-mesh redistribute errors.
        # This mirrors llama4's design: TP plan touches router.gate +
        # shared_experts only; routed experts are EP/ETP territory.
        if is_moe and not skip_expert_params:
            ffn = layer.moe
            # Post-merge common MoE tree: routed experts live at
            # _moe.routed_experts.inner_experts with shape-suffixed
            # params (w1_EFD / w2_EDF / w3_EFD).
            experts = ffn._moe.routed_experts.inner_experts
            for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
                p = getattr(experts, name, None)
                if p is not None and not isinstance(p, DTensor):
                    setattr(
                        experts,
                        name,
                        nn.Parameter(
                            distribute_tensor(
                                p.data,
                                tp_mesh,
                                [Replicate()],
                            ),
                            requires_grad=p.requires_grad,
                        ),
                    )


def apply_ep_kimi_k3(model: nn.Module, parallel_dims) -> None:
    """Expert Parallel plan for kimi_linear MoE flavors.

    Calls ``_moe.parallelize(parallel_dims)`` on every MoE layer: the
    upstream common MoE distributes its GroupedExperts states over the
    "ep" mesh (per-Module sharding_config) and wires the token
    dispatcher's ep/tp meshes for all-to-all dispatch + combine.

    Layers without MoE (``layer.is_moe == False``, i.e. dense MLP at
    the first ``first_k_dense_replace`` indices) are skipped — they
    have no experts to shard.
    """
    moe_layers_wrapped = 0
    expected: list = []
    for layer in model.layers.values():
        if not bool(getattr(layer, "is_moe", False)):
            continue
        # `moe`, not `ffn`: the block's attribute was renamed when the FFN position
        # split into moe XOR feed_forward, and this call site was missed. getattr then
        # returned None for every layer and EP was silently never applied -- the log line
        # below said "wrapped 0 MoE layer experts" through many green ep cells, because a
        # model whose experts are simply not sharded still trains.
        ffn = getattr(layer, "moe", None)
        if ffn is None:
            continue
        # KimiMoE wraps the torchtitan common MoE as self._moe. Upstream
        # removed the standalone ExpertParallel style: EP is now module-
        # internal -- MoE.parallelize(parallel_dims) distributes the
        # GroupedExperts states over the "ep" mesh via each Module's
        # sharding_config and wires the token dispatcher's ep/tp meshes.
        moe = getattr(ffn, "_moe", None)
        if moe is None or not hasattr(moe, "parallelize"):
            raise ValueError(
                f"layer {layer.layer_idx} MoE ffn missing a parallelizable "
                "_moe; EP needs the standard torchtitan MoE wrapping."
            )
        # NOT parallelized here. The declarative driver reaches this MoE through the
        # layer's own Module.parallelize, which recurses into every child and passes the
        # same parallel_dims -- so EP is already wired by the time the driver returns, and
        # a second call raises "MoE has already been parallelized". Verified by
        # experiment: fixing the attribute name so this function found its layers turned a
        # working run into that error, which is what showed who the real caller was.
        #
        # What was actually broken was the reporting: this function read layer.ffn after
        # the attribute became layer.moe, found nothing, and logged "wrapped 0" through
        # many green EP cells. EP worked; the line lied about who did it.
        moe_layers_wrapped += 1
        expected.append((layer.layer_idx, moe))

    if not moe_layers_wrapped:
        raise ValueError(
            "expert parallel is enabled but no layer reporting is_moe has a `moe` "
            "attribute, so nothing would carry the ep mesh. The block layout and this "
            "plan disagree."
        )
    logger.info(
        "EP: %d MoE layer(s) to be wired by the declarative driver.", moe_layers_wrapped
    )
    return expected


def verify_ep_applied(expected) -> None:
    """Assert the routed experts actually landed on the ep mesh.

    Called after the declarative driver, because that is what wires them. Without this
    the only signal was a log line, and a log line is what hid the attribute-name bug for
    as long as it did: EP not being applied looks exactly like EP being applied, from the
    loss.
    """
    for layer_idx, moe in expected:
        experts = getattr(getattr(moe, "routed_experts", None), "inner_experts", None)
        if experts is None:
            raise ValueError(f"layer {layer_idx}: MoE has no routed_experts.inner_experts")
        sharded = [
            n
            for n, prm in experts.named_parameters(recurse=False)
            if isinstance(prm, _DTensor)
            and any(not pl.is_replicate() for pl in prm.placements)
        ]
        if not sharded:
            raise ValueError(
                f"layer {layer_idx}: expert parallel is enabled but no routed-expert "
                f"parameter is sharded -- "
                f"{[n for n, _ in experts.named_parameters(recurse=False)]} are all "
                f"replicated or plain. The driver did not carry the ep mesh here."
            )


def _sweep_remaining_to_replicate(
    model: nn.Module, tp_mesh: DeviceMesh, skip_expert_params: bool = False
) -> None:
    """Promote whatever nothing else distributed to DTensor(Replicate) on tp_mesh.

    Called AFTER the declarative driver, not from inside apply_tp. Running it
    first meant it claimed declared-but-not-yet-distributed parameters, and the
    driver then found them distributed with the wrong placement and refused --
    every shared_experts projection reported 'has Replicate, declared Shard'.
    Its purpose is the leftovers, and it can only tell what is left over once
    everything with an opinion has spoken.
    """
    # Final sweep: any remaining plain Tensor parameters (typically
    # ``A_log``, ``dt_bias`` on KDA layers' delta_attention that NoParallel
    # didn't catch because they're bare ``nn.Parameter``s on the
    # ``delta_attention`` module rather than children) — promote them to
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
            ffn = getattr(layer, "moe", None)
            if ffn is None or getattr(ffn, "_moe", None) is None:
                continue
            # Exclude the ENTIRE _moe subtree: the module-internal MoE
            # path (sharding configs + _moe.parallelize) owns every param
            # under it (gate, shared experts, routed experts), and runs
            # AFTER this sweep -- a Replicate promotion here would
            # conflict with the declared shardings.
            for p in ffn._moe.parameters():
                expert_param_ids.add(id(p))
    # Skip modules that DECLARE their own placement. The sweep runs inside
    # apply_tp, which is before _drive_declarative_sharding, so without this it
    # promotes every declared-but-not-yet-distributed parameter first -- and the
    # driver then finds the whole tree already distributed and enters nothing.
    # That is why declarations kept looking inert: the sweep, not the imperative
    # plan, was doing their work.
    for module in model.modules():
        cfg = getattr(module, "_sharding_config", None)
        declared = set(cfg.state_shardings or ()) if cfg is not None else set()
        for name, p in list(module._parameters.items()):
            # Skip a declared parameter only once the declaration has ACTUALLY
            # been applied. Three mechanisms can each believe another handled it:
            # the imperative plan distributes a module's children, so
            # _already_distributed reports that subtree done and the driver skips
            # the PARENT -- whose own declared parameters then never get
            # distributed, while this sweep skips them for being declared.
            # Measured exactly that way: 18 plain gradients at clip time, the
            # A_log and dt_bias of the nine KDA layers, and clip_grad_norm_ died
            # with "aten._foreach_mul_.Tensor got mixed".
            if name in declared and isinstance(p, DTensor):
                continue
            if (
                p is not None
                and not isinstance(p, DTensor)
                and id(p) not in expert_param_ids
            ):
                module._parameters[name] = nn.Parameter(
                    distribute_tensor(p.data, tp_mesh, [Replicate()]),
                    requires_grad=p.requires_grad,
                )



def _drive_declarative_sharding(model: nn.Module, parallel_dims: ParallelDims) -> int:
    """Start upstream's declarative sharding from a plain-``nn.Module`` root.

    ``Module.parallelize`` recurses through its own children and looks THROUGH
    non-``Module`` containers, but something has to call it. Our containers
    (``KimiDecoderLayer``, ``KimiK3Model``, ``KimiMoE``) are plain ``nn.Module``, so
    nothing ever did -- which left the 64 modules that already carry a
    ``sharding_config`` declaring into the void. Measured with a probe: after this
    driver they hold DTensors with exactly the declared placements
    (``gate_proj`` Shard(0), ``down_proj`` Shard(1), ``q_a_proj`` Replicate).

    Already-parallelized subtrees are SKIPPED rather than re-entered:
    ``Module.parallelize`` raises on a second call, and ``apply_ep_kimi_k3`` calls it on
    each MoE itself. Skipping the whole subtree is correct because that call already
    recursed into it.

    Returns the class names entered, so a small count can be READ rather than guessed --
    with TP on the imperative plan covers most modules and only a handful remain.
    """
    from torch.distributed.tensor import DTensor as _DTensor

    from torchtitan.protocols.module import Module as _TTModule

    def _already_distributed(m: nn.Module) -> bool:
        """Has the imperative plan (or an earlier pass) already distributed this subtree?

        ``_distribute_states`` raises "already a DTensor with placements ..." on a second
        distribution of the same weight, and during the migration BOTH mechanisms are
        live: ``apply_tp_kimi_k3`` covers some of the same modules the declarations do.
        Skipping what is already distributed makes this driver activate exactly the
        declarations the imperative plan does NOT cover, so imperative pieces can be
        deleted one at a time and the declarations take over as they go.
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
        if isinstance(child, _TTModule) and not getattr(child, "_parallelized", False):
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
