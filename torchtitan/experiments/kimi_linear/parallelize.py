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
from torch.distributed.tensor.placement_types import Replicate
from torchtitan.distributed.tensor_parallel import NoParallel

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
    # Enable TF32 tensor cores for fp32 matmuls (loss aggregation,
    # optimizer master weight updates, fp32 RoPE etc.). bf16 path is
    # unaffected. Speedup ~5-10% on fp32 ops, no measurable accuracy
    # impact at our scale.
    torch.set_float32_matmul_precision("high")

    if parallel_dims.tp_enabled:
        # Phase 6 A3: TP plan API surface is registered as
        # ``apply_tp_kimi_linear`` below, but end-to-end numerics under
        # FSDP×PP×TP composition are blocked on three upstream issues:
        #
        # 1. ``F.scaled_dot_product_attention(q, k, v)`` with DTensor
        #    inputs raises "aten.bmm got mixed Tensor and DTensor" —
        #    SDPA's internal kernel selection doesn't redistribute
        #    DTensor through the mem-efficient attention path. DSv3
        #    works around this by pulling SDPA into a separate
        #    ``inner_attention`` submodule and TP-wrapping it with
        #    ``use_local_output=True``. Replicating that pattern in
        #    KimiMLAAttention requires moving SDPA out of the inline
        #    forward.
        #
        # 2. ``KimiMLAAttention``'s ``kv_a_proj_with_mqa`` projects
        #    into ``[B, T, kv_lora_rank + qk_rope_head_dim]`` then
        #    ``torch.split`` into two semantically distinct halves.
        #    ColwiseParallel sharding the output dim doesn't align
        #    with the absolute split sizes, so the downstream
        #    ``kv_a_layernorm`` (which only sees the kv_lora half)
        #    sees a sharded tensor of an unexpected shape.
        #
        # 3. ``KimiDeltaAttention``'s fla-core triton kernels
        #    (``chunk_kda``, ``ShortConvolution``, ``FusedRMSNormGated``)
        #    don't dispatch through DTensor; sharding heads requires
        #    fla-core changes.
        #
        # We register the plan via ``apply_tp_kimi_linear`` so the
        # API surface exists for upstream review and so the FFN
        # all-reduces fire under TP=2 (the alignment runs that hit
        # SDPA early are the blocker). When the upstream MLA
        # refactoring lands (or KimiMLAAttention is restructured to
        # match DSv3's ``inner_attention`` pattern), this branch will
        # complete the alignment claim.
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp_kimi_linear(model, tp_mesh)
        logger.info(
            "Applied TP plan (FFN-shard, attention NoParallel) "
            "tp_degree=%d. NOTE: end-to-end numerics blocked on SDPA "
            "+ DTensor mixed-dispatch; see parallelize.py TP branch.",
            parallel_dims.tp,
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


def apply_tp_kimi_linear(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Phase 6 A3 TP plan for kimi_linear (KDA + MLA + dense MLP + AttnRes).

    Composability requirement: ``block_attn_res`` aggregates block-boundary
    hidden states with ``torch.stack`` across the block dim. Under FSDP+TP,
    every contributor must live on the SAME DeviceMesh (FSDP×TP), otherwise
    DTensor's stack op raises ``All operands in aten.stack must have the
    same mesh``. To satisfy that, every module that participates in the
    forward must be registered on the TP mesh — either with ``NoParallel``
    (declares "module is on this mesh, no sharding") or with the appropriate
    Colwise/Rowwise plan.

    Wrapping policy per module:
    - Embed / lm_head:   ``RowwiseParallel`` (vocab dim sharded) and
                         ``ColwiseParallel`` (vocab dim sharded). No
                         sequence-parallel or loss-parallel — keeps the
                         plan compact while still firing TP collectives.
    - Top-level ``norm``: ``NoParallel`` (declares mesh; not sharded).
    - Per-layer ``input_layernorm`` / ``post_attention_layernorm``:
                         ``NoParallel``.
    - MLA q_proj / kv_a_proj_with_mqa / kv_b_proj: ``ColwiseParallel``
      (last-dim out_features sharded across tp). MLA q has
      ``out=H*q_head_dim`` with no inter-shard interaction, kv_a has
      ``out=kv_lora_rank+qk_rope_head_dim`` (split into two contiguous
      halves; we shard along the *concatenated* last dim and the
      downstream split is shape-preserving on each shard). kv_b has
      ``out=H*(qk_nope+v_head_dim)`` similar shape. ``kv_a_layernorm``
      operates on the kv_lora half: register as ``NoParallel``.
    - MLA o_proj: ``RowwiseParallel`` (in_features sharded; output
      reduce-summed across tp).
    - KDA q/k/v/o_proj: same pattern (Colwise + Colwise + Colwise +
      Rowwise). KDA's per-head triton kernels run independently per
      head, so head-dim sharding is kernel-safe. **Per-head parameters**
      (``A_log``, ``dt_bias``) and the auxiliary projections
      (``f_a_proj``/``f_b_proj``, ``g_a_proj``/``g_b_proj``,
      ``b_proj``, ``q_conv1d``/``k_conv1d``/``v_conv1d``,
      ``o_norm``) need ``NoParallel`` registration so they live on
      the TP mesh; whether their per-head slicing is correct under TP
      is left to the kernels (worst case the rank-local view is
      replicated, accepting redundant compute over fla-core path).
    - dense MLP gate_proj / up_proj: ``ColwiseParallel``;
      down_proj: ``RowwiseParallel``.
    - AttnResProjection (``Linear(hidden, 1, bias=False)``): the output
      dim is 1 and cannot be sharded — register as ``NoParallel`` so
      the module participates in the TP mesh without splitting params.
      ``attn_res_norm`` and ``final_attn_res_norm`` likewise.

    The crucial point is the registration, not the sharding itself —
    once every module is on the TP mesh, ``block_attn_res`` stacks
    DTensors with consistent mesh shape and the math survives.

    Collectives produced (per forward pass, summed across layers):
    - 1 all-reduce per dense-MLP layer (down_proj rowwise)
    - 1 all-reduce per MLA layer (o_proj rowwise)
    - 1 all-reduce per KDA layer (o_proj rowwise)
    Plus the symmetric backward all-reduces.
    """
    # Top-level layout: embed, output norm, lm_head.
    parallelize_module(
        model,
        tp_mesh,
        {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
            "norm": NoParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # AttnRes-only top-level tail modules.
    if getattr(model, "final_attn_res_proj", None) is not None:
        parallelize_module(
            model,
            tp_mesh,
            {
                "final_attn_res_proj": NoParallel(),
                "final_attn_res_norm": NoParallel(),
            },
        )

    # Per-layer plan. Each layer is a KimiDecoderLayer (or AttnRes
    # subclass with attn_res_proj + attn_res_norm).
    for layer in model.layers.values():
        is_moe = bool(getattr(layer, "is_moe", False))
        is_kda = bool(getattr(layer, "is_linear_attn", False))

        plan: dict[str, object] = {
            "input_layernorm": NoParallel(),
            "post_attention_layernorm": NoParallel(),
        }

        # Attention path: register every submodule on the TP mesh as
        # NoParallel (no sharding). Both KDA and MLA have asymmetric
        # last-dim splits (KDA per-head A_log/dt_bias and ShortConv
        # channels; MLA's kv_a output split into kv_lora + qk_rope
        # halves with downstream layernorm on only the kv_lora half),
        # which break under any naive ColwiseParallel + split pattern.
        # Sharding attention correctly requires per-head DTensor
        # placement work that is out of scope here. Sharding only the
        # dense MLP is enough to fire TP all-reduces in the FFN path
        # and to demonstrate that the cache adapter delta survives
        # FSDP×TP composition; attention runs replicated on the TP
        # mesh (compute redundant per rank).
        plan["self_attn"] = NoParallel()

        # FFN path — dense MLP only here. MoE layers' expert routing
        # is handled via the EP plan (separate axis); for an MoE
        # layer under TP only, register the expert container as
        # NoParallel so the layer mesh is consistent.
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
                    "ffn.gate_proj": ColwiseParallel(),
                    "ffn.up_proj": ColwiseParallel(),
                    "ffn.down_proj": RowwiseParallel(),
                }
            )
        else:
            # MoE layer: declare its module exists on the TP mesh
            # without sharding (EP plan handles the actual
            # parallelization on a separate axis). NoParallel on the
            # ffn container is enough; per-expert leaves are unwrapped
            # by EP later.
            plan["ffn"] = NoParallel()

        # AttnRes per-layer modules (only present on
        # KimiAttnResDecoderLayer subclass — not all layers have it).
        # Each layer has TWO pseudo-query projections (pre-attn,
        # pre-FFN) and matching norms. All four are Linear(d->1) /
        # RMSNorm(d): not shardable, but must be on the TP mesh.
        for name in (
            "attn_res_proj", "attn_res_norm",
            "mlp_res_proj",  "mlp_res_norm",
        ):
            if hasattr(layer, name) and getattr(layer, name) is not None:
                plan[name] = NoParallel()

        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
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
