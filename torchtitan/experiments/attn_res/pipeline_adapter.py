# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-stage caching adapter + custom ``pipelining_fn`` for AttnRes.

Two pieces live here:

1. :class:`CrossStageCacheAdapter` — an ``nn.Module`` that wraps a per-stage
   AttnRes decoder. Inter-stage forward bandwidth becomes **constant in
   stage id**: each stage only receives / sends the blocks newly committed
   by the immediately preceding stage; prior-stage blocks are cached
   locally per microbatch.
2. :func:`pipeline_llm_with_cache_adapter` — a ``pipelining_fn`` plugged
   into the experiment's ``ModelSpec``. It calls core's
   :func:`torchtitan.distributed.pipeline_parallel.pipeline_llm`, then
   walks the resulting schedule's stages and wraps each ``stage.submod``
   with the adapter. This keeps all AttnRes-specific pipeline wiring
   inside ``experiments/attn_res/`` — no modifications to core torchtitan.

Activation is gated by ``TORCHTITAN_ATTNRES_CACHE=1`` in the environment
so naive (full-stack send) and cached modes are both A/B-runnable from
the same binary without rebuilding model specs. When the flag is off the
wrapper still calls core ``pipeline_llm`` and simply skips the wrapping
step, matching Phase-2 behavior exactly.

See ``adapter_design.md`` in the project root for the full state machine,
invariants, and open unknowns this draft rests on.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
)

from torchtitan.experiments.attn_res.attn_res import stack_blocks, unstack_blocks


def adapter_enabled() -> bool:
    """Env-flag gate so the adapter is opt-in until we trust it."""
    return os.environ.get("TORCHTITAN_ATTNRES_CACHE") == "1"


class _PerMicrobatchCache:
    """Maps microbatch-key -> list of cached block tensors (from earlier stages).

    Keying strategy (adapter_design.md open-unknown #1): use ``id(tensor)``
    of the ``partial_block`` the adapter received. Activation identity is
    stable across one microbatch's forward + backward. If the schedule
    ever re-enters forward for rematerialization, the key will miss and
    the adapter must fall back to recomputing from the send path — which
    it currently does NOT. Document and revisit.
    """

    def __init__(self) -> None:
        self._forward: dict[int, list[torch.Tensor]] = {}
        self._grad_accum: dict[int, list[torch.Tensor]] = defaultdict(list)

    def put_forward(self, key: int, blocks: list[torch.Tensor]) -> None:
        self._forward[key] = blocks

    def get_forward(self, key: int) -> list[torch.Tensor]:
        return self._forward.get(key, [])

    def drop(self, key: int) -> None:
        self._forward.pop(key, None)
        self._grad_accum.pop(key, None)

    def add_grad(self, key: int, block_idx: int, grad: torch.Tensor) -> None:
        slot = self._grad_accum[key]
        while len(slot) <= block_idx:
            slot.append(None)
        if slot[block_idx] is None:
            slot[block_idx] = grad.detach().clone()
        else:
            slot[block_idx] = slot[block_idx] + grad.detach()

    def pop_grads(self, key: int) -> list[torch.Tensor]:
        return self._grad_accum.pop(key, [])


class CrossStageCacheAdapter(nn.Module):
    """Wraps an ``AttnResLlama3Model`` stage; caches prior-stage blocks.

    Args:
        wrapped: the per-stage AttnRes decoder (already parallelized).
        stage_id: global stage index (counts virtual stages for VP, not
            just pp_rank).
        num_stages: total number of stages across PP + VP.

    Note: ``wrapped.forward`` is expected to accept ``_return_only_new_blocks``
    behavior when this flag is set on the adapter. This draft sets it by
    setting ``wrapped._return_only_new_blocks = True``; the wrapped model
    must honor it (currently a TODO in
    ``torchtitan/experiments/attn_res/model.py``).
    """

    def __init__(
        self,
        wrapped: nn.Module,
        *,
        stage_id: int,
        num_stages: int,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.stage_id = stage_id
        self.num_stages = num_stages
        self._cache = _PerMicrobatchCache()

        # Flip the wrapped model into "return only new blocks" mode.
        # See adapter_design.md "Required model change" for the exact
        # contract this relies on.
        if hasattr(wrapped, "_return_only_new_blocks"):
            wrapped._return_only_new_blocks = True
        else:
            # Wrapped model is an older Phase-2-only model; we fall back
            # to the naive full-stack path and log a warning.
            import warnings

            warnings.warn(
                "Wrapped model does not expose _return_only_new_blocks; "
                "adapter will run in naive (full-stack) mode with no "
                "bandwidth saving."
            )

    # --- first stage: receives (tokens,), produces (partial, new_blocks) ---
    def _forward_first_stage(
        self, tokens: torch.Tensor, *model_args, **model_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.wrapped(tokens, *model_args, blocks=None, **model_kwargs)
        # wrapped returned (partial, stacked_new_blocks) in "new only" mode
        assert isinstance(out, tuple) and len(out) == 2, (
            f"First stage expected tuple from wrapped model, got {type(out)}"
        )
        partial, new_blocks_tensor = out
        # No cache entry for stage 0 — it has no earlier blocks to cache.
        # Still key the entry so later stages' send/recv bookkeeping stays
        # symmetric across the schedule.
        self._cache.put_forward(id(partial), [])
        return partial, new_blocks_tensor

    # --- middle stage: receives (partial, new_blocks_from_prev_stage) ---
    def _forward_middle_stage(
        self,
        partial: torch.Tensor,
        new_blocks_tensor: torch.Tensor,
        *model_args,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Look up cached blocks from earlier stages, keyed on... partial id.
        # For the very first time stage s sees this microbatch, there is
        # no entry; that is expected. We synthesize an empty one so
        # subsequent lookups (AC rerun, etc.) see a cache hit.
        earlier_key = self._prev_stage_key(partial)
        earlier_blocks = self._cache.get_forward(earlier_key)

        # Hand the wrapped model the FULL blocks list (cached + new_from_prev).
        new_blocks_list = unstack_blocks(new_blocks_tensor)
        full_blocks = earlier_blocks + new_blocks_list
        full_blocks_tensor = stack_blocks(full_blocks) if full_blocks else None

        # Register backward hooks on the cached blocks to accumulate grads.
        self._register_grad_accumulators(earlier_key, earlier_blocks)

        out = self.wrapped(partial, blocks=full_blocks_tensor, *model_args, **model_kwargs)

        # Last stage or middle stage?
        if not isinstance(out, tuple):
            # last stage returned logits; nothing to cache going forward
            return out

        partial_next, new_blocks_from_this_stage = out
        # Cache the blocks committed by THIS stage under the new microbatch key.
        new_list = unstack_blocks(new_blocks_from_this_stage)
        self._cache.put_forward(id(partial_next), full_blocks + new_list)
        return partial_next, new_blocks_from_this_stage

    def forward(self, *args, **kwargs):
        # Torch pipelining unpacks tuple returns; stage 0 gets just the tokens
        # (and maybe attention_masks / positions in kwargs). Middle/last
        # stages get (partial, blocks_tensor) as positional.
        if len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].dim() == 4:
            # (partial[BxTxD], blocks[NxBxTxD], *rest) — middle or last
            return self._forward_middle_stage(args[0], args[1], *args[2:], **kwargs)
        # first stage: tokens positional
        return self._forward_first_stage(*args, **kwargs)

    # --- backward grad accumulation helpers ---
    def _register_grad_accumulators(
        self, key: int, blocks: list[torch.Tensor]
    ) -> None:
        """Attach hooks that accumulate per-block grads into the cache.

        When the wrapped model's backward runs over the stacked blocks, each
        cached block receives a grad contribution. We accumulate per-block
        so that when this stage's backward completes, we can send the
        accumulated grad back to the stage that produced the block.

        Open unknown #3 in adapter_design.md: verify these hooks fire under
        PipelineScheduleMulti. If not, replace with a custom
        ``torch.autograd.Function`` whose ``backward`` writes into the cache.
        """
        for i, b in enumerate(blocks):
            if not b.requires_grad:
                continue

            def _hook(grad: torch.Tensor, *, _key=key, _i=i) -> torch.Tensor:
                self._cache.add_grad(_key, _i, grad)
                return grad

            b.register_hook(_hook)

    def _prev_stage_key(self, partial: torch.Tensor) -> int:
        """Heuristic: key cached blocks by id of the partial tensor that
        arrived WITH them. Works iff schedule forwards the same Python tensor
        object through the microbatch's whole life.
        """
        return id(partial)

    # --- hook points the trainer can invoke at end of microbatch/step ---
    def on_microbatch_end(self, partial_key: int) -> None:
        """Called (future work) by the schedule when a microbatch's
        forward+backward are complete, so the adapter can drop its cache.
        """
        self._cache.drop(partial_key)

    def extra_repr(self) -> str:
        return f"stage_id={self.stage_id}, num_stages={self.num_stages}"


def _iter_schedule_stages(schedule: _PipelineSchedule):
    """Yield the ``PipelineStage`` objects held by a schedule.

    ``PipelineScheduleSingle`` stores one stage on ``_stage``;
    ``PipelineScheduleMulti`` holds a list on ``_stages``. Both attributes
    are private in torch today; the fallback branch catches future renames
    early with an explicit error instead of an opaque AttributeError.
    """
    if isinstance(schedule, PipelineScheduleSingle):
        yield schedule._stage
    elif isinstance(schedule, PipelineScheduleMulti):
        yield from schedule._stages
    else:
        raise RuntimeError(
            f"Unexpected pipeline schedule class {type(schedule).__name__}; "
            "cross-stage cache adapter wiring does not know how to locate "
            "its stages. Extend _iter_schedule_stages in "
            "torchtitan.experiments.attn_res.pipeline_adapter."
        )


def pipeline_llm_with_cache_adapter(model: nn.Module, **kwargs):
    """Custom ``pipelining_fn`` for AttnRes.

    Calls core :func:`torchtitan.distributed.pipeline_parallel.pipeline_llm`,
    then (when ``TORCHTITAN_ATTNRES_CACHE=1``) wraps each stage's submodule
    with :class:`CrossStageCacheAdapter`. The wrapper also flips
    ``submod._return_only_new_blocks = True`` on the wrapped model so it
    returns only this stage's committed blocks at intermediate stages.

    All kwargs are forwarded unchanged to core; the signature must stay in
    sync with ``ParallelizeFunction``/pipelining contract.

    Returns the same 4-tuple as core ``pipeline_llm``.
    """
    # Deferred import so this module stays importable without torchtitan's
    # distributed wiring (e.g. for CPU unit tests).
    from torchtitan.distributed.pipeline_parallel import pipeline_llm

    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )

    if not adapter_enabled():
        return pp_schedule, model_parts, has_first_stage, has_last_stage

    stages = list(_iter_schedule_stages(pp_schedule))
    num_stages = len(stages)
    for i, stage in enumerate(stages):
        wrapped = CrossStageCacheAdapter(
            stage.submod, stage_id=stage.stage_index, num_stages=num_stages
        )
        stage.submod = wrapped
        # Also update model_parts so downstream (optimizer build, checkpoint
        # state_dict) sees the wrapped module. torchtitan iterates
        # model_parts to bind optimizers and compilers.
        if i < len(model_parts):
            model_parts[i] = wrapped

    return pp_schedule, model_parts, has_first_stage, has_last_stage
