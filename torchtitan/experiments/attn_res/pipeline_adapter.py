# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-stage caching adapter + custom ``pipelining_fn`` for AttnRes.

Two pieces live here:

1. :class:`CrossStageCacheAdapter` -- an ``nn.Module`` that wraps a per-stage
   AttnRes decoder. Inter-stage forward bandwidth becomes **constant in
   stage id**: each stage only receives / sends the blocks newly committed
   by the immediately preceding stage; prior-stage blocks are cached
   locally per microbatch and their gradients are sent back to the
   original producer stage out-of-band during backward.
2. :func:`pipeline_llm_with_cache_adapter` -- a ``pipelining_fn`` plugged
   into the experiment's ``ModelSpec``. It calls core's
   :func:`torchtitan.distributed.pipeline_parallel.pipeline_llm`, then
   walks the resulting schedule's stages and wraps each ``stage.submod``
   with the adapter. This keeps all AttnRes-specific pipeline wiring
   inside ``experiments/attn_res/`` -- no modifications to core torchtitan.

Activation is gated by ``TORCHTITAN_ATTNRES_CACHE=1`` in the environment
so naive (full-stack send) and cached modes are both A/B-runnable from
the same binary without rebuilding model specs. When the flag is off the
wrapper still calls core ``pipeline_llm`` and simply skips the wrapping
step, matching Phase-2 behavior exactly.

Microbatch keying
-----------------
The adapter keys its per-microbatch cache by the integer microbatch
index supplied by torch's schedule (``fwd_chunk_id`` / ``bwd_chunk_id``
in :meth:`PipelineStage.forward_one_chunk` and
:meth:`backward_one_chunk`). We monkey-patch those two methods on each
wrapped stage to stash the index on a thread-local keyed per-adapter
right before the submod is invoked. Because forward and backward run
synchronously on the same thread that called the patched stage method,
the thread-local is live for the whole duration of the inner forward
and inner backward respectively -- so autograd hooks that accumulate
per-block grads into the cache can read the mb index even though they
only fire during backward.

The key is stable across P2P crossings because each stage computes it
from the schedule-owned chunk id, not from ``id()`` of a tensor (which
would change when NCCL allocated a fresh recv buffer on the consumer).

Backward grad send-back
-----------------------
Cached-prefix blocks received at a middle/last stage carry autograd
connections back into the wrapped model's local graph, but the *other*
end of those connections -- the producer stage that originally emitted
the block on its own forward -- lives in a different process. The
consumer therefore has to ship the accumulated grad back over P2P.

We do this with :class:`_SendBlockGradsBack`, a ``torch.autograd.Function``
whose forward is a structural pass-through (it repackages the cached
prefix into the consumer's autograd graph) and whose backward does a
batched ``dist.isend`` of the per-block grads it received back to the
producer stage over the pipeline process group.

The producer side symmetrically wraps the blocks it commits in
:class:`_RecvBlockGradsFromConsumers`, whose backward issues one
``dist.irecv`` per consumer stage for each committed block, sums them,
and adds them into the producer's local grad.

See ``adapter_design.md`` in the project root for the full state
machine, invariants, and the resolved open unknowns this draft rests on.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

from torchtitan.experiments.attn_res.attn_res import stack_blocks, unstack_blocks


def adapter_enabled() -> bool:
    """Env-flag gate so the adapter is opt-in until we trust it."""
    return os.environ.get("TORCHTITAN_ATTNRES_CACHE") == "1"


# --------------------------------------------------------------------------- #
# Per-microbatch cache
# --------------------------------------------------------------------------- #


class _PerMicrobatchCache:
    """Maps ``(stage_id, mb_index)`` -> cached prior-stage blocks and the
    per-block grads accumulated from the local backward.

    The key is the integer microbatch index owned by torch's pipeline
    schedule; see the module docstring for how it gets routed into the
    adapter. Unlike the previous ``id(partial_block)`` strategy, the
    integer survives P2P crossings: producer and consumer both look up
    by the same chunk id that the schedule issued.

    Grad semantics: ``add_grad`` is called from a backward tensor hook,
    one call per cached block per microbatch. ``pop_grads`` retrieves
    them (as a list indexed by block position) so the adapter can ship
    them back to the producer stage.
    """

    def __init__(self) -> None:
        self._forward: dict[int, list[torch.Tensor]] = {}
        # Per-cached-block metadata: (producer_rank, producer_stage_id,
        # block_index_within_producer). We need all three to tag the
        # P2P send-back correctly: (producer_stage, block_index) chooses
        # the tag; producer_rank chooses the dst rank (usually but not
        # always == producer_stage under VP).
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        self._grad_accum: dict[int, list[torch.Tensor | None]] = defaultdict(list)

    def put_forward(
        self,
        mb_index: int,
        blocks: list[torch.Tensor],
        producer_meta: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self._forward[mb_index] = blocks
        if producer_meta is not None:
            self._producer_meta[mb_index] = producer_meta

    def get_forward(self, mb_index: int) -> list[torch.Tensor]:
        return self._forward.get(mb_index, [])

    def get_producer_meta(self, mb_index: int) -> list[tuple[int, int, int]]:
        return self._producer_meta.get(mb_index, [])

    def drop(self, mb_index: int) -> None:
        self._forward.pop(mb_index, None)
        self._producer_meta.pop(mb_index, None)
        self._grad_accum.pop(mb_index, None)

    def add_grad(self, mb_index: int, block_idx: int, grad: torch.Tensor) -> None:
        slot = self._grad_accum[mb_index]
        while len(slot) <= block_idx:
            slot.append(None)
        contrib = grad.detach()
        if slot[block_idx] is None:
            slot[block_idx] = contrib.clone()
        else:
            slot[block_idx] = slot[block_idx] + contrib

    def pop_grads(self, mb_index: int) -> list[torch.Tensor | None]:
        return self._grad_accum.pop(mb_index, [])


# --------------------------------------------------------------------------- #
# Autograd plumbing for cross-stage grad send-back
# --------------------------------------------------------------------------- #


class _SendBlockGradsBack(torch.autograd.Function):
    """Consumer-side identity in forward, P2P-send-per-block in backward.

    Wraps the *cached prefix* of blocks the consumer stage hands to its
    wrapped model. Forward is a pure pass-through (the blocks flow
    straight into the local autograd graph). Backward intercepts the
    grads on those blocks and sends each one back to the rank that
    originally produced it.

    Why we need this over a plain ``register_hook``: hooks run during the
    consumer's local backward, but there's no autograd node that would
    naturally emit P2P sends. We need a custom Function whose
    ``backward`` is invoked on the grads of its *inputs*, so we can
    isend them before returning. Returning ``None`` for each input means
    the local graph stops here -- the only way grads reach the producer
    is over the P2P we just posted.

    Per-block grad accumulation into ``cache._grad_accum[mb_index]``
    also happens here -- it's the single source of truth for "what grad
    would this stage ship back". Tests can assert on it without needing
    a real NCCL send; production runs use it as the payload for the
    ``dist.isend``.
    """

    @staticmethod
    def forward(ctx, producer_ranks, tags, group, cache, mb_index, *blocks):
        assert len(producer_ranks) == len(blocks) == len(tags), (
            "producer_ranks / tags / blocks length mismatch"
        )
        ctx.producer_ranks = tuple(producer_ranks)
        ctx.tags = tuple(tags)
        ctx.group = group
        ctx.cache = cache
        ctx.mb_index = mb_index
        ctx.num_blocks = len(blocks)
        if blocks:
            ctx.dtype = blocks[0].dtype
            ctx.device = blocks[0].device
            ctx.shape = blocks[0].shape
        # Return detached clones in a tuple. We clone so the local
        # autograd graph sees this Function as the *source* of the
        # cached blocks -- otherwise autograd might route grads through
        # the pre-clone tensor and skip our backward altogether.
        outs = tuple(b.clone() for b in blocks)
        return outs

    @staticmethod
    def backward(ctx, *grad_outputs):
        group = ctx.group

        # Record each block's accumulated grad in the adapter's cache so
        # unit tests and telemetry can inspect it even when no PG is
        # wired. This is also the payload we ship over P2P below.
        recorded_grads: list[torch.Tensor] = []
        for i, g in enumerate(grad_outputs):
            if g is None:
                # Autograd can emit None for a block that wasn't used
                # (e.g. unused intermediate). The producer is still
                # expecting an irecv, so we materialize a zero grad.
                g = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
            recorded_grads.append(g)
            if ctx.cache is not None:
                ctx.cache.add_grad(ctx.mb_index, i, g)

        if group is not None and dist.is_available() and dist.is_initialized():
            reqs = []
            for i, g in enumerate(recorded_grads):
                dst = ctx.producer_ranks[i]
                tag = ctx.tags[i]
                req = dist.isend(g.contiguous(), dst=dst, group=group, tag=tag)
                reqs.append(req)
            for r in reqs:
                r.wait()
        # None for each non-tensor arg (producer_ranks, tags, group,
        # cache, mb_index) and None per block input -- grads leave via P2P.
        return (None, None, None, None, None) + tuple(None for _ in grad_outputs)


class _RecvBlockGradsFromConsumers(torch.autograd.Function):
    """Producer-side identity in forward, sum-of-irecvs in backward.

    Wraps the blocks a producer stage emits onto the forward send path.
    Forward is identity. Backward receives one grad tensor per consumer
    stage (stage_id > producer_stage_id) per block, sums them, and adds
    them into the grad that the local autograd graph would otherwise
    have produced.

    In the current AttnRes model, EVERY later stage reads EVERY earlier
    block through its attention over blocks, so every consumer sends
    back a grad contribution. The number of consumer stages per producer
    is ``num_stages - producer_stage_id - 1``.
    """

    @staticmethod
    def forward(ctx, consumer_ranks, group, tag_base, blocks_tensor):
        ctx.consumer_ranks = tuple(consumer_ranks)
        ctx.group = group
        ctx.tag_base = tag_base
        ctx.shape = blocks_tensor.shape
        ctx.dtype = blocks_tensor.dtype
        ctx.device = blocks_tensor.device
        return blocks_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        if group is None or not (dist.is_available() and dist.is_initialized()):
            # Unit-test path: no cross-stage grad to add.
            return None, None, None, grad_output
        # One incoming grad per consumer, per block.
        N = ctx.shape[0]  # number of blocks this stage committed
        per_block_shape = ctx.shape[1:]
        extra = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        pending = []
        for c_idx, src in enumerate(ctx.consumer_ranks):
            for b in range(N):
                buf = torch.empty(
                    per_block_shape, dtype=ctx.dtype, device=ctx.device
                )
                tag = ctx.tag_base + b
                req = dist.irecv(buf, src=src, group=group, tag=tag)
                pending.append((req, buf, b))
        for req, buf, b in pending:
            req.wait()
            extra[b] = extra[b] + buf
        return None, None, None, grad_output + extra


# --------------------------------------------------------------------------- #
# Microbatch-index threading
# --------------------------------------------------------------------------- #


# One thread-local shared across all adapters: each adapter stashes the
# *current* mb index under its own object id. Autograd hooks registered
# inside the adapter read from the same thread-local. Forward and
# backward execution of a single microbatch run synchronously on the
# same thread (torch's schedule drives them serially), so a thread-local
# suffices -- no cross-thread handoff needed.
_mb_state = threading.local()


def _current_mb_index(adapter_key: int) -> int | None:
    d = getattr(_mb_state, "indices", None)
    if not d:
        return None
    return d.get(adapter_key)


def _set_mb_index(adapter_key: int, mb_index: int | None) -> None:
    d = getattr(_mb_state, "indices", None)
    if d is None:
        d = {}
        _mb_state.indices = d
    if mb_index is None:
        d.pop(adapter_key, None)
    else:
        d[adapter_key] = mb_index


# --------------------------------------------------------------------------- #
# The adapter module
# --------------------------------------------------------------------------- #


class CrossStageCacheAdapter(nn.Module):
    """Wraps an ``AttnResLlama3Model`` stage; caches prior-stage blocks.

    Args:
        wrapped: the per-stage AttnRes decoder (already parallelized).
        stage_id: global stage index (counts virtual stages for VP, not
            just pp_rank).
        num_stages: total number of stages across PP + VP.
        group: pipeline process group (``stage.group``); used for P2P
            grad send-back. ``None`` falls back to a degenerate local-only
            mode (useful for CPU unit tests).
        stage_to_rank: optional ``stage_id -> rank-in-group`` map. When
            ``None`` we assume the trivial identity mapping (stage i
            lives on rank i of ``group``), which is what torchtitan's
            default PP layout produces. Interleaved VP schedules override
            this; the adapter's wrapper in
            :func:`pipeline_llm_with_cache_adapter` resolves it from the
            schedule at setup time.

    Note: ``wrapped.forward`` is expected to accept
    ``_return_only_new_blocks`` behavior when this flag is set on the
    adapter. This draft sets it by setting
    ``wrapped._return_only_new_blocks = True``; the wrapped model honors
    it; see ``model.py``.
    """

    def __init__(
        self,
        wrapped: nn.Module,
        *,
        stage_id: int,
        num_stages: int,
        group: "dist.ProcessGroup | None" = None,
        stage_to_rank: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.stage_id = stage_id
        self.num_stages = num_stages
        self._group = group
        self._stage_to_rank = stage_to_rank or {i: i for i in range(num_stages)}
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

        # Count of blocks committed by each stage; we learn it lazily as
        # forward runs and need it so the producer side of the
        # grad-send-back autograd.Function knows how many irecvs to post.
        self._committed_block_count: dict[int, int] = {}

    # ----- microbatch keying helpers ---------------------------------- #

    def _adapter_key(self) -> int:
        return id(self)

    def _current_mb(self) -> int:
        mb = _current_mb_index(self._adapter_key())
        assert mb is not None, (
            "CrossStageCacheAdapter.forward called without a microbatch "
            "index in the thread-local. This happens when the adapter's "
            "stage.forward_one_chunk monkey-patch was not installed; see "
            "pipeline_llm_with_cache_adapter."
        )
        return mb

    # ----- forward ---------------------------------------------------- #

    def _forward_first_stage(
        self, tokens: torch.Tensor, *model_args, **model_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.wrapped(tokens, *model_args, blocks=None, **model_kwargs)
        assert (
            isinstance(out, tuple) and len(out) == 2
        ), f"First stage expected tuple from wrapped model, got {type(out)}"
        partial, new_blocks_tensor = out

        mb = self._current_mb()
        # No cache entry for stage 0 -- it has no earlier blocks to cache.
        self._cache.put_forward(mb, [], producer_meta=[])

        # Wrap the emitted blocks so their local grad gets augmented by
        # the sum of grads sent back from each consumer stage.
        wrapped_blocks = self._wrap_producer_blocks(new_blocks_tensor)
        return partial, wrapped_blocks

    def _forward_middle_stage(
        self,
        partial: torch.Tensor,
        new_blocks_tensor: torch.Tensor,
        *model_args,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        mb = self._current_mb()

        # 1) Pull cached earlier-stage blocks (from this stage's cache,
        #    keyed by the schedule-issued mb index). Metadata carries
        #    per-block (producer_rank, producer_stage_id,
        #    index_within_producer) so the grad-send-back can tag each
        #    P2P op unambiguously.
        earlier_blocks = self._cache.get_forward(mb)
        earlier_meta = self._cache.get_producer_meta(mb)
        assert len(earlier_blocks) == len(earlier_meta), (
            "cache inconsistency: blocks and producer_meta length mismatch"
        )

        # 2) Parse the new blocks from the immediately previous stage.
        #    new_blocks_tensor may have shape [0, B, T, D] when the prior
        #    stage committed no blocks; unstack_blocks returns [] then.
        new_blocks_list = unstack_blocks(new_blocks_tensor)
        prev_stage_id = self.stage_id - 1
        prev_rank = self._stage_to_rank.get(prev_stage_id, prev_stage_id)
        new_meta = [
            (prev_rank, prev_stage_id, idx) for idx in range(len(new_blocks_list))
        ]

        # 3) Route the earlier-cached blocks through _SendBlockGradsBack
        #    so their backward accumulates into _grad_accum and, if a
        #    real PG is wired, posts a dist.isend per block back to
        #    the original producer.
        wrapped_earlier = self._wrap_cached_prefix_for_send_back(
            earlier_blocks, earlier_meta, mb
        )

        # 4) Concatenate wrapped cached + fresh-from-prev blocks and
        #    hand them to the wrapped model.
        full_blocks = list(wrapped_earlier) + new_blocks_list
        full_blocks_tensor = (
            stack_blocks(full_blocks) if full_blocks else None
        )

        out = self.wrapped(
            partial,
            blocks=full_blocks_tensor,
            *model_args,
            **model_kwargs,
        )

        # Last stage? then out is logits.
        if not isinstance(out, tuple):
            return out

        partial_next, new_blocks_from_this_stage = out

        # 5) Cache (cached prefix + newly-received prev blocks + this
        #    stage's commits) under the SAME mb index, so the next stage
        #    will see them.
        new_list = unstack_blocks(new_blocks_from_this_stage)
        my_rank = self._stage_to_rank.get(self.stage_id, self.stage_id)
        own_meta = [(my_rank, self.stage_id, idx) for idx in range(len(new_list))]
        self._cache.put_forward(
            mb,
            earlier_blocks + new_blocks_list + new_list,
            producer_meta=earlier_meta + new_meta + own_meta,
        )

        # 6) Wrap this stage's commits on the way out so grad sent from
        #    future consumers is added to the producer-side local grad.
        wrapped_new = self._wrap_producer_blocks(new_blocks_from_this_stage)
        return partial_next, wrapped_new

    def forward(self, *args, **kwargs):
        # Torch pipelining unpacks tuple returns; stage 0 gets just the
        # tokens (and maybe attention_masks / positions in kwargs).
        # Middle/last stages get (partial, blocks_tensor) as positional.
        if (
            len(args) >= 2
            and isinstance(args[1], torch.Tensor)
            and args[1].dim() == 4
        ):
            return self._forward_middle_stage(args[0], args[1], *args[2:], **kwargs)
        return self._forward_first_stage(*args, **kwargs)

    # ----- grad send-back wrapping ------------------------------------ #

    def _wrap_cached_prefix_for_send_back(
        self,
        blocks: list[torch.Tensor],
        meta: list[tuple[int, int, int]],
        mb_index: int,
    ) -> list[torch.Tensor]:
        """Route each cached block through :class:`_SendBlockGradsBack`.

        The Function's backward records each block's grad into
        ``_cache._grad_accum[mb_index]`` (payload for inspection /
        tests) and then P2P-sends it back to the block's producer rank.

        Tag scheme: each cached block has a producer stage ``p`` and an
        index-within-producer ``b``. Its send-back tag is
        ``_grad_tag_base(mb, p) + b``. The producer side of
        :class:`_RecvBlockGradsFromConsumers` computes the SAME
        ``_grad_tag_base(mb, p)`` locally (p is the producer's own
        stage_id) and irecvs at ``+ b`` for each of its committed
        blocks. Since one producer may be drained by many consumers,
        the consumer's identity is implicit in the (src, dst) of the
        underlying P2P op; we just need the tag unique per
        ``(mb, producer, block)`` tuple which ``_grad_tag_base`` reserves.
        """
        if not blocks:
            return []

        producer_ranks = [m[0] for m in meta]
        tags = [
            _grad_tag_base(mb_index, producer_stage) + idx_in_producer
            for (_, producer_stage, idx_in_producer) in meta
        ]
        wrapped = _SendBlockGradsBack.apply(
            producer_ranks,
            tags,
            self._group,
            self._cache,
            mb_index,
            *blocks,
        )
        return list(wrapped)

    def _wrap_producer_blocks(self, blocks_tensor: torch.Tensor) -> torch.Tensor:
        """Route blocks this stage commits through
        :class:`_RecvBlockGradsFromConsumers` so every later stage's
        send-back lands in the local grad.
        """
        if blocks_tensor.shape[0] == 0:
            return blocks_tensor
        # Bookkeeping: number of blocks this stage committed at the
        # current microbatch. _RecvBlockGradsFromConsumers reads it off
        # ctx.shape[0] directly; we stash it too for diagnostics.
        self._committed_block_count[self.stage_id] = blocks_tensor.shape[0]

        # Consumer ranks = every stage with id > this one. Each of them
        # caches our committed blocks and will isend back one grad per
        # (consumer, block) pair.
        consumer_ranks = [
            self._stage_to_rank.get(s, s)
            for s in range(self.stage_id + 1, self.num_stages)
        ]
        if not consumer_ranks:
            return blocks_tensor  # last stage has no consumers below it

        # Tag base keyed on (mb, this producer stage); block index is
        # added inside the Function. Must match the consumer-side
        # send tag computed in ``_wrap_cached_prefix_for_send_back``.
        mb = self._current_mb()
        tag_base = _grad_tag_base(mb, self.stage_id)
        return _RecvBlockGradsFromConsumers.apply(
            consumer_ranks, self._group, tag_base, blocks_tensor
        )

    # ----- schedule hook points --------------------------------------- #

    def on_microbatch_end(self, mb_index: int) -> None:
        """Called by the schedule wrapper when a microbatch's
        forward+backward are complete, so the adapter can drop its
        cache. Safe to call redundantly.
        """
        self._cache.drop(mb_index)

    def extra_repr(self) -> str:
        return f"stage_id={self.stage_id}, num_stages={self.num_stages}"


def _grad_tag_base(mb_index: int, producer_stage_id: int) -> int:
    """Pick a P2P tag base that's unique per (mb, producer_stage).

    Block index within the producer is added on top. We reserve 1024
    blocks of tag space per (mb, producer) pair; AttnRes' num_blocks
    is tiny (order 8) so this is wildly conservative. Different mb /
    producer combinations must not collide because backward sends can
    be in flight concurrently when the schedule interleaves.
    """
    return (mb_index * 1024 * 64) + (producer_stage_id * 1024)


# --------------------------------------------------------------------------- #
# Stage iteration + monkey-patching
# --------------------------------------------------------------------------- #


def _iter_schedule_stages(schedule: _PipelineSchedule):
    """Yield the ``PipelineStage`` objects held by a schedule.

    ``PipelineScheduleSingle`` stores one stage on ``_stage``;
    ``PipelineScheduleMulti`` holds a list on ``_stages``. Both
    attributes are private in torch today; the fallback branch catches
    future renames early with an explicit error instead of an opaque
    AttributeError.
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


def _install_mb_index_patch(stage, adapter: CrossStageCacheAdapter) -> None:
    """Monkey-patch ``stage.forward_one_chunk`` and
    ``stage.backward_one_chunk`` so the adapter learns the mb index the
    schedule hands it.

    Both patches are per-stage (bound via closure to the specific stage
    + adapter pair), so wrapping multiple stages in the same process
    (virtual-pipelining case) correctly demuxes.

    After each patched call returns, we drop the mb key from the
    thread-local: not strictly necessary but keeps the state of the
    cache observable in tests.
    """
    adapter_key = id(adapter)
    orig_forward_one_chunk = stage.forward_one_chunk
    orig_backward_one_chunk = stage.backward_one_chunk

    def patched_forward_one_chunk(
        fwd_chunk_id, args, kwargs=None, save_forward_output=True
    ):
        _set_mb_index(adapter_key, fwd_chunk_id)
        try:
            return orig_forward_one_chunk(
                fwd_chunk_id,
                args,
                kwargs,
                save_forward_output=save_forward_output,
            )
        finally:
            _set_mb_index(adapter_key, None)

    def patched_backward_one_chunk(
        bwd_chunk_id,
        loss=None,
        full_backward: bool = True,
        last_backward: bool = False,
    ):
        _set_mb_index(adapter_key, bwd_chunk_id)
        try:
            return orig_backward_one_chunk(
                bwd_chunk_id,
                loss=loss,
                full_backward=full_backward,
                last_backward=last_backward,
            )
        finally:
            # Drop this microbatch's cache entry -- its forward+backward
            # pair is complete.
            adapter.on_microbatch_end(bwd_chunk_id)
            _set_mb_index(adapter_key, None)

    stage.forward_one_chunk = patched_forward_one_chunk
    stage.backward_one_chunk = patched_backward_one_chunk


# --------------------------------------------------------------------------- #
# FQN-split injection
# --------------------------------------------------------------------------- #


_ATTN_RES_EXTRA_LAST_STAGE_FQNS = ("final_attn_res_proj", "final_attn_res_norm")


def _inject_attn_res_fqns(model: nn.Module, kwargs: dict) -> None:
    """Extend the PP module-FQN split so AttnRes-specific top-level
    modules survive ``pipeline_module_split``.

    Core ``generate_llm_fqn_per_model_part`` only names ``tok_embeddings``,
    ``layers.*``, ``norm``, ``output``. Any other top-level child of the
    model -- in AttnRes' case ``final_attn_res_proj`` and
    ``final_attn_res_norm`` -- gets replaced with ``None`` on every
    stage (see ``pipeline_module_split`` in
    ``distributed/pipeline_parallel.py``), which blows up the last
    stage's forward when it tries to call the final cross-block
    aggregation's norm.

    We therefore compute the same FQN layout core would have produced,
    then append the AttnRes-specific modules to the LAST stage's list,
    and inject into ``parallelism.module_fqns_per_model_part``. If the
    user already supplied an explicit layout we respect theirs.
    """
    if not any(hasattr(model, n) for n in _ATTN_RES_EXTRA_LAST_STAGE_FQNS):
        return

    parallelism = kwargs.get("parallelism")
    if parallelism is None or parallelism.module_fqns_per_model_part is not None:
        return

    import math as _math

    from torch.distributed.pipelining.schedules import PipelineScheduleSingle

    from torchtitan.distributed.pipeline_parallel import (
        generate_llm_fqn_per_model_part,
        get_schedule_class,
    )

    parallel_dims = kwargs["parallel_dims"]
    pp = parallel_dims.pp
    if pp <= 1:
        return

    model_config = kwargs.get("model_config")
    if model_config is None or not hasattr(model_config, "layers"):
        return
    num_layers = len(model_config.layers)
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is not None:
        num_virtual_stages = _math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
    else:
        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
        is_single = issubclass(schedule_class, PipelineScheduleSingle)
        stages_per_rank = 1 if is_single else 2
        num_virtual_stages = pp * stages_per_rank

    fqns = generate_llm_fqn_per_model_part(
        num_virtual_stages, num_layers, input_weight, output_weight
    )
    extras = [n for n in _ATTN_RES_EXTRA_LAST_STAGE_FQNS if hasattr(model, n)]
    fqns[-1].extend(extras)
    parallelism.module_fqns_per_model_part = fqns


# --------------------------------------------------------------------------- #
# Custom pipelining_fn
# --------------------------------------------------------------------------- #


def _build_stage_to_rank(stages) -> dict[int, int]:
    """Map ``stage_id -> rank in pipeline process group``.

    For the non-VP path (one stage per rank) it's identity. For looped
    / interleaved VP the map is non-trivial: stage 0 and N are both on
    rank 0, stage 1 and N+1 on rank 1, etc. We read it straight off each
    stage's ``.group_rank`` which the PipelineStage base class populates
    from ``dist.get_rank(self.group)`` at stage construction.
    """
    mapping: dict[int, int] = {}
    for stage in stages:
        mapping[stage.stage_index] = stage.group_rank
    return mapping


def pipeline_llm_with_cache_adapter(model: nn.Module, **kwargs):
    """Custom ``pipelining_fn`` for AttnRes.

    Calls core :func:`torchtitan.distributed.pipeline_parallel.pipeline_llm`,
    then (when ``TORCHTITAN_ATTNRES_CACHE=1``) wraps each stage's
    submodule with :class:`CrossStageCacheAdapter` AND monkey-patches
    ``forward_one_chunk`` / ``backward_one_chunk`` on the stage so the
    adapter receives the schedule's microbatch index.

    Before delegating to core, we extend the auto-generated FQN split
    to include AttnRes-specific top-level modules on the last stage
    (``final_attn_res_proj``, ``final_attn_res_norm``) -- otherwise
    core's module-split step replaces them with ``None`` and the last
    stage's final cross-block aggregation crashes.

    All kwargs are forwarded unchanged to core; the signature must stay
    in sync with ``ParallelizeFunction``/pipelining contract.

    Returns the same 4-tuple as core ``pipeline_llm``.
    """
    # Deferred import so this module stays importable without torchtitan's
    # distributed wiring (e.g. for CPU unit tests).
    from torchtitan.distributed.pipeline_parallel import pipeline_llm

    _inject_attn_res_fqns(model, kwargs)

    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )

    if not adapter_enabled():
        return pp_schedule, model_parts, has_first_stage, has_last_stage

    stages = list(_iter_schedule_stages(pp_schedule))
    num_stages = len(stages)
    stage_to_rank = _build_stage_to_rank(stages)

    for i, stage in enumerate(stages):
        adapter = CrossStageCacheAdapter(
            stage.submod,
            stage_id=stage.stage_index,
            num_stages=num_stages,
            group=getattr(stage, "group", None),
            stage_to_rank=stage_to_rank,
        )
        stage.submod = adapter
        _install_mb_index_patch(stage, adapter)
        # Also update model_parts so downstream (optimizer build,
        # checkpoint state_dict) sees the wrapped module. torchtitan
        # iterates model_parts to bind optimizers and compilers.
        if i < len(model_parts):
            model_parts[i] = adapter

    return pp_schedule, model_parts, has_first_stage, has_last_stage
