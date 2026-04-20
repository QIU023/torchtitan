# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-stage caching adapter + custom ``pipelining_fn`` for AttnRes.

:class:`CrossStageCacheAdapter` wraps a per-stage AttnRes decoder. In
delta mode (Interleaved1F1B with :class:`BlockLayoutTables` from
:mod:`.layout`) each hop ships only the blocks the receiver's shared
rank cache doesn't already have; the receiver rebuilds the full stack
by concatenating its cached prefix with the incoming delta. Per-block
backward grads P2P-travel to the original producer via
:class:`_SendBlockGradsBack` (consumer) and
:class:`_RecvBlockGradsFromConsumers` (producer). Without layout tables
the adapter is a naive full-stack passthrough.

Grad send-back transport: :class:`_SendBlockGradsBack.backward` and
:class:`_RecvBlockGradsFromConsumers.backward` ONLY compute local grad
math; they do NOT issue any NCCL calls. The actual cross-rank P2P is
deferred to :meth:`CrossStageCacheAdapter._flush_grad_sendback`, which
runs OUTSIDE the autograd engine's backward thread, right after
``stage.backward_one_chunk`` returns. This is required under
Interleaved1F1B: calling NCCL inside ``autograd.Function.backward``
deadlocks because the engine is single-threaded depth-first and the
schedule's own SEND_B/RECV_B ops interleave unpredictably with custom
P2P. Flushing after ``backward_one_chunk`` removes the interleaving.

:func:`pipeline_llm_with_cache_adapter` is a ``pipelining_fn`` plugged
into the experiment's ``ModelSpec``; it delegates to core
``pipeline_llm`` and (when ``TORCHTITAN_ATTNRES_CACHE=1``) wraps each
stage's submod. Schedule must be Interleaved1F1B; otherwise we warn.

Microbatch keying: the adapter keys its per-microbatch cache by the
schedule-owned integer chunk id. ``forward_one_chunk`` /
``backward_one_chunk`` are monkey-patched to stash the index on a
thread-local keyed per-adapter; forward and backward run synchronously
on the same thread, so autograd hooks that fire during backward can
read the mb index. The integer key is stable across P2P crossings
(unlike ``id()`` of a tensor).

See ``adapter_design.md`` at the project root for the full state
machine and invariants.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

# Resolve Interleaved1F1B at import time so the schedule guard is a direct
# isinstance check instead of a string-match.
try:
    from torch.distributed.pipelining.schedules import get_schedule_class

    _INTERLEAVED_1F1B_CLASS = get_schedule_class("Interleaved1F1B")
except Exception:  # pragma: no cover - fallback for older torch
    _INTERLEAVED_1F1B_CLASS = None

from torchtitan.experiments.attn_res.attn_res import unstack_blocks
from torchtitan.experiments.attn_res.layout import (
    _grad_tag_base,
    _infer_block_layout_tables_from_stages,
    BlockLayoutTables,
)


def adapter_enabled() -> bool:
    """Env-flag gate: adapter is opt-in until we trust it."""
    return os.environ.get("TORCHTITAN_ATTNRES_CACHE") == "1"


# ----- Per-microbatch cache ------------------------------------------------ #


class _PerMicrobatchCache:
    """Maps ``mb_index`` -> cached blocks + per-block grads.

    Retained for back-compat with unit tests; :class:`RankLocalCache` is
    a strict superset used for the shared-across-virtual-stages path.
    Also tracks producer-side pending-recv markers stashed by
    :class:`_RecvBlockGradsFromConsumers.backward` so the adapter's
    deferred P2P flush knows which recvs to post.
    """

    def __init__(self) -> None:
        self._forward: dict[int, list[torch.Tensor]] = {}
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        self._grad_accum: dict[int, list[torch.Tensor | None]] = defaultdict(list)
        # (mb_index, producer_stage_id) -> dict describing the recv.
        self._pending_recv: dict[tuple[int, int], dict] = {}

    def put_forward(
        self, mb_index: int, blocks: list[torch.Tensor],
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
        keys = [k for k in self._pending_recv if k[0] == mb_index]
        for k in keys:
            self._pending_recv.pop(k, None)

    def add_grad(self, mb_index: int, block_idx: int, grad: torch.Tensor) -> None:
        slot = self._grad_accum[mb_index]
        while len(slot) <= block_idx:
            slot.append(None)
        contrib = grad.detach()
        slot[block_idx] = contrib.clone() if slot[block_idx] is None else slot[block_idx] + contrib

    def pop_grads(self, mb_index: int) -> list[torch.Tensor | None]:
        return self._grad_accum.pop(mb_index, [])

    def mark_pending_recv(
        self, mb_index: int, *, producer_stage_id: int, num_blocks: int,
        per_block_shape: tuple, dtype, device,
        consumer_ranks: tuple, tag_base: int,
    ) -> None:
        """Record that ``producer_stage_id`` expects ``num_blocks`` recvs
        per consumer. Called from ``_RecvBlockGradsFromConsumers.backward``;
        consumed by the adapter's :meth:`_flush_grad_sendback`.
        """
        self._pending_recv[(mb_index, producer_stage_id)] = {
            "num_blocks": num_blocks,
            "per_block_shape": tuple(per_block_shape),
            "dtype": dtype,
            "device": device,
            "consumer_ranks": tuple(consumer_ranks),
            "tag_base": tag_base,
        }

    def pop_pending_recv(self, mb_index: int, producer_stage_id: int):
        return self._pending_recv.pop((mb_index, producer_stage_id), None)


# ----- Rank-shared cache across virtual stages ----------------------------- #

class RankLocalCache:
    """Per-rank, per-microbatch accumulator shared across virtual stages.

    Every adapter on the same physical rank reads/writes the SAME cache
    (Kimi §4.1 invariant). Canonical API: ``append`` / ``get_blocks`` /
    ``get_meta``. ``put_forward`` / ``get_forward`` / ``get_producer_meta``
    are back-compat shims.

    Also holds deferred-P2P state the adapter's grad-send-back flush
    consumes: ``_grad_accum`` (consumer side, written by
    :class:`_SendBlockGradsBack.backward`) and ``_pending_recv``
    (producer side, written by
    :class:`_RecvBlockGradsFromConsumers.backward`).
    """

    def __init__(self) -> None:
        self._blocks: dict[int, list[torch.Tensor]] = {}
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        self._grad_accum: dict[int, list[torch.Tensor | None]] = defaultdict(list)
        self._pending_recv: dict[tuple[int, int], dict] = {}

    def append(
        self, mb_index: int, block: torch.Tensor, meta: tuple[int, int, int],
    ) -> None:
        self._blocks.setdefault(mb_index, []).append(block)
        self._producer_meta.setdefault(mb_index, []).append(meta)

    def get_blocks(self, mb_index: int) -> list[torch.Tensor]:
        return self._blocks.get(mb_index, [])

    def get_meta(self, mb_index: int) -> list[tuple[int, int, int]]:
        return self._producer_meta.get(mb_index, [])

    # Back-compat shims; ``put_forward`` overwrites any prior per-mb state.
    def put_forward(
        self, mb_index: int, blocks: list[torch.Tensor],
        producer_meta: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self._blocks[mb_index] = list(blocks)
        if producer_meta is not None:
            self._producer_meta[mb_index] = list(producer_meta)

    def get_forward(self, mb_index: int) -> list[torch.Tensor]:
        return self.get_blocks(mb_index)

    def get_producer_meta(self, mb_index: int) -> list[tuple[int, int, int]]:
        return self.get_meta(mb_index)

    def drop(self, mb_index: int) -> None:
        self._blocks.pop(mb_index, None)
        self._producer_meta.pop(mb_index, None)
        self._grad_accum.pop(mb_index, None)
        keys = [k for k in self._pending_recv if k[0] == mb_index]
        for k in keys:
            self._pending_recv.pop(k, None)

    def add_grad(self, mb_index: int, block_idx: int, grad: torch.Tensor) -> None:
        slot = self._grad_accum[mb_index]
        while len(slot) <= block_idx:
            slot.append(None)
        contrib = grad.detach()
        slot[block_idx] = contrib.clone() if slot[block_idx] is None else slot[block_idx] + contrib

    def pop_grads(self, mb_index: int) -> list[torch.Tensor | None]:
        return self._grad_accum.pop(mb_index, [])

    def mark_pending_recv(
        self, mb_index: int, *, producer_stage_id: int, num_blocks: int,
        per_block_shape: tuple, dtype, device,
        consumer_ranks: tuple, tag_base: int,
    ) -> None:
        """Record a producer-side pending-recv descriptor for the flush."""
        self._pending_recv[(mb_index, producer_stage_id)] = {
            "num_blocks": num_blocks,
            "per_block_shape": tuple(per_block_shape),
            "dtype": dtype,
            "device": device,
            "consumer_ranks": tuple(consumer_ranks),
            "tag_base": tag_base,
        }

    def pop_pending_recv(self, mb_index: int, producer_stage_id: int):
        return self._pending_recv.pop((mb_index, producer_stage_id), None)


# One RankLocalCache per pipeline-group rank, shared by every adapter
# on that rank. Lock-protected against concurrent construction.
_rank_caches: dict[int, RankLocalCache] = {}
_rank_caches_lock = threading.Lock()


def _get_or_create_rank_cache(pp_rank: int) -> RankLocalCache:
    """Return (creating if absent) the shared cache for ``pp_rank``."""
    cache = _rank_caches.get(pp_rank)
    if cache is not None:
        return cache
    with _rank_caches_lock:
        cache = _rank_caches.get(pp_rank)
        if cache is None:
            cache = RankLocalCache()
            _rank_caches[pp_rank] = cache
        return cache


def _reset_rank_caches_for_testing() -> None:
    """Clear the module-level registry. Unit-test isolation only."""
    with _rank_caches_lock:
        _rank_caches.clear()


# ----- Autograd plumbing for cross-stage grad send-back -------------------- #

class _SendBlockGradsBack(torch.autograd.Function):
    """Consumer-side identity forward; backward ONLY accumulates grads.

    The Function exists so each cached-prefix block participates in the
    consumer's local autograd graph (the clones give it a graph node we
    own). Backward records ``grad_outputs[i]`` into
    ``cache._grad_accum[mb_index]`` keyed by block-position; the actual
    P2P send to the producer rank is deferred to
    :meth:`CrossStageCacheAdapter._flush_grad_sendback`, which runs
    OUTSIDE the autograd engine (right after ``backward_one_chunk``).
    Returning ``None`` for each tensor input stops the local graph here:
    the "real" grad for cached blocks leaves via the deferred isend.
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
        # Clone so the Function is the source of the blocks in the local
        # autograd graph (otherwise backward would route past).
        return tuple(b.clone() for b in blocks)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Accumulate ONLY — no NCCL here. The flush helper picks these
        # up after ``backward_one_chunk`` returns and posts the isends.
        # During the flush's secondary backward (which walks from the
        # producer's committed-block tensor back through the wrapped
        # model, potentially crossing cached_prefix inputs that were
        # routed through this Function), the cache's
        # ``_in_secondary_backward`` flag is set — skip accumulation in
        # that case to avoid double-counting.
        if ctx.cache is not None and not getattr(
            ctx.cache, "_in_secondary_backward", False
        ):
            for i, g in enumerate(grad_outputs):
                if g is None:
                    g = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
                ctx.cache.add_grad(ctx.mb_index, i, g)
        return (None, None, None, None, None) + tuple(None for _ in grad_outputs)


class _RecvBlockGradsFromConsumers(torch.autograd.Function):
    """Producer-side identity forward; backward ONLY returns local grad.

    The Function exists so committed blocks participate in the
    producer's local autograd graph (the clone gives it a graph node we
    own). Backward returns ``grad_output`` unchanged so the local
    contribution flows into params normally. The cross-stage
    contribution (a sum of per-consumer irecvs) is applied in a separate
    tiny backward pass driven by
    :meth:`CrossStageCacheAdapter._flush_grad_sendback`, which posts the
    irecvs OUTSIDE the autograd engine (after ``backward_one_chunk``
    returns). Marker recorded on the cache so the flush helper knows
    which producer-stage block set to poll for.
    """

    @staticmethod
    def forward(ctx, consumer_ranks, group, tag_base, cache, mb_index,
                producer_stage_id, blocks_tensor):
        ctx.consumer_ranks = tuple(consumer_ranks)
        ctx.group = group
        ctx.tag_base = tag_base
        ctx.shape = blocks_tensor.shape
        ctx.dtype = blocks_tensor.dtype
        ctx.device = blocks_tensor.device
        ctx.cache = cache
        ctx.mb_index = mb_index
        ctx.producer_stage_id = producer_stage_id
        return blocks_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Record marker: flush helper reads this to know how many recvs
        # to post per committed block for this (mb, producer_stage).
        # No NCCL here; returning ``grad_output`` unchanged lets the
        # local producer-side autograd propagate normally into params.
        if ctx.cache is not None:
            ctx.cache.mark_pending_recv(
                ctx.mb_index,
                producer_stage_id=ctx.producer_stage_id,
                num_blocks=ctx.shape[0],
                per_block_shape=tuple(ctx.shape[1:]),
                dtype=ctx.dtype,
                device=ctx.device,
                consumer_ranks=ctx.consumer_ranks,
                tag_base=ctx.tag_base,
            )
        return (None, None, None, None, None, None, grad_output)


# ----- Microbatch-index threading ------------------------------------------ #

# Each adapter stashes its *current* mb index under its own object id.
# Forward and backward of a single mb run on the same thread.
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


# ----- state_dict key rewriting -------------------------------------------- #
# The adapter stores its wrapped model under ``self.wrapped``. The Llama3 HF
# state_dict_adapter keys off raw names like ``tok_embeddings.weight``, so
# we strip the prefix on save and re-prepend on load.

_WRAPPED_PREFIX = "wrapped."


def _strip_wrapped_prefix_hook(
    module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict
) -> dict:
    """Save hook: drop the adapter's ``wrapped.`` namespace."""
    target = prefix + _WRAPPED_PREFIX
    rewrites = [k for k in state_dict if k.startswith(target)]
    for old_key in rewrites:
        new_key = prefix + old_key[len(target) :]
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def _prepend_wrapped_prefix_pre_hook(
    state_dict: dict, prefix: str, local_metadata: dict, strict: bool,
    missing_keys: list, unexpected_keys: list, error_msgs: list,
) -> None:
    """Load pre-hook: add the ``wrapped.`` namespace back."""
    target = prefix + _WRAPPED_PREFIX
    rewrites = [
        k for k in state_dict if k.startswith(prefix) and not k.startswith(target)
    ]
    for old_key in rewrites:
        inner = old_key[len(prefix) :]
        state_dict[target + inner] = state_dict.pop(old_key)


# ----- The adapter module -------------------------------------------------- #

class CrossStageCacheAdapter(nn.Module):
    """Wraps an ``AttnResLlama3Model`` stage with cross-stage caching.

    In delta mode (``layout_tables`` supplied) each forward pulls earlier
    blocks from the shared :class:`RankLocalCache`, receives the incoming
    delta, rebuilds the full block stack in block-index order, and routes
    cached/committed blocks through :class:`_SendBlockGradsBack` /
    :class:`_RecvBlockGradsFromConsumers` for the grad send-back path.
    Without layout tables the adapter is a naive passthrough.

    Adapters sharing a ``pp_rank`` share ONE :class:`RankLocalCache`.
    """

    def __init__(
        self, wrapped: nn.Module, *,
        stage_id: int, num_stages: int,
        group: "dist.ProcessGroup | None" = None,
        stage_to_rank: dict[int, int] | None = None,
        pp_rank: int | None = None,
        layout_tables: BlockLayoutTables | None = None,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.stage_id = stage_id
        self.num_stages = num_stages
        self._group = group
        self._stage_to_rank = stage_to_rank or {i: i for i in range(num_stages)}
        if pp_rank is None:
            pp_rank = self._stage_to_rank.get(stage_id, stage_id)
        self.pp_rank = pp_rank
        self._cache = _get_or_create_rank_cache(self.pp_rank)
        self._layout = layout_tables
        self._delta_mode = layout_tables is not None
        # Diagnostic counter; Functions read shapes off ctx directly.
        self._committed_block_count: dict[int, int] = {}

        # Delta mode: wrapped returns only own commits. Naive mode: full stack.
        if hasattr(wrapped, "_return_only_new_blocks"):
            wrapped._return_only_new_blocks = bool(self._delta_mode)
        else:
            warnings.warn(
                "Wrapped model does not expose _return_only_new_blocks; "
                "adapter will run in naive (full-stack) mode.",
                stacklevel=2,
            )

        # Hide ``wrapped.`` from state_dict consumers.
        self._register_state_dict_hook(_strip_wrapped_prefix_hook)
        self._register_load_state_dict_pre_hook(
            _prepend_wrapped_prefix_pre_hook, with_module=False
        )

    # Torchtitan trainer iterates model_parts and calls init_weights /
    # init_states; __getattr__ delegates the rest.
    def init_weights(self, *args, **kwargs) -> None:
        self.wrapped.init_weights(*args, **kwargs)

    def init_states(self, *args, **kwargs) -> None:
        self.wrapped.init_states(*args, **kwargs)

    def __getattr__(self, name: str):
        """Fall back to the wrapped model for unknown attributes."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        wrapped = self.__dict__.get("_modules", {}).get("wrapped")
        if wrapped is None:
            raise AttributeError(
                f"'CrossStageCacheAdapter' object has no attribute '{name}' "
                "and wrapped model is not yet bound."
            )
        return getattr(wrapped, name)

    def _adapter_key(self) -> int:
        return id(self)

    def _current_mb(self) -> int:
        mb = _current_mb_index(self._adapter_key())
        assert mb is not None, (
            "CrossStageCacheAdapter.forward called without an mb index; "
            "stage.forward_one_chunk monkey-patch missing."
        )
        return mb

    @staticmethod
    def _has_blocks_signature(args) -> bool:
        """True if ``args[1]`` is a 4-D block tensor (middle/last stage)."""
        return (
            len(args) >= 2
            and isinstance(args[1], torch.Tensor)
            and args[1].dim() == 4
        )

    def _call_wrapped_naive(self, args, kwargs):
        """Dispatch to the wrapped model with the appropriate signature."""
        if self._has_blocks_signature(args):
            partial, new_blocks_tensor, *rest = args
            return self.wrapped(
                partial, *rest, blocks=new_blocks_tensor, **kwargs
            )
        return self.wrapped(*args, blocks=None, **kwargs)

    def forward(self, *args, **kwargs):
        """Dispatch to delta-P2P, shape inference, or naive passthrough."""
        # ``PipelineStage._shape_inference`` invokes ``self.submod(...)``
        # directly, bypassing the ``forward_one_chunk`` patch that stashes
        # the mb index. Route to the shape-inference helper in that case.
        if _current_mb_index(self._adapter_key()) is None:
            return self._forward_shape_inference(*args, **kwargs)
        if self._delta_mode:
            return self._forward_delta(*args, **kwargs)
        return self._call_wrapped_naive(args, kwargs)

    def _forward_shape_inference(self, *args, **kwargs):
        """Run wrapped model and reshape its blocks output to the delta
        size the runtime will emit; pipelining uses this return shape to
        size the next stage's recv buffer.
        """
        wrapped_out = self._call_wrapped_naive(args, kwargs)
        if not isinstance(wrapped_out, tuple):  # last stage
            return wrapped_out

        partial_out, new_blocks_out = wrapped_out
        if not self._delta_mode or self._layout is None:
            return partial_out, new_blocks_out

        expected_K = len(self._layout.delta_to_send(self.stage_id))
        if expected_K == new_blocks_out.shape[0]:
            return partial_out, new_blocks_out

        per_block_shape = (
            new_blocks_out.shape[1:]
            if new_blocks_out.shape[0] > 0 else partial_out.shape
        )
        return partial_out, partial_out.new_zeros((expected_K, *per_block_shape))

    def _forward_delta(self, *args, **kwargs):
        """Interleaved1F1B delta-P2P forward (spec §4.1)."""
        mb = self._current_mb()
        layout = self._layout
        assert layout is not None, "_forward_delta called without layout tables"

        if self.stage_id == 0:
            partial_out, new_blocks_tensor = self.wrapped(*args, blocks=None, **kwargs)
            return self._finish_forward(
                mb, partial_out, new_blocks_tensor,
                prev_recv_tensor=None, incoming_block_indices=[],
            )

        if not self._has_blocks_signature(args):
            return self.wrapped(*args, blocks=None, **kwargs)
        partial, recv_delta_tensor, *rest = args

        # Unstack incoming delta; wire order MUST match sender's layout.
        incoming_block_indices = layout.delta_to_send(self.stage_id - 1)
        recv_list = unstack_blocks(recv_delta_tensor)
        assert len(recv_list) == len(incoming_block_indices), (
            f"Incoming delta size mismatch at stage {self.stage_id} mb {mb}: "
            f"expected {len(incoming_block_indices)}, got {len(recv_list)}."
        )

        # Pull earlier cached blocks, route through send-back Function.
        earlier_blocks = list(self._cache.get_blocks(mb))
        earlier_meta = list(self._cache.get_meta(mb))
        wrapped_earlier = self._wrap_cached_prefix_for_send_back(
            earlier_blocks, earlier_meta, mb
        )
        cached_indices = [
            layout.commits_at(meta[1])[meta[2]] for meta in earlier_meta
        ]

        # Rebuild the full blocks tensor in block-index order.
        pairs = list(zip(cached_indices, wrapped_earlier)) + list(
            zip(incoming_block_indices, recv_list)
        )
        pairs.sort(key=lambda p: p[0])
        ordered_blocks = [p[1] for p in pairs]
        blocks_tensor = (
            torch.stack(ordered_blocks, dim=0)
            if ordered_blocks else recv_delta_tensor
        )

        wrapped_ret = self.wrapped(partial, *rest, blocks=blocks_tensor, **kwargs)

        if self.stage_id == self.num_stages - 1:
            # Last stage: keepalive keeps recv tensor on the autograd graph.
            return self._keepalive_touch(wrapped_ret, recv_delta_tensor)

        partial_out, new_blocks_tensor = wrapped_ret
        return self._finish_forward(
            mb, partial_out, new_blocks_tensor,
            prev_recv_tensor=recv_delta_tensor,
            incoming_block_indices=incoming_block_indices,
        )

    def _finish_forward(
        self, mb: int, partial_out: torch.Tensor,
        new_blocks_tensor: torch.Tensor, *,
        prev_recv_tensor: torch.Tensor | None,
        incoming_block_indices: list[int],
    ):
        """Common tail for first + middle stages: append relayed and
        committed blocks to the shared rank cache, wrap own commits
        through :class:`_RecvBlockGradsFromConsumers`, and stack the
        outgoing delta.
        """
        layout = self._layout
        assert layout is not None
        my_commits = layout.commits_at(self.stage_id)
        assert new_blocks_tensor.shape[0] == len(my_commits), (
            f"Wrapped model returned {new_blocks_tensor.shape[0]} new "
            f"blocks at stage {self.stage_id}, expected {len(my_commits)}."
        )

        # Append relayed blocks so later virtual stages on this rank see
        # them; producer metadata comes from the static layout.
        if prev_recv_tensor is not None:
            recv_list = unstack_blocks(prev_recv_tensor)
            for bidx, blk in zip(incoming_block_indices, recv_list):
                producer_stage = layout.producer_stage_of_block(bidx)
                producer_rank = self._stage_to_rank.get(producer_stage, producer_stage)
                block_idx_in_producer = layout.commits_at(producer_stage).index(bidx)
                self._cache.append(
                    mb, blk,
                    (producer_rank, producer_stage, block_idx_in_producer),
                )

        # Append own commits; wrap through _RecvBlockGradsFromConsumers
        # so local grads absorb every consumer's backward irecv.
        new_blocks_list = unstack_blocks(new_blocks_tensor)
        for local_idx, (bidx, blk) in enumerate(zip(my_commits, new_blocks_list)):
            self._cache.append(mb, blk, (self.pp_rank, self.stage_id, local_idx))
        wrapped_new_blocks_tensor = self._wrap_producer_blocks(new_blocks_tensor, mb=mb)

        # Build outgoing delta: subset of (cache + new), by canonical bidx.
        out_indices = layout.delta_to_send(self.stage_id)
        cache_by_bidx = {
            layout.commits_at(meta[1])[meta[2]]: blk
            for meta, blk in zip(
                self._cache.get_meta(mb), self._cache.get_blocks(mb)
            )
        }
        wrapped_new_by_bidx = {
            my_commits[i]: wrapped_new_blocks_tensor[i]
            for i in range(wrapped_new_blocks_tensor.shape[0])
        }
        send_pieces: list[torch.Tensor] = []
        for bidx in out_indices:
            if bidx in wrapped_new_by_bidx:
                send_pieces.append(wrapped_new_by_bidx[bidx])
            elif bidx in cache_by_bidx:
                send_pieces.append(cache_by_bidx[bidx])
            else:
                raise RuntimeError(
                    f"Outgoing delta asks for block {bidx} at stage "
                    f"{self.stage_id} but it's neither cached nor committed."
                )

        out_blocks_tensor = (
            torch.stack(send_pieces, dim=0) if send_pieces
            else partial_out.new_zeros((0, *partial_out.shape))
        )
        partial_out = self._keepalive_touch(partial_out, prev_recv_tensor)
        return partial_out, out_blocks_tensor

    @staticmethod
    def _keepalive_touch(payload, prev_recv_tensor: torch.Tensor | None):
        """Ensure ``prev_recv_tensor`` is on the autograd graph that
        produces ``payload``. Preserves tuple returns.
        """
        if prev_recv_tensor is None:
            return payload
        touch = 0.0 * prev_recv_tensor.sum()
        if isinstance(payload, tuple):
            head, *tail = payload
            return (head + touch, *tail)
        return payload + touch

    def _wrap_cached_prefix_for_send_back(
        self, blocks: list[torch.Tensor],
        meta: list[tuple[int, int, int]], mb_index: int,
    ) -> list[torch.Tensor]:
        """Route each cached block through :class:`_SendBlockGradsBack`.

        Tag: ``_grad_tag_base(mb, producer_stage) + block_idx_in_producer``
        so the producer's :class:`_RecvBlockGradsFromConsumers` irecvs
        at the matching tag.
        """
        if not blocks:
            return []
        producer_ranks = [m[0] for m in meta]
        tags = [
            _grad_tag_base(mb_index, producer_stage) + idx_in_producer
            for (_, producer_stage, idx_in_producer) in meta
        ]
        wrapped = _SendBlockGradsBack.apply(
            producer_ranks, tags, self._group, self._cache, mb_index, *blocks,
        )
        return list(wrapped)

    def _wrap_producer_blocks(
        self, blocks_tensor: torch.Tensor, *, mb: int | None = None
    ) -> torch.Tensor:
        """Route commits through :class:`_RecvBlockGradsFromConsumers`
        so every downstream consumer's send-back lands in the local grad.
        """
        if blocks_tensor.shape[0] == 0:
            return blocks_tensor
        self._committed_block_count[self.stage_id] = blocks_tensor.shape[0]

        # Consumer ranks: delta mode -> static layout; else every later stage.
        if self._layout is not None:
            consumer_stages = self._layout.consumer_stages_of(self.stage_id)
        else:
            consumer_stages = list(range(self.stage_id + 1, self.num_stages))
        consumer_ranks = [self._stage_to_rank.get(s, s) for s in consumer_stages]
        if not consumer_ranks:
            return blocks_tensor

        if mb is None:
            mb = self._current_mb()
        tag_base = _grad_tag_base(mb, self.stage_id)
        return _RecvBlockGradsFromConsumers.apply(
            consumer_ranks, self._group, tag_base,
            self._cache, mb, self.stage_id, blocks_tensor,
        )

    def _flush_grad_sendback(self, mb_index: int) -> None:
        """Post the deferred grad send-back P2P for ``mb_index``.

        Runs AFTER ``stage.backward_one_chunk`` returns (invoked from
        the backward monkey-patch), so it's outside the autograd
        engine's backward thread and can safely call NCCL.

        Sequence:

        1. Consumer side: iterate ``cache._grad_accum[mb]`` entries; for
           each block whose producer rank is NOT this rank, build a
           ``P2POp(isend, grad, producer_rank, group, tag)``.
        2. Producer side: for each committed block at this stage, read
           the pending-recv descriptor recorded by
           :class:`_RecvBlockGradsFromConsumers.backward` and build a
           ``P2POp(irecv, buf, consumer_rank, group, tag)`` per
           (consumer_rank, block_idx). Track the buffers so we can
           reduce them into a per-block sum after the wait completes.
        3. ``batch_isend_irecv`` the ops (one NCCL group call) and
           ``wait()`` on every returned handle. If ``batch_isend_irecv``
           is unavailable we fall back to per-op ``isend/irecv`` + wait.
        4. For each committed block with received contributions, run
           ``torch.autograd.backward([cached_block], [recv_sum])`` so
           the remote contribution flows into the producer's params.
           The primary backward pass MUST have been run with
           ``retain_graph=True`` so the producer's subgraph activations
           are still alive; the backward monkey-patch forces this.
        """
        if not self._delta_mode or self._layout is None:
            return
        if self._group is None or not (dist.is_available() and dist.is_initialized()):
            return

        cache = self._cache
        meta = list(cache.get_meta(mb_index))
        blocks = list(cache.get_blocks(mb_index))
        grads = cache.pop_grads(mb_index)

        p2p_ops: list = []
        # Keep handles for recv buffers so we can sum them after wait().
        # Key: (producer_stage_id, block_local_idx) -> list[torch.Tensor]
        recv_bufs: dict[tuple[int, int], list[torch.Tensor]] = {}

        # --- 1) Consumer side: isend each accumulated grad to its producer ---
        for i, grad in enumerate(grads):
            if grad is None:
                continue
            if i >= len(meta):
                continue
            producer_rank, producer_stage_id, idx_in_producer = meta[i]
            if producer_rank == self.pp_rank:
                # Same physical rank — the producer's flush will
                # pick up the grad from the shared cache directly, no
                # P2P needed. Re-stash it so the producer can find it
                # under the (stage, block_idx) key.
                cache.mark_pending_recv  # reference kept; flow uses _grad_accum readback
                # Write the grad under a (mb, producer_stage, idx) slot
                # the producer flush can poll. We use a dedicated local
                # dict on the cache for clarity.
                key = (mb_index, producer_stage_id, idx_in_producer)
                if not hasattr(cache, "_intra_rank_grads"):
                    cache._intra_rank_grads = {}  # type: ignore[attr-defined]
                cache._intra_rank_grads[key] = grad.detach()  # type: ignore[attr-defined]
                continue
            tag = _grad_tag_base(mb_index, producer_stage_id) + idx_in_producer
            p2p_ops.append(
                dist.P2POp(
                    dist.isend, grad.contiguous().detach(),
                    peer=producer_rank, group=self._group, tag=tag,
                )
            )

        # --- 2) Producer side: irecv each consumer contribution ---
        pending = cache.pop_pending_recv(mb_index, self.stage_id)
        my_commits = self._layout.commits_at(self.stage_id)
        if pending is not None and my_commits:
            num_blocks = pending["num_blocks"]
            per_block_shape = pending["per_block_shape"]
            dtype = pending["dtype"]
            device = pending["device"]
            tag_base = pending["tag_base"]
            for src_rank in pending["consumer_ranks"]:
                for b in range(num_blocks):
                    if src_rank == self.pp_rank:
                        # Same-rank consumer: grad lives in shared cache
                        # under the intra_rank_grads slot.
                        continue
                    buf = torch.zeros(per_block_shape, dtype=dtype, device=device)
                    p2p_ops.append(
                        dist.P2POp(
                            dist.irecv, buf,
                            peer=src_rank, group=self._group, tag=tag_base + b,
                        )
                    )
                    recv_bufs.setdefault((self.stage_id, b), []).append(buf)

        # --- 3) Drive the P2P and wait ---
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for r in reqs:
                r.wait()

        # --- 4) Apply the received sums via a tiny secondary backward ---
        # Also fold in any same-rank grads the consumer left in the
        # shared cache (intra_rank_grads).
        intra = getattr(cache, "_intra_rank_grads", None) if pending is not None else None

        if pending is not None and my_commits:
            summed: dict[int, torch.Tensor] = {}
            for b in range(pending["num_blocks"]):
                bufs = recv_bufs.get((self.stage_id, b), [])
                running: torch.Tensor | None = None
                for buf in bufs:
                    running = buf if running is None else running + buf
                if intra is not None:
                    intra_key = (mb_index, self.stage_id, b)
                    extra = intra.pop(intra_key, None)
                    if extra is not None:
                        running = extra.clone() if running is None else running + extra
                if running is not None:
                    summed[b] = running

            if summed:
                # Locate each committed-block tensor in the shared cache
                # by its (producer_stage=self.stage_id, block_local_idx)
                # meta. The producer append order matches local_idx.
                per_block_tensor: dict[int, torch.Tensor] = {}
                for pos, m in enumerate(meta):
                    if m[1] == self.stage_id:
                        per_block_tensor[m[2]] = blocks[pos]
                targets: list[torch.Tensor] = []
                target_grads: list[torch.Tensor] = []
                for b, g in summed.items():
                    tensor = per_block_tensor.get(b)
                    if tensor is None or not tensor.requires_grad:
                        continue
                    targets.append(tensor)
                    target_grads.append(g)
                if targets:
                    # Secondary backward pass for the cross-rank grad
                    # contribution. Restrict propagation to THIS stage's
                    # parameters via ``inputs=`` so we don't double-
                    # count grads that the primary backward has already
                    # routed through cached-prefix tensors (those are
                    # wrapped by _SendBlockGradsBack, whose backward
                    # also silently skips during the secondary pass
                    # via the cache's ``_in_secondary_backward`` flag).
                    # The primary backward kept the graph alive via
                    # ``retain_graph=True`` in the monkey-patch.
                    params = [
                        p for p in self.wrapped.parameters() if p.requires_grad
                    ]
                    cache._in_secondary_backward = True  # type: ignore[attr-defined]
                    try:
                        if params:
                            torch.autograd.backward(
                                targets,
                                grad_tensors=target_grads,
                                retain_graph=False,
                                inputs=params,
                            )
                        else:
                            # No local params; fall back to a plain backward
                            # (the graph will still free correctly).
                            torch.autograd.backward(
                                targets,
                                grad_tensors=target_grads,
                                retain_graph=False,
                            )
                    finally:
                        cache._in_secondary_backward = False  # type: ignore[attr-defined]

    def on_microbatch_end(self, mb_index: int) -> None:
        """Drop this mb's cache once forward+backward is done.

        Under VP only the LAST virtual stage on a rank drops (the shared
        cache is READ by later virtual stages); characterized by
        ``stage_id + P >= num_stages``.
        """
        if self._delta_mode:
            pp_size = self._layout.P if self._layout is not None else self.num_stages
            if self.stage_id + pp_size < self.num_stages:
                return
        self._cache.drop(mb_index)

    def extra_repr(self) -> str:
        return f"stage_id={self.stage_id}, num_stages={self.num_stages}"


# ----- Stage iteration + monkey-patching ----------------------------------- #

def _iter_schedule_stages(schedule: _PipelineSchedule):
    """Yield the ``PipelineStage`` objects a schedule holds."""
    if isinstance(schedule, PipelineScheduleSingle):
        yield schedule._stage
    elif isinstance(schedule, PipelineScheduleMulti):
        yield from schedule._stages
    else:
        raise RuntimeError(
            f"Unexpected pipeline schedule class {type(schedule).__name__}; "
            "extend _iter_schedule_stages."
        )


def _install_mb_index_patch(stage, adapter: CrossStageCacheAdapter) -> None:
    """Patch ``forward_one_chunk`` / ``backward_one_chunk`` to stash the
    schedule-owned mb index for the adapter. Per-(stage, adapter) via
    closure so multi-stage ranks (VP) demux correctly.

    The backward patch additionally (i) forces ``retain_graph=True`` on
    the primary ``torch.autograd.backward`` call so the secondary
    flush backward can re-enter the producer's subgraph, and (ii) fires
    :meth:`CrossStageCacheAdapter._flush_grad_sendback` right after the
    schedule's own backward returns, so all cross-rank grad P2P lives
    OUTSIDE the autograd engine's backward thread.
    """
    adapter_key = id(adapter)
    orig_fwd = stage.forward_one_chunk
    orig_bwd = stage.backward_one_chunk

    def patched_fwd(fwd_chunk_id, args, kwargs=None, save_forward_output=True):
        _set_mb_index(adapter_key, fwd_chunk_id)
        try:
            return orig_fwd(
                fwd_chunk_id, args, kwargs,
                save_forward_output=save_forward_output,
            )
        finally:
            _set_mb_index(adapter_key, None)

    def patched_bwd(
        bwd_chunk_id, loss=None, full_backward: bool = True,
        last_backward: bool = False,
    ):
        _set_mb_index(adapter_key, bwd_chunk_id)
        # Only install the retain-graph override when delta mode is
        # active AND a real process group is wired; naive mode and the
        # no-pg CPU test path don't need a secondary backward and
        # shouldn't pay the retain-graph memory cost.
        override_installed = False
        orig_autograd_backward = torch.autograd.backward
        if adapter._delta_mode and adapter._group is not None:
            def _retain_graph_backward(*a, **kw):
                # Force retain_graph=True for the PRIMARY backward so
                # the flush's secondary backward can re-enter the
                # producer's subgraph. The flush explicitly passes
                # retain_graph=False itself, which wins here since
                # kw["retain_graph"] will already be set.
                kw.setdefault("retain_graph", True)
                return orig_autograd_backward(*a, **kw)
            torch.autograd.backward = _retain_graph_backward  # type: ignore[assignment]
            override_installed = True
        try:
            result = orig_bwd(
                bwd_chunk_id, loss=loss, full_backward=full_backward,
                last_backward=last_backward,
            )
            # Drop the retain-graph override BEFORE the flush so the
            # flush's secondary backward actually frees intermediate
            # activations (retain_graph=False path).
            if override_installed:
                torch.autograd.backward = orig_autograd_backward  # type: ignore[assignment]
                override_installed = False
            # Flush the deferred grad P2P. Must happen after
            # ``orig_bwd`` has accumulated the local grads and the
            # Function backwards have populated _grad_accum /
            # _pending_recv on the cache.
            adapter._flush_grad_sendback(bwd_chunk_id)
            return result
        finally:
            if override_installed:
                torch.autograd.backward = orig_autograd_backward  # type: ignore[assignment]
            adapter.on_microbatch_end(bwd_chunk_id)
            _set_mb_index(adapter_key, None)

    stage.forward_one_chunk = patched_fwd
    stage.backward_one_chunk = patched_bwd


# ----- FQN-split injection ------------------------------------------------- #

_ATTN_RES_EXTRA_LAST_STAGE_FQNS = ("final_attn_res_proj", "final_attn_res_norm")


def _inject_attn_res_fqns(model: nn.Module, kwargs: dict) -> None:
    """Extend the PP module-FQN split so AttnRes top-level modules
    (``final_attn_res_proj`` / ``final_attn_res_norm``) survive
    ``pipeline_module_split`` on the last stage.
    """
    if not any(hasattr(model, n) for n in _ATTN_RES_EXTRA_LAST_STAGE_FQNS):
        return
    parallelism = kwargs.get("parallelism")
    if parallelism is None or parallelism.module_fqns_per_model_part is not None:
        return
    model_config = kwargs.get("model_config")
    pp = kwargs["parallel_dims"].pp
    if pp <= 1 or model_config is None or not hasattr(model_config, "layers"):
        return

    import math as _math
    from torch.distributed.pipelining.schedules import PipelineScheduleSingle
    from torchtitan.distributed.pipeline_parallel import (
        generate_llm_fqn_per_model_part,
        get_schedule_class,
    )

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
        stages_per_rank = 1 if issubclass(schedule_class, PipelineScheduleSingle) else 2
        num_virtual_stages = pp * stages_per_rank

    fqns = generate_llm_fqn_per_model_part(
        num_virtual_stages, num_layers, input_weight, output_weight
    )
    extras = [n for n in _ATTN_RES_EXTRA_LAST_STAGE_FQNS if hasattr(model, n)]
    fqns[-1].extend(extras)
    parallelism.module_fqns_per_model_part = fqns


# ----- Custom pipelining_fn ------------------------------------------------ #

def _build_stage_to_rank(stages) -> dict[int, int]:
    """Map ``stage_id -> rank`` from live ``PipelineStage`` objects."""
    mapping: dict[int, int] = {}
    for stage in stages:
        mapping[stage.stage_index] = stage.group_rank
    return mapping


def pipeline_llm_with_cache_adapter(model: nn.Module, **kwargs):
    """Custom ``pipelining_fn`` for AttnRes.

    Delegates to core ``pipeline_llm``; when
    ``TORCHTITAN_ATTNRES_CACHE=1`` and the schedule is Interleaved1F1B,
    wraps each stage's submod with :class:`CrossStageCacheAdapter`.
    Returns the same 4-tuple as core.
    """
    # Deferred import so this module stays importable without torchtitan's
    # distributed wiring (CPU unit tests).
    from torchtitan.distributed.pipeline_parallel import pipeline_llm

    _inject_attn_res_fqns(model, kwargs)
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )
    passthrough = (pp_schedule, model_parts, has_first_stage, has_last_stage)

    if not adapter_enabled():
        return passthrough

    # Schedule guard: Interleaved1F1B is the only delta protocol we
    # characterize; anything else falls back with a warn.
    if _INTERLEAVED_1F1B_CLASS is None or not isinstance(
        pp_schedule, _INTERLEAVED_1F1B_CLASS
    ):
        warnings.warn(
            "cross-stage caching currently supports only Interleaved1F1B; "
            "running without optimization"
        )
        return passthrough

    stages = list(_iter_schedule_stages(pp_schedule))
    # ``stages`` is this rank's LOCAL stages; ``stage.stage_index`` is
    # GLOBAL. Total virtual stages = parallel_dims.pp * V.
    parallel_dims = kwargs.get("parallel_dims")
    pp_size = parallel_dims.pp if parallel_dims is not None else len(stages)
    num_stages = pp_size * len(stages)
    stage_to_rank = {s: s % pp_size for s in range(num_stages)}

    # Find the AttnRes config on any stage's inner model.
    inner0 = getattr(stages[0], "submod", None)
    inner0_cfg = getattr(inner0, "config", None)
    attn_res_config = getattr(inner0_cfg, "attn_res", None) if inner0_cfg else None
    if attn_res_config is None or not getattr(attn_res_config, "enabled", False):
        warnings.warn(
            "Could not locate an enabled AttnRes config on stage 0; "
            "cross-stage caching falls back to naive PP."
        )
        return passthrough

    num_blocks = attn_res_config.num_blocks
    # ``inner0_cfg.layers`` is this stage's slice; total layers live on
    # the outer ``model_config``.
    n_layers_total = len(inner0_cfg.layers) * num_stages
    model_config = kwargs.get("model_config")
    if model_config is not None and hasattr(model_config, "layers"):
        n_layers_total = len(model_config.layers)
    layers_per_block = n_layers_total // num_blocks

    try:
        layout_tables = _infer_block_layout_tables_from_stages(
            stages, pp_size=pp_size, num_blocks=num_blocks,
            n_layers=n_layers_total, layers_per_block=layers_per_block,
        )
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(
            f"Failed to build static block-layout tables ({e!r}); "
            "cross-stage caching falls back to naive PP."
        )
        return passthrough

    for i, stage in enumerate(stages):
        adapter = CrossStageCacheAdapter(
            stage.submod,
            stage_id=stage.stage_index,
            num_stages=num_stages,
            group=getattr(stage, "group", None),
            stage_to_rank=stage_to_rank,
            pp_rank=getattr(stage, "group_rank", None),
            layout_tables=layout_tables,
        )
        stage.submod = adapter
        _install_mb_index_patch(stage, adapter)
        # Keep model_parts in sync for optimizer/compile paths.
        if i < len(model_parts):
            model_parts[i] = adapter

    return pp_schedule, model_parts, has_first_stage, has_last_stage
