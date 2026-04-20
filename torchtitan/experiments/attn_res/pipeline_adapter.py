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
    """

    def __init__(self) -> None:
        self._forward: dict[int, list[torch.Tensor]] = {}
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        self._grad_accum: dict[int, list[torch.Tensor | None]] = defaultdict(list)

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

    def add_grad(self, mb_index: int, block_idx: int, grad: torch.Tensor) -> None:
        slot = self._grad_accum[mb_index]
        while len(slot) <= block_idx:
            slot.append(None)
        contrib = grad.detach()
        slot[block_idx] = contrib.clone() if slot[block_idx] is None else slot[block_idx] + contrib

    def pop_grads(self, mb_index: int) -> list[torch.Tensor | None]:
        return self._grad_accum.pop(mb_index, [])


# ----- Rank-shared cache across virtual stages ----------------------------- #

class RankLocalCache:
    """Per-rank, per-microbatch accumulator shared across virtual stages.

    Every adapter on the same physical rank reads/writes the SAME cache
    (Kimi §4.1 invariant). Canonical API: ``append`` / ``get_blocks`` /
    ``get_meta``. ``put_forward`` / ``get_forward`` / ``get_producer_meta``
    are back-compat shims.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, list[torch.Tensor]] = {}
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        self._grad_accum: dict[int, list[torch.Tensor | None]] = defaultdict(list)

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

    def add_grad(self, mb_index: int, block_idx: int, grad: torch.Tensor) -> None:
        slot = self._grad_accum[mb_index]
        while len(slot) <= block_idx:
            slot.append(None)
        contrib = grad.detach()
        slot[block_idx] = contrib.clone() if slot[block_idx] is None else slot[block_idx] + contrib

    def pop_grads(self, mb_index: int) -> list[torch.Tensor | None]:
        return self._grad_accum.pop(mb_index, [])


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
    """Consumer-side identity forward, P2P-send-per-block backward.

    Backward accumulates per-block grads into the cache and isend-s each
    one to the block's original producer rank. Returning ``None`` per
    block input stops the local autograd graph here.
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
        recorded_grads: list[torch.Tensor] = []
        for i, g in enumerate(grad_outputs):
            if g is None:
                g = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
            recorded_grads.append(g)
            if ctx.cache is not None:
                ctx.cache.add_grad(ctx.mb_index, i, g)

        group = ctx.group
        if group is not None and dist.is_available() and dist.is_initialized():
            reqs = [
                dist.isend(
                    g.contiguous(),
                    dst=ctx.producer_ranks[i], group=group, tag=ctx.tags[i],
                )
                for i, g in enumerate(recorded_grads)
            ]
            for r in reqs:
                r.wait()
        return (None, None, None, None, None) + tuple(None for _ in grad_outputs)


class _RecvBlockGradsFromConsumers(torch.autograd.Function):
    """Producer-side identity forward, sum-of-irecvs backward.

    Backward irecvs one grad tensor per (consumer, block) and sums them
    into the grad the local graph produced.
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
        N = ctx.shape[0]
        per_block_shape = ctx.shape[1:]
        extra = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        pending = []
        for src in ctx.consumer_ranks:
            for b in range(N):
                buf = torch.empty(per_block_shape, dtype=ctx.dtype, device=ctx.device)
                req = dist.irecv(buf, src=src, group=group, tag=ctx.tag_base + b)
                pending.append((req, buf, b))
        for req, buf, b in pending:
            req.wait()
            extra[b] = extra[b] + buf
        return None, None, None, grad_output + extra


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
            consumer_ranks, self._group, tag_base, blocks_tensor
        )

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
        try:
            return orig_bwd(
                bwd_chunk_id, loss=loss, full_backward=full_backward,
                last_backward=last_backward,
            )
        finally:
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
