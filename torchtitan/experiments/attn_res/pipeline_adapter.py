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
by concatenating its cached prefix with the incoming delta.

Per-block backward grads ride the normal autograd + PP SEND_B path:

* Blocks this rank received from an earlier PP hop are slices of a
  ``recv_delta_tensor``; their grad flows back through that tensor
  and PP's built-in ``SEND_B`` ships it to the previous rank.
* Blocks this rank committed in an earlier virtual stage are wrapped
  in :class:`_LocalCacheAugment` at producer emission and
  :class:`_LocalCacheCapture` at consumer read. The Capture severs
  the consumer->producer autograd link (so the consumer's backward
  never traverses and frees the producer's forward graph) and
  deposits the grad in a rank-local slot; the Augment pops the slot
  and sums the captured grad into the producer's incoming grad when
  the producer's own backward runs. Both classes are pure local
  Python + a dict -- zero NCCL, zero cross-rank state.

The grad thus walks backwards hop-by-hop along the same PP stage chain
that forward uses. No custom P2P, no deadlock risk (PP owns all NCCL),
each stage's forward graph is traversed exactly once per mb so peak
memory is the naive-PP baseline.

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
    _infer_block_layout_tables_from_stages,
    BlockLayoutTables,
)


def adapter_enabled() -> bool:
    """Env-flag gate: adapter is opt-in until we trust it."""
    return os.environ.get("TORCHTITAN_ATTNRES_CACHE") == "1"


# ----- Rank-shared cache across virtual stages ----------------------------- #

class RankLocalCache:
    """Per-rank, per-microbatch forward-block cache shared across VP stages.

    Every adapter on the same physical rank reads/writes the SAME cache
    (Kimi §4.1 invariant). Holds only forward-path state: the cached
    block tensors (autograd-live against their original source) and
    producer metadata for layout bookkeeping.

    Grad-send-back has no state here: backward rides the autograd graph
    via PP's built-in SEND_B, so there's nothing for this cache to
    track on the backward path.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, list[torch.Tensor]] = {}
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        # Every backward marks its mb here so the step-end drop sweep
        # on the last virtual stage knows which mbs to evict.
        self._seen_mbs: set[int] = set()
        # Captured grads for the local-only _LocalCacheAugment/Capture
        # dance. Keyed by (mb_index, producer_stage_id, block_idx). A
        # consumer-side Capture.backward accumulates grad here; the
        # producer-side Augment.backward pops and sums the captured
        # grad into its incoming grad when stage R's own backward runs.
        self._captured_grads: dict[tuple[int, int, int], torch.Tensor] = {}

    def append(
        self, mb_index: int, block: torch.Tensor, meta: tuple[int, int, int],
    ) -> None:
        self._blocks.setdefault(mb_index, []).append(block)
        self._producer_meta.setdefault(mb_index, []).append(meta)

    def get_blocks(self, mb_index: int) -> list[torch.Tensor]:
        return self._blocks.get(mb_index, [])

    def get_meta(self, mb_index: int) -> list[tuple[int, int, int]]:
        return self._producer_meta.get(mb_index, [])

    def put_forward(
        self, mb_index: int, blocks: list[torch.Tensor],
        producer_meta: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Back-compat shim used by unit tests: overwrite the per-mb list."""
        self._blocks[mb_index] = list(blocks)
        if producer_meta is not None:
            self._producer_meta[mb_index] = list(producer_meta)

    def drop(self, mb_index: int) -> None:
        self._blocks.pop(mb_index, None)
        self._producer_meta.pop(mb_index, None)
        self._seen_mbs.discard(mb_index)
        # Drop any leftover captured-grad slots for this mb (defensive).
        for key in list(self._captured_grads.keys()):
            if key[0] == mb_index:
                self._captured_grads.pop(key, None)

    # ----- captured-grad slot helpers -------------------------------- #

    def capture_grad(
        self, key: tuple[int, int, int], grad: torch.Tensor,
    ) -> None:
        """Accumulate (sum) ``grad`` into the captured-grad slot at
        ``key``. Multiple consumer-side Captures for the same producer
        block (V>2, one cached block read by >1 later virtual stage on
        the same rank) sum into the same slot.
        """
        prior = self._captured_grads.get(key)
        if prior is None:
            # detach()+clone() would lose autograd on the grad itself;
            # the grad tensor is just a regular tensor at this point
            # (it's a backward-produced value, not leaf), so storing a
            # reference is fine. We deliberately do NOT clone so that
            # in the common V=2 case there's zero extra memory.
            self._captured_grads[key] = grad
        else:
            self._captured_grads[key] = prior + grad

    def pop_grad(
        self, key: tuple[int, int, int],
    ) -> torch.Tensor | None:
        """Return-and-clear the captured-grad slot at ``key``. Returns
        ``None`` if the slot is empty (the producer's cached block was
        never consumed on this rank during this mb's backward window).
        """
        return self._captured_grads.pop(key, None)

    def has_captured_for_mb(self, mb_index: int) -> bool:
        """True iff any captured-grad slot for ``mb_index`` survives.
        Called by the mb-end assertion as a lingering-bug canary.
        """
        return any(k[0] == mb_index for k in self._captured_grads)


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


# ----- Local-only autograd Functions for own-rank cached commits ----------- #
#
# These two Functions together replace the prior process-global
# `retain_graph=True` monkey-patch. Both are LOCAL to a single rank: no
# NCCL, no cross-rank side effects, no collective ops. Their only state
# is a Python dict slot on the rank-local :class:`RankLocalCache`, keyed
# by ``(mb_index, producer_stage_id, block_idx_within_producer)``.
#
# They only wrap ONE specific autograd liability: a block that stage R
# commits on this rank during an earlier virtual v and that a LATER
# virtual stage on THE SAME rank reads back from the shared cache.
# Without any intervention, that later stage's backward traverses into
# stage R's forward graph via torch.cat's grad path and FREES IT;
# stage R's own backward later (from PP SEND_B on the outgoing delta)
# tries to traverse the same graph and dies with "backward through the
# graph a second time".
#
# _LocalCacheCapture (applied at consumer read time) severs the
# consumer->producer graph link by returning None for its tensor
# input's grad — the producer's graph is never traversed by the
# consumer's backward, so it's not freed early. It deposits the grad
# that WOULD have flowed into the producer into a slot on the rank
# cache.
#
# _LocalCacheAugment (applied at producer emission, BEFORE caching)
# sits on the producer's own autograd path. When stage R's own
# backward finally runs, any incoming grad at the augment is summed
# with the captured slot before it flows into stage R's forward graph
# and ultimately into R's wrapped params. Net effect: params get the
# exact same total grad they would get from naive autograd, but each
# stage's graph is traversed exactly once.
#
# Recv-originated cached blocks (sliced from a prior recv_delta tensor)
# are NOT wrapped: their grad already flows back through PP's built-in
# SEND_B to the producer rank via the recv-delta autograd chain, and
# wrapping them would strand that cross-rank grad channel.


class _LocalCacheAugment(torch.autograd.Function):
    """Identity forward; backward sums (incoming_grad + captured_grad).

    ``slot_key`` is ``(mb_index, producer_stage_id, block_idx)``. The
    rank_cache reference is passed at call time so the Function stays
    process-global / stateless and the cache stays per-rank.
    """

    @staticmethod
    def forward(ctx, block_tensor, slot_key, rank_cache):  # type: ignore[override]
        ctx.slot_key = slot_key
        ctx.rank_cache = rank_cache
        return block_tensor

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        captured = ctx.rank_cache.pop_grad(ctx.slot_key)
        if captured is None:
            combined = grad_out
        else:
            combined = grad_out + captured
        # slot_key and rank_cache are non-Tensor inputs -> None grads.
        return combined, None, None


class _LocalCacheCapture(torch.autograd.Function):
    """Identity forward; backward deposits grad in the slot and STOPS.

    Returning ``None`` for the tensor input tells autograd NOT to
    propagate grad further upstream. This is exactly what prevents the
    consumer's backward from traversing the producer stage's forward
    graph and freeing it before stage R's own backward runs.
    """

    @staticmethod
    def forward(ctx, block_tensor, slot_key, rank_cache):  # type: ignore[override]
        ctx.slot_key = slot_key
        ctx.rank_cache = rank_cache
        return block_tensor

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        # Accumulate (sum) into the slot. V>2 / multiple later virtual
        # stages on this rank reading the same producer block will each
        # fire a Capture.backward; the cache's capture_grad sums them.
        ctx.rank_cache.capture_grad(ctx.slot_key, grad_out)
        # None for tensor input: autograd stops traversing here.
        return None, None, None


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
    delta, rebuilds the full block stack in block-index order, and lets
    backward flow through the autograd graph. Cached-prefix blocks are
    handled two ways depending on who committed them:

    * **Different rank** (producer_rank != self.pp_rank) → cached block
      is a slice of an older ``recv_delta_tensor``. Passed through
      unwrapped; its grad flows back via that tensor and PP's built-in
      ``SEND_B`` drains it to the producer rank.
    * **Same rank** (producer_rank == self.pp_rank) → cached block
      came from an earlier virtual stage on this rank. Wrapped in
      :class:`_LocalCacheCapture` at read time, severing the
      consumer->producer autograd link (so the consumer's backward
      does not traverse the producer's forward graph and free it
      early) and depositing the grad in a rank-local slot. The
      matching producer-side :class:`_LocalCacheAugment` wraps the
      block at emission time and re-injects the captured grad during
      the PRODUCER's own backward pass.

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
        """Interleaved1F1B delta forward (spec §4.1).

        Cached-prefix blocks whose producer is on a DIFFERENT rank are
        passed through unwrapped: their autograd graph already goes
        back to the original ``recv_delta_tensor`` and PP's built-in
        ``SEND_B`` drains it to the producer rank. Cached-prefix
        blocks whose producer is ON THIS RANK (earlier virtual stage)
        are wrapped in :class:`_LocalCacheCapture` at read time,
        severing the consumer->producer autograd link and depositing
        their grad in a rank-local captured-grad slot for the matching
        producer-side :class:`_LocalCacheAugment` to re-inject during
        the producer's own backward pass.
        """
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

        # Pull earlier cached blocks out of the rank cache. Recv-
        # originated entries (``producer_rank != self.pp_rank``) stay
        # unwrapped: their grad flows back via the original
        # ``recv_delta_tensor`` they were sliced from, which PP's
        # built-in ``SEND_B`` drains back to the producer rank.
        # Own-rank cached commits (producer_rank == self.pp_rank) get
        # wrapped in :class:`_LocalCacheCapture`; that severs the
        # consumer->producer autograd link so the producer's forward
        # graph is not freed by THIS stage's backward. The grad that
        # would have flowed upstream is deposited in the rank cache's
        # captured-grad slot, to be summed into the producer stage's
        # incoming grad by the matching :class:`_LocalCacheAugment`
        # when the producer's own backward runs.
        earlier_blocks_raw = list(self._cache.get_blocks(mb))
        earlier_meta = list(self._cache.get_meta(mb))
        cached_indices = [
            layout.commits_at(meta[1])[meta[2]] for meta in earlier_meta
        ]
        earlier_blocks: list[torch.Tensor] = []
        for blk, meta in zip(earlier_blocks_raw, earlier_meta):
            producer_rank, producer_stage, block_idx_in_producer = meta
            if producer_rank == self.pp_rank:
                slot_key = (mb, producer_stage, block_idx_in_producer)
                earlier_blocks.append(
                    _LocalCacheCapture.apply(blk, slot_key, self._cache)
                )
            else:
                earlier_blocks.append(blk)

        # Rebuild the full blocks tensor in block-index order.
        pairs = list(zip(cached_indices, earlier_blocks)) + list(
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
        committed blocks to the shared rank cache, then stack the
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
        # them; producer metadata comes from the static layout. Slices
        # of ``prev_recv_tensor`` stay autograd-live against it, so
        # PP's SEND_B on backward will drain their grads upstream.
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

        # Append own commits. Each block is wrapped in
        # :class:`_LocalCacheAugment` BEFORE caching and before being
        # used to build the outgoing delta; the wrapped (identity-
        # forward) tensor is the same object in both places, so when
        # this stage's own backward runs on the delta path the augment
        # sees the incoming grad plus any captured grad deposited by a
        # later same-rank virtual stage's :class:`_LocalCacheCapture`.
        new_blocks_list = unstack_blocks(new_blocks_tensor)
        wrapped_new_blocks: list[torch.Tensor] = []
        for local_idx, blk in enumerate(new_blocks_list):
            slot_key = (mb, self.stage_id, local_idx)
            wrapped = _LocalCacheAugment.apply(blk, slot_key, self._cache)
            wrapped_new_blocks.append(wrapped)
            self._cache.append(
                mb, wrapped, (self.pp_rank, self.stage_id, local_idx)
            )

        # Build outgoing delta: subset of (cache + new), by canonical bidx.
        # ``cache_by_bidx`` reads from the rank cache directly so
        # relayed (recv-originated) blocks that show up in the outgoing
        # delta also route grad correctly via their existing autograd
        # link to ``prev_recv_tensor``.
        out_indices = layout.delta_to_send(self.stage_id)
        cache_by_bidx = {
            layout.commits_at(meta[1])[meta[2]]: blk
            for meta, blk in zip(
                self._cache.get_meta(mb), self._cache.get_blocks(mb)
            )
        }
        new_by_bidx = {
            my_commits[i]: wrapped_new_blocks[i]
            for i in range(len(wrapped_new_blocks))
        }
        send_pieces: list[torch.Tensor] = []
        for bidx in out_indices:
            if bidx in new_by_bidx:
                send_pieces.append(new_by_bidx[bidx])
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

    def _drop_all_seen_and_clear(self) -> None:
        """Drop every mb the cache saw during the step and clear the
        seen-set. Called by the step-end monkey-patch after every
        adapter on this rank has finished backward. Honors the VP
        drop-guard: only the LAST virtual stage on the rank evicts;
        earlier virtual stages no-op so the shared cache survives for
        them.
        """
        if self._delta_mode:
            pp_size = self._layout.P if self._layout is not None else self.num_stages
            if self.stage_id + pp_size < self.num_stages:
                return
        seen = list(self._cache._seen_mbs)
        for mb_index in seen:
            self._cache.drop(mb_index)
        # Defensive: ensure the seen-set is clear even if drop() didn't
        # remove every entry.
        self._cache._seen_mbs.clear()

    def on_microbatch_end(self, mb_index: int) -> None:
        """Mark ``mb_index`` as seen on this rank so the step-end sweep
        drops it. Actual eviction is deferred to ``pp_schedule.step``
        return; see :func:`_install_step_drop_patch`.

        In delta mode, this is also the moment to assert that every
        :class:`_LocalCacheCapture` deposit for this mb has been drained
        by a matching :class:`_LocalCacheAugment` -- a surviving slot
        would mean a producer's backward never ran, which is a bug.
        Interleaved1F1B runs backward in reverse virtual-stage order
        on each rank, so the EARLIEST virtual stage on this rank is
        the last to call on_microbatch_end for a given mb. That is
        the only point at which every producer-side Augment has had a
        chance to drain its slot, so we guard the assertion to fire
        there only (``stage_id < pp_size`` == "this rank's earliest
        virtual stage").
        """
        self._cache._seen_mbs.add(mb_index)
        if self._delta_mode and self._layout is not None:
            pp_size = self._layout.P
            # Earliest virtual stage on this rank: stage_id < pp_size.
            # Its backward fires LAST among the rank's virtual stages
            # for this mb, so by the time we reach here every slot
            # for this mb should have been popped by an Augment.
            if self.stage_id < pp_size:
                assert not self._cache.has_captured_for_mb(mb_index), (
                    f"Captured grad slot for mb {mb_index} survived past "
                    f"stage {self.stage_id}'s backward on rank {self.pp_rank}; "
                    "producer-side _LocalCacheAugment never fired. "
                    "This indicates a producer forward graph was never "
                    "backward-traversed for this mb."
                )

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

    Backward is a plain call: no retain_graph override, no custom
    transport. The cached-prefix autograd graph + PP's built-in SEND_B
    route all cross-rank grads; the adapter only needs the mb index
    threaded through forward + on_microbatch_end.
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
        # Plain backward pass. The double-backward risk on own-rank
        # cached commits is now handled structurally by
        # :class:`_LocalCacheAugment` / :class:`_LocalCacheCapture`:
        # the consumer-side Capture severs the consumer->producer
        # autograd link (so the producer's graph is NOT traversed or
        # freed by this stage's backward), and the producer-side
        # Augment sums the captured grad into the producer's own
        # incoming grad when THE PRODUCER's backward runs. Each
        # stage's forward graph is thus traversed exactly once per mb,
        # which is the naive-PP baseline.
        _set_mb_index(adapter_key, bwd_chunk_id)
        try:
            return orig_bwd(
                bwd_chunk_id, loss=loss, full_backward=full_backward,
                last_backward=last_backward,
            )
        finally:
            # Mark the mb as seen so the step-end drop sweep evicts it.
            # We don't drop here: the shared rank cache is still live
            # for peers / later virtual stages.
            adapter.on_microbatch_end(bwd_chunk_id)
            _set_mb_index(adapter_key, None)

    stage.forward_one_chunk = patched_fwd
    stage.backward_one_chunk = patched_bwd


def _install_step_drop_patch(
    pp_schedule: _PipelineSchedule, adapters: list[CrossStageCacheAdapter]
) -> None:
    """Wrap ``pp_schedule.step`` so every registered adapter on this
    rank evicts its seen mbs from the shared cache EXACTLY ONCE after
    ``orig_step`` returns. The VP drop-guard inside
    :meth:`_drop_all_seen_and_clear` ensures only the last virtual
    stage on the rank actually frees memory.
    """
    orig_step = pp_schedule.step

    def patched_step(*args, **kwargs):
        try:
            return orig_step(*args, **kwargs)
        finally:
            for adapter in adapters:
                try:
                    adapter._drop_all_seen_and_clear()
                except Exception:
                    # Swallow per-adapter drop failures so one poisoned
                    # adapter doesn't prevent the others from clearing.
                    pass

    pp_schedule.step = patched_step  # type: ignore[method-assign]


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

    installed_adapters: list[CrossStageCacheAdapter] = []
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
        installed_adapters.append(adapter)
        # Keep model_parts in sync for optimizer/compile paths.
        if i < len(model_parts):
            model_parts[i] = adapter

    # Install step-end drop sweep AFTER every per-mb patch. The VP
    # drop-guard in _drop_all_seen_and_clear ensures only the last
    # virtual stage on the rank frees memory.
    _install_step_drop_patch(pp_schedule, installed_adapters)

    return pp_schedule, model_parts, has_first_stage, has_last_stage
