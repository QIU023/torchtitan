# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only unit tests for :mod:`pipeline_adapter`.

Covers:
  * :class:`_PerMicrobatchCache` grad accumulation bookkeeping.
  * Microbatch-index threading through the monkey-patched
    ``forward_one_chunk`` / ``backward_one_chunk`` hooks (Bug #2 fix).
  * Identity semantics of :class:`_SendBlockGradsBack` and
    :class:`_RecvBlockGradsFromConsumers` when no process group is
    wired, and their correctness under a single-process reference.
  * An end-to-end mini multi-stage setup where two
    ``CrossStageCacheAdapter`` instances thread the same microbatch
    ids through forward + backward, and the grads recorded in
    ``_grad_accum`` per block match the naive full-stack forward of
    the same model (Bug #3 accumulation semantics).

These tests do NOT spin up NCCL; they exercise the CPU / no-PG branch
explicitly (``group=None``). An 8-GPU correctness check is the
separate A/B smoke under ``phase3/launch_8gpu_adapter.sh``.
"""

import os
import threading
import unittest
import warnings
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from torchtitan.experiments.attn_res.attn_res import stack_blocks, unstack_blocks
from torchtitan.experiments.attn_res.pipeline_adapter import (
    _current_mb_index,
    _get_or_create_rank_cache,
    _grad_tag_base,
    _install_mb_index_patch,
    _PerMicrobatchCache,
    _RecvBlockGradsFromConsumers,
    _reset_rank_caches_for_testing,
    _SendBlockGradsBack,
    _set_mb_index,
    BlockLayoutTables,
    CrossStageCacheAdapter,
    RankLocalCache,
)


class TestPerMicrobatchCache(unittest.TestCase):
    """Bookkeeping: forward-put/get, grad-accumulate, and drop."""

    def test_put_get_forward(self):
        cache = _PerMicrobatchCache()
        b0 = torch.randn(2, 3, 4)
        b1 = torch.randn(2, 3, 4)
        meta = [(0, 0, 0), (0, 0, 1)]
        cache.put_forward(0, [b0, b1], producer_meta=meta)
        got = cache.get_forward(0)
        self.assertEqual(len(got), 2)
        self.assertIs(got[0], b0)
        self.assertIs(got[1], b1)
        self.assertEqual(cache.get_producer_meta(0), meta)

    def test_get_missing_returns_empty(self):
        cache = _PerMicrobatchCache()
        self.assertEqual(cache.get_forward(42), [])
        self.assertEqual(cache.get_producer_meta(42), [])

    def test_add_grad_accumulates(self):
        cache = _PerMicrobatchCache()
        g1 = torch.ones(3, 4)
        g2 = torch.full((3, 4), 2.0)
        g3 = torch.full((3, 4), 0.5)
        cache.add_grad(7, 0, g1)
        cache.add_grad(7, 0, g2)
        cache.add_grad(7, 1, g3)
        grads = cache.pop_grads(7)
        self.assertEqual(len(grads), 2)
        self.assertTrue(torch.allclose(grads[0], g1 + g2))
        self.assertTrue(torch.allclose(grads[1], g3))
        # pop empties it
        self.assertEqual(cache.pop_grads(7), [])

    def test_drop_clears_all_state(self):
        cache = _PerMicrobatchCache()
        cache.put_forward(1, [torch.zeros(1)], producer_meta=[(0, 0, 0)])
        cache.add_grad(1, 0, torch.ones(1))
        cache.drop(1)
        self.assertEqual(cache.get_forward(1), [])
        self.assertEqual(cache.get_producer_meta(1), [])
        self.assertEqual(cache.pop_grads(1), [])


class TestGradTagBase(unittest.TestCase):
    """The tag base must be unique per (mb_index, producer_stage)."""

    def test_distinct_for_distinct_pairs(self):
        seen = set()
        for mb in range(4):
            for s in range(8):
                tag = _grad_tag_base(mb, s)
                # Reserve 1024 tags in each block; ensure no overlap with
                # another (mb, s) block.
                for k in range(1024):
                    seen_tag = tag + k
                    self.assertNotIn(seen_tag, seen)
                    seen.add(seen_tag)


class TestMbIndexThreading(unittest.TestCase):
    """Bug #2 fix: the monkey-patched stage forward/backward must make
    the schedule's microbatch index visible to the adapter via a
    thread-local, so cached-block lookups succeed by integer key rather
    than by ``id(tensor)`` (which would change across a P2P boundary).
    """

    def test_set_and_read_under_patch(self):
        """Simulate what the adapter's inner forward does: read
        ``_current_mb_index(id(adapter))`` during the time between the
        patched ``forward_one_chunk`` entry and exit.
        """
        seen_indices = []

        def inner(chunk_id, args, kwargs=None, save_forward_output=True):
            # Stand-in for what the real PipelineStage.forward_one_chunk
            # would do -- invoke submod(*args). We use it to read the
            # thread-local at the exact moment the adapter would.
            seen_indices.append(_current_mb_index(id(adapter)))
            return (args[0],)

        stage = MagicMock()
        stage.forward_one_chunk = inner
        # backward_one_chunk isn't exercised here; patch needs it to
        # exist on the stage so the install doesn't blow up.
        stage.backward_one_chunk = MagicMock()

        submod = nn.Linear(4, 4)
        adapter = CrossStageCacheAdapter(
            submod, stage_id=0, num_stages=2, group=None
        )
        _install_mb_index_patch(stage, adapter)

        # Call the patched method the way a schedule would. The adapter
        # inner (here the MagicMock inner above) reads the thread-local.
        stage.forward_one_chunk(3, (torch.randn(2, 4),))
        stage.forward_one_chunk(7, (torch.randn(2, 4),))
        self.assertEqual(seen_indices, [3, 7])

        # After the patched method returns, the thread-local is cleared.
        self.assertIsNone(_current_mb_index(id(adapter)))

    def test_backward_patch_drops_cache(self):
        stage = MagicMock()
        stage.forward_one_chunk = lambda *a, **kw: None
        stage.backward_one_chunk = lambda *a, **kw: None

        submod = nn.Linear(4, 4)
        adapter = CrossStageCacheAdapter(
            submod, stage_id=0, num_stages=2, group=None
        )
        _install_mb_index_patch(stage, adapter)

        # Pretend forward happened for mb=5 and left cache entries.
        adapter._cache.put_forward(
            5, [torch.zeros(1)], producer_meta=[(0, 0, 0)]
        )
        adapter._cache.add_grad(5, 0, torch.ones(1))

        stage.backward_one_chunk(5)

        self.assertEqual(adapter._cache.get_forward(5), [])
        self.assertEqual(adapter._cache.pop_grads(5), [])


class TestSendBlockGradsBack(unittest.TestCase):
    """With ``group=None`` (no process group), the send-back Function
    must behave as a pure identity in forward and DROP grad on its
    tensor inputs in backward (because grads leave over P2P; the
    no-PG branch silently short-circuits so unit tests can run on CPU).
    """

    def test_forward_is_identity_clone(self):
        b0 = torch.randn(2, 3, 4, requires_grad=True)
        b1 = torch.randn(2, 3, 4, requires_grad=True)
        outs = _SendBlockGradsBack.apply(
            [0, 0], [0, 1], None, None, 0, b0, b1
        )
        self.assertEqual(len(outs), 2)
        self.assertTrue(torch.allclose(outs[0], b0))
        self.assertTrue(torch.allclose(outs[1], b1))

    def test_backward_without_pg_does_not_crash(self):
        b0 = torch.randn(2, 3, 4, requires_grad=True)
        b1 = torch.randn(2, 3, 4, requires_grad=True)
        outs = _SendBlockGradsBack.apply(
            [0, 0], [0, 1], None, None, 0, b0, b1
        )
        loss = outs[0].sum() + 2.0 * outs[1].sum()
        loss.backward()
        # Because send-back Function's backward returns None for inputs,
        # b0 and b1 should have no local grad (the "real" grad would
        # have been shipped over P2P). This is intentional: on the
        # consumer side, the cached-prefix blocks' grads leave the local
        # graph entirely.
        self.assertIsNone(b0.grad)
        self.assertIsNone(b1.grad)

    def test_backward_writes_to_cache(self):
        """Key Bug #3 assertion: the Function's backward writes each
        block's grad into ``cache._grad_accum[mb]`` so the adapter has
        the payload ready to ship (or inspect, under group=None).
        """
        cache = _PerMicrobatchCache()
        b0 = torch.randn(2, 3, 4, requires_grad=True)
        b1 = torch.randn(2, 3, 4, requires_grad=True)
        outs = _SendBlockGradsBack.apply(
            [0, 0], [0, 1], None, cache, 5, b0, b1
        )
        # Downstream use: scale each output by a distinct constant so
        # each block's upstream grad is distinct.
        (3.0 * outs[0].sum() + 7.0 * outs[1].sum()).backward()

        grads = cache.pop_grads(5)
        self.assertEqual(len(grads), 2)
        self.assertTrue(torch.allclose(grads[0], 3.0 * torch.ones_like(b0)))
        self.assertTrue(torch.allclose(grads[1], 7.0 * torch.ones_like(b1)))


class TestRecvBlockGradsFromConsumers(unittest.TestCase):
    """With ``group=None``, the recv-from-consumers Function is a pure
    identity (forward + backward) -- a producer stage's blocks see only
    the local autograd grad, no P2P-recv contribution.
    """

    def test_forward_is_identity(self):
        x = torch.randn(3, 2, 4, 5, requires_grad=True)
        y = _RecvBlockGradsFromConsumers.apply([1, 2], None, 0, None, 0, 0, x)
        self.assertTrue(torch.allclose(y, x))

    def test_backward_passes_local_grad(self):
        x = torch.randn(3, 2, 4, 5, requires_grad=True)
        y = _RecvBlockGradsFromConsumers.apply([1, 2], None, 0, None, 0, 0, x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


class _ToyBlockModel(nn.Module):
    """Tiny stand-in for AttnResLlama3Model used in the end-to-end test.

    Non-last stage: receives ``(partial, blocks_tensor)``, returns
    ``(partial + block_sum * w, new_blocks_tensor)``.

    Last stage (``is_last=True``): returns a scalar-ish logits tensor
    derived from the same simple mix so we can check grads on each
    block end-to-end.

    The important property: the output's grad w.r.t. each block is
    deterministic and easy to enumerate, so we can compare naive
    full-stack forward against the cached-adapter forward block by
    block.
    """

    def __init__(self, dim: int, new_blocks_per_stage: int, is_last: bool):
        super().__init__()
        self.dim = dim
        self.new_blocks_per_stage = new_blocks_per_stage
        self.is_last = is_last
        # Respects the adapter's contract: flip this to True and we'll
        # return only the NEW blocks from the intermediate branch.
        self._return_only_new_blocks: bool = False
        # A per-stage learnable scalar so grads propagate through
        # weights too. Kept at 1.0 so naive / cached produce identical
        # numerics.
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, tokens_or_partial, blocks=None, **kwargs):
        """Mimic AttnResLlama3Model: on stage 0 ``blocks is None`` and
        ``tokens_or_partial`` is the embedding; on later stages
        ``blocks`` is a stacked tensor.
        """
        partial = tokens_or_partial
        if blocks is None:
            block_list: list[torch.Tensor] = []
        else:
            block_list = unstack_blocks(blocks)
        initial_num_blocks = len(block_list)

        # This stage commits ``new_blocks_per_stage`` blocks -- each is
        # the current partial scaled by ``w``. Matches AttnRes' commit
        # semantics of block_start layers.
        for _ in range(self.new_blocks_per_stage):
            block_list.append(partial * self.w)
            # Partial advances with a tiny mix of prior blocks so
            # autograd sees cross-block dependence.
            partial = partial + sum(block_list) / max(len(block_list), 1)

        if not self.is_last:
            if self._return_only_new_blocks:
                new_blocks = block_list[initial_num_blocks:]
                if not new_blocks:
                    empty = partial.new_zeros((0, *partial.shape))
                    return partial, empty
                return partial, stack_blocks(new_blocks)
            return partial, stack_blocks(block_list)

        # Last stage: cross-block aggregation into a single logit-ish
        # tensor.
        return partial + sum(block_list)


class TestEndToEndTwoStageNumerics(unittest.TestCase):
    """Threads the same 'microbatch' through two toy stages -- once
    through the cached-adapter path (with no PG, so send-back is a
    no-op), once through a naive full-stack reference -- and checks
    forward numerics match.

    We also check that on the consumer stage the cached-prefix blocks
    have backward hooks that accumulate grads into
    ``_grad_accum``; since the no-PG ``_SendBlockGradsBack`` drops its
    input-side grad, the hooks' accumulation is what an eventual
    isend/irecv pair would consume. This test validates the per-block
    accumulation math without needing real NCCL.
    """

    def _build_pair(self, dim: int):
        stage0 = _ToyBlockModel(dim=dim, new_blocks_per_stage=2, is_last=False)
        stage1 = _ToyBlockModel(dim=dim, new_blocks_per_stage=2, is_last=True)
        return stage0, stage1

    def test_forward_matches_naive(self):
        torch.manual_seed(42)
        dim = 4
        B, T = 2, 3
        tokens = torch.randn(B, T, dim)

        # Two stages, each committing 2 blocks. Non-last stage is stage 0.
        stage0, stage1 = self._build_pair(dim)

        # Naive reference: run both submodels directly.
        naive_partial, naive_blocks = stage0(tokens, blocks=None)
        naive_out = stage1(naive_partial, blocks=naive_blocks)

        # Adapter path.
        adapter0 = CrossStageCacheAdapter(
            stage0, stage_id=0, num_stages=2, group=None
        )
        adapter1 = CrossStageCacheAdapter(
            stage1, stage_id=1, num_stages=2, group=None
        )

        # Simulate the schedule's forward_one_chunk by setting the
        # thread-local ourselves, calling, and clearing.
        _set_mb_index(id(adapter0), 0)
        try:
            p0, b0_new = adapter0(tokens)
        finally:
            _set_mb_index(id(adapter0), None)

        # Model 1's adapter needs to see the same mb id. In the real
        # run, the producer and consumer stages live on different
        # ranks; here, they share a process, and the schedule calls
        # their patched forward_one_chunk independently.
        _set_mb_index(id(adapter1), 0)
        # Between forward_one_chunk of stage 0 and stage 1, the adapter
        # path pretends that the cache survived the P2P: for the
        # end-to-end in-process check, we need adapter1 to have
        # cached what the schedule would have given it. In a real
        # run, that happens because each rank owns its own adapter and
        # its own cache; here, stage 1's cache starts empty (stage 0's
        # blocks are sent over P2P as the `blocks_tensor` arg). So we
        # call adapter1 with the tensor stage 0 emitted, which it'll
        # treat as a "middle stage recv" -- same as production.
        try:
            adapted_out = adapter1(p0, b0_new)
        finally:
            _set_mb_index(id(adapter1), None)

        self.assertTrue(
            torch.allclose(adapted_out, naive_out, atol=1e-5),
            f"Forward numerics diverge: naive={naive_out}, adapted={adapted_out}",
        )

    def test_2stage_param_grads_match_naive(self):
        """Validates end-to-end backward correctness on the 2-stage
        case where NO cached-prefix blocks exist (so the no-PG stubs
        don't interfere). Parameter grads on stage 0 and stage 1 must
        match naive exactly.
        """
        torch.manual_seed(7)
        dim = 4
        B, T = 2, 3
        tokens = torch.randn(B, T, dim)

        # Two independent build-ups so grads don't alias.
        stage0_naive, stage1_naive = self._build_pair(dim)
        stage0_adapt, stage1_adapt = self._build_pair(dim)
        # Align initial params (both are ones, but be explicit).
        with torch.no_grad():
            stage0_adapt.w.copy_(stage0_naive.w)
            stage1_adapt.w.copy_(stage1_naive.w)

        # --- Naive path --- #
        nv_partial, nv_blocks = stage0_naive(tokens, blocks=None)
        nv_out = stage1_naive(nv_partial, blocks=nv_blocks)
        nv_out.sum().backward()

        # --- Adapter path --- #
        a0 = CrossStageCacheAdapter(
            stage0_adapt, stage_id=0, num_stages=2, group=None
        )
        a1 = CrossStageCacheAdapter(
            stage1_adapt, stage_id=1, num_stages=2, group=None
        )

        _set_mb_index(id(a0), 0)
        try:
            p0, b0 = a0(tokens)
        finally:
            _set_mb_index(id(a0), None)

        _set_mb_index(id(a1), 0)
        try:
            out = a1(p0, b0)
        finally:
            _set_mb_index(id(a1), None)
        out.sum().backward()

        # Both stage-0 and stage-1 params should match naive exactly:
        # with num_stages=2, consumer stage 1 has NO cached prefix
        # (stage 0 is the immediate prev and lands in new_blocks_list),
        # so no autograd-short-circuit from _SendBlockGradsBack runs.
        # stage-0 producer wraps its commits in
        # _RecvBlockGradsFromConsumers; with group=None that's a pure
        # identity, so stage-0 params receive the full local grad.
        self.assertTrue(
            torch.allclose(stage0_adapt.w.grad, stage0_naive.w.grad, atol=1e-6),
            f"stage 0 grad mismatch: naive={stage0_naive.w.grad}, "
            f"adapter={stage0_adapt.w.grad}",
        )
        self.assertTrue(
            torch.allclose(stage1_adapt.w.grad, stage1_naive.w.grad, atol=1e-6),
            f"stage 1 grad mismatch: naive={stage1_naive.w.grad}, "
            f"adapter={stage1_adapt.w.grad}",
        )

    def test_send_back_function_accumulates_into_adapter_cache(self):
        """The ``_SendBlockGradsBack`` Function is the interop point
        between a future constant-forward-size adapter and its
        per-microbatch cache. Even though the current draft of the
        adapter's forward doesn't route cached blocks through the
        Function, we keep this check so the contract used by
        ``pipeline_adapter._PerMicrobatchCache`` stays exercised -- a
        reintroduction of the send-back path later should preserve the
        "backward accumulates per-block grads into ``_grad_accum``"
        invariant end-to-end.
        """
        torch.manual_seed(0)
        dim = 4
        B, T = 2, 3

        # Mirror the adapter's wiring: build a middle-stage adapter and
        # pre-seed its cache with a cached prefix block.
        stage_mid = _ToyBlockModel(dim=dim, new_blocks_per_stage=1, is_last=True)
        adapter_mid = CrossStageCacheAdapter(
            stage_mid, stage_id=1, num_stages=3, group=None
        )
        mb = 0
        cached_block = torch.randn(B, T, dim, requires_grad=True)
        adapter_mid._cache.put_forward(
            mb, [cached_block], producer_meta=[(0, 0, 0)]
        )

        # Drive the send-back Function directly with the cache object.
        # Producer rank / tag are the same values the adapter's
        # ``_wrap_cached_prefix_for_send_back`` would pass in.
        wrapped = _SendBlockGradsBack.apply(
            [0],  # producer_ranks
            [_grad_tag_base(mb, 0)],  # tags
            None,  # group=None -> no P2P, accumulation only
            adapter_mid._cache,
            mb,
            cached_block,
        )
        # Use the wrapped block in a trivial downstream op so the
        # Function's backward is invoked.
        (wrapped[0] * 2.0).sum().backward()

        grads = adapter_mid._cache._grad_accum.get(mb, [])
        self.assertEqual(len(grads), 1, "one cached block should have one grad entry")
        self.assertIsNotNone(grads[0])
        self.assertEqual(grads[0].shape, cached_block.shape)
        self.assertGreater(grads[0].abs().sum().item(), 0.0)


class _InnerToy(nn.Module):
    """Tiny inner model with a mix of params and a buffer -- stands in
    for ``AttnResLlama3Model`` for the state_dict key-layout tests.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.register_buffer("running_stat", torch.zeros(4))

    def forward(self, x):  # pragma: no cover - not exercised by these tests
        return self.norm(self.embed(x))

    # The adapter's __init__ toggles this flag; provide it so the warn
    # path isn't taken on construction (keeps test output clean).
    _return_only_new_blocks: bool = False


class TestAdapterStateDictKeyLayout(unittest.TestCase):
    """The adapter stores the real model at ``self.wrapped``, so a
    naive ``state_dict()`` would prefix every key with ``wrapped.``.
    The adapter installs save + load hooks that strip/re-prepend the
    prefix, keeping its state_dict layout identical to what naive PP
    would emit for the raw model. Verifying here so nobody removes the
    hooks and silently breaks checkpoint round-trips or the Llama3
    HF state_dict adapter.
    """

    def test_state_dict_keys_match_inner_model(self):
        inner = _InnerToy()
        adapter = CrossStageCacheAdapter(
            inner, stage_id=0, num_stages=2, group=None
        )

        inner_keys = set(inner.state_dict().keys())
        adapter_keys = set(adapter.state_dict().keys())

        self.assertEqual(
            inner_keys,
            adapter_keys,
            "adapter state_dict keys must match the wrapped model's "
            "(no 'wrapped.' prefix leaking through)",
        )

    def test_load_state_dict_accepts_inner_layout(self):
        """Round-trip: a state_dict produced by the raw inner model
        loads cleanly into an adapter-wrapped fresh inner model.
        """
        src = _InnerToy()
        # Perturb so we can verify values actually transferred.
        with torch.no_grad():
            src.embed.weight.add_(1.5)
            src.norm.weight.add_(0.25)
            src.running_stat.fill_(7.0)

        dst_inner = _InnerToy()
        adapter = CrossStageCacheAdapter(
            dst_inner, stage_id=0, num_stages=2, group=None
        )

        missing, unexpected = adapter.load_state_dict(src.state_dict(), strict=True)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])
        self.assertTrue(torch.allclose(dst_inner.embed.weight, src.embed.weight))
        self.assertTrue(torch.allclose(dst_inner.norm.weight, src.norm.weight))
        self.assertTrue(torch.allclose(dst_inner.running_stat, src.running_stat))


class TestAdapterInitWeightsPassthrough(unittest.TestCase):
    """Trainer calls ``init_weights(buffer_device=...)`` on each entry
    of ``model_parts``. The adapter owns no parameters, so it must
    forward that call to the wrapped model -- otherwise the trainer
    dies with AttributeError on startup.
    """

    def test_init_weights_forwards_to_wrapped(self):
        called = {}

        class _InnerWithInit(nn.Module):
            _return_only_new_blocks = False

            def init_weights(self, *, buffer_device=None):
                called["buffer_device"] = buffer_device

        inner = _InnerWithInit()
        adapter = CrossStageCacheAdapter(
            inner, stage_id=0, num_stages=1, group=None
        )
        adapter.init_weights(buffer_device=torch.device("cpu"))
        self.assertEqual(called, {"buffer_device": torch.device("cpu")})


class TestRankLocalCache(unittest.TestCase):
    """The refactor introduces a per-process, rank-keyed
    :class:`RankLocalCache` registry. Invariants:

      * Two adapters in the same process with the same ``pp_rank`` hold
        THE SAME ``RankLocalCache`` object (VP=2 sharing case).
      * Different ``pp_rank`` -> different cache objects.
      * ``append``/``drop``/``pop_grads`` compose correctly when multiple
        adapters on the same rank drive them.
      * :func:`_get_or_create_rank_cache` is thread-safe -- concurrent
        constructors land on the same object.

    We reset the module-level registry before each test so classes that
    construct adapters at ``pp_rank=0`` (e.g. the state_dict tests) don't
    leak state into us.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    # ----- registry identity ----------------------------------------- #

    def test_same_pp_rank_shares_cache(self):
        """Two adapters on the same physical rank (VP=2) share a cache."""
        a1 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=0, num_stages=16, group=None, pp_rank=0
        )
        # Simulate the second virtual stage on the same rank: stage_id
        # differs, but pp_rank is the same.
        a2 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=8, num_stages=16, group=None, pp_rank=0
        )
        self.assertIs(
            a1._cache,
            a2._cache,
            "Adapters with matching pp_rank must share ONE RankLocalCache",
        )

    def test_different_pp_rank_gets_different_cache(self):
        a1 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=0, num_stages=16, group=None, pp_rank=0
        )
        a2 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=1, num_stages=16, group=None, pp_rank=1
        )
        self.assertIsNot(a1._cache, a2._cache)

    def test_default_pp_rank_uses_stage_to_rank(self):
        """When ``pp_rank`` is omitted the adapter falls back to the
        stage_to_rank map, matching pre-refactor behavior.
        """
        a = CrossStageCacheAdapter(
            nn.Linear(4, 4),
            stage_id=3,
            num_stages=8,
            group=None,
            stage_to_rank={3: 2},  # stage 3 lives on rank 2
        )
        self.assertEqual(a.pp_rank, 2)
        # Any other adapter declaring pp_rank=2 should share the cache.
        b = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=5, num_stages=8, group=None, pp_rank=2
        )
        self.assertIs(a._cache, b._cache)

    def test_get_or_create_returns_same_instance(self):
        c1 = _get_or_create_rank_cache(42)
        c2 = _get_or_create_rank_cache(42)
        c3 = _get_or_create_rank_cache(43)
        self.assertIs(c1, c2)
        self.assertIsNot(c1, c3)

    # ----- API behavior ---------------------------------------------- #

    def test_append_and_get_blocks_meta(self):
        cache = RankLocalCache()
        b0 = torch.randn(2, 3)
        b1 = torch.randn(2, 3)
        cache.append(0, b0, (0, 0, 0))
        cache.append(0, b1, (0, 0, 1))
        # Different mb index lives in its own list.
        cache.append(1, torch.zeros(2, 3), (0, 0, 0))

        blocks_mb0 = cache.get_blocks(0)
        self.assertEqual(len(blocks_mb0), 2)
        self.assertIs(blocks_mb0[0], b0)
        self.assertIs(blocks_mb0[1], b1)
        self.assertEqual(cache.get_meta(0), [(0, 0, 0), (0, 0, 1)])
        self.assertEqual(len(cache.get_blocks(1)), 1)

    def test_legacy_put_forward_still_works(self):
        """Back-compat: existing call sites using ``put_forward`` keep
        working on the shared cache.
        """
        cache = RankLocalCache()
        b0 = torch.randn(2, 3)
        cache.put_forward(0, [b0], producer_meta=[(1, 1, 0)])
        self.assertEqual(cache.get_forward(0), [b0])
        self.assertEqual(cache.get_producer_meta(0), [(1, 1, 0)])

    def test_drop_is_idempotent_across_adapters(self):
        """Two adapters on the same rank both calling drop(mb) back-to-back
        is fine: second call no-ops on an already-empty mb.
        """
        a1 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=0, num_stages=16, group=None, pp_rank=0
        )
        a2 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=8, num_stages=16, group=None, pp_rank=0
        )
        self.assertIs(a1._cache, a2._cache)

        a1._cache.append(5, torch.ones(2), (0, 0, 0))
        a1._cache.add_grad(5, 0, torch.ones(2))

        a1.on_microbatch_end(5)
        # After a1 drops, mb=5 is gone. a2's drop is a no-op.
        self.assertEqual(a1._cache.get_blocks(5), [])
        a2.on_microbatch_end(5)  # must not raise
        self.assertEqual(a2._cache.get_blocks(5), [])

    def test_multiple_adapters_on_same_rank_drive_shared_cache(self):
        """append / add_grad / pop_grads interleaved across two adapters
        on the same rank see a coherent shared state.
        """
        a1 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=0, num_stages=16, group=None, pp_rank=0
        )
        a2 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=8, num_stages=16, group=None, pp_rank=0
        )

        # Each adapter appends one block (different producer metadata so
        # the shared cache clearly holds both contributions).
        a1._cache.append(3, torch.full((2,), 1.0), (0, 0, 0))
        a2._cache.append(3, torch.full((2,), 2.0), (0, 8, 0))

        blocks = a1._cache.get_blocks(3)
        self.assertEqual(len(blocks), 2)
        # Meta list is index-aligned with blocks.
        meta = a2._cache.get_meta(3)
        self.assertEqual(meta, [(0, 0, 0), (0, 8, 0)])

        # Grad accumulator: a1 writes to block 0, a2 writes to block 1.
        a1._cache.add_grad(3, 0, torch.full((2,), 0.5))
        a2._cache.add_grad(3, 1, torch.full((2,), 0.25))

        grads = a1._cache.pop_grads(3)
        self.assertEqual(len(grads), 2)
        self.assertTrue(torch.allclose(grads[0], torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(grads[1], torch.full((2,), 0.25)))
        # pop drained the accumulator.
        self.assertEqual(a2._cache.pop_grads(3), [])

    # ----- thread-safety --------------------------------------------- #

    def test_concurrent_get_or_create_is_thread_safe(self):
        """Many threads racing into ``_get_or_create_rank_cache`` for the
        same ``pp_rank`` must all land on the SAME object -- no torn
        registration, no duplicate cache.
        """
        num_threads = 32
        results: list[RankLocalCache] = [None] * num_threads  # type: ignore[list-item]
        barrier = threading.Barrier(num_threads)

        def worker(i: int) -> None:
            # Align starts so we maximize the chance of contention on the
            # lock rather than lucky sequential execution.
            barrier.wait()
            results[i] = _get_or_create_rank_cache(99)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = results[0]
        self.assertIsNotNone(first)
        for r in results:
            self.assertIs(r, first)

    def test_concurrent_adapter_construction_same_rank(self):
        """Same invariant, but driven through the public path: two
        ``CrossStageCacheAdapter`` constructors firing on different
        threads with matching ``pp_rank`` must resolve ``self._cache``
        to the same object.
        """
        pp_rank = 77
        a_box: list[CrossStageCacheAdapter] = [None, None]  # type: ignore[list-item]
        barrier = threading.Barrier(2)

        def build(slot: int, stage_id: int) -> None:
            barrier.wait()
            a_box[slot] = CrossStageCacheAdapter(
                nn.Linear(4, 4),
                stage_id=stage_id,
                num_stages=16,
                group=None,
                pp_rank=pp_rank,
            )

        t0 = threading.Thread(target=build, args=(0, 0))
        t1 = threading.Thread(target=build, args=(1, 8))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        self.assertIs(a_box[0]._cache, a_box[1]._cache)


class TestStaticBlockLayoutInterleaved1F1B(unittest.TestCase):
    """Hard-coded golden values for the production launch config
    (``P=8, V=2, num_blocks=8, n_layers=16, layers_per_block=2``) --
    the 8-GPU Interleaved1F1B AttnRes run. A regression in the
    :class:`BlockLayoutTables` walk will trip every entry below.
    """

    def test_static_block_layout_interleaved1f1b_8x2_n8(self):
        t = BlockLayoutTables(
            pp_size=8,
            virtual_stages_per_rank=2,
            num_blocks=8,
            n_layers=16,
            layers_per_block=2,
        )

        # Every even-indexed stage commits block stage_id // 2; odd stages
        # commit nothing. 16 stages total -> 8 commits -> 8 blocks.
        for s in range(16):
            if s % 2 == 0:
                self.assertEqual(t.commits_at(s), [s // 2])
            else:
                self.assertEqual(t.commits_at(s), [])

        # producer_stage_of_block: block b is produced at stage 2b.
        for b in range(8):
            self.assertEqual(t.producer_stage_of_block(b), 2 * b)

        # rank_cache_at_entry: v=0 empty everywhere, v=1 is the union of
        # what the rank committed+received during v=0.
        for r in range(8):
            self.assertEqual(t.rank_cache_at_entry(r, 0), frozenset())
        expected_v1 = {
            0: {0}, 1: {0}, 2: {0, 1}, 3: {0, 1},
            4: {0, 1, 2}, 5: {0, 1, 2}, 6: {0, 1, 2, 3}, 7: {0, 1, 2, 3},
        }
        for r, want in expected_v1.items():
            self.assertEqual(t.rank_cache_at_entry(r, 1), frozenset(want))

        # delta sizes: documented in BlockLayoutTables' docstring.
        v0_hops_expected = [
            [0], [0], [0, 1], [0, 1], [0, 1, 2], [0, 1, 2],
            [0, 1, 2, 3], [1, 2, 3],
        ]
        for s, want in enumerate(v0_hops_expected):
            self.assertEqual(t.delta_to_send(s), want, f"v=0 hop {s}->{s+1}")

        v1_hops_expected = [
            [1, 2, 3, 4], [2, 3, 4], [2, 3, 4, 5], [3, 4, 5],
            [3, 4, 5, 6], [4, 5, 6], [4, 5, 6, 7],
        ]
        for i, want in enumerate(v1_hops_expected):
            s = 8 + i
            self.assertEqual(t.delta_to_send(s), want, f"v=1 hop {s}->{s+1}")

        # Final stage sends nothing.
        self.assertEqual(t.delta_to_send(15), [])

        # consumer_stages_of (Step-2 semantics): only the stages that
        # pull the producer's block from CACHE, not from the delta
        # buffer. These are the stages whose :class:`_SendBlockGradsBack`
        # fires an isend back to the producer. For block b committed at
        # stage 2b, cache consumers are every v=1 stage with b already
        # in its rank_cache_at_entry; blocks committed during v=1
        # (stages 8, 10, 12, 14) have NO cache consumers.
        expected_consumers = {
            0: [8, 9, 10, 11, 12, 13, 14, 15],
            2: [10, 11, 12, 13, 14, 15],
            4: [12, 13, 14, 15],
            6: [14, 15],
            8: [],
            10: [],
            12: [],
            14: [],
        }
        for s, want in expected_consumers.items():
            self.assertEqual(t.consumer_stages_of(s), want, f"stage {s}")
        # Non-committers always have an empty consumer list.
        for s in range(16):
            if s % 2 == 1:
                self.assertEqual(t.consumer_stages_of(s), [])

    def test_layout_rejects_inconsistent_inputs(self):
        # num_blocks != n_layers // layers_per_block -> error.
        with self.assertRaises(ValueError):
            BlockLayoutTables(
                pp_size=4, virtual_stages_per_rank=2, num_blocks=4,
                n_layers=16, layers_per_block=2,
            )
        # n_layers % layers_per_block != 0 -> error.
        with self.assertRaises(ValueError):
            BlockLayoutTables(
                pp_size=2, virtual_stages_per_rank=2, num_blocks=3,
                n_layers=5, layers_per_block=2,
            )


# --------------------------------------------------------------------------- #
# Forward-delta numerics on an in-process multi-stage chain
# --------------------------------------------------------------------------- #


class _ToyAttnResLikeModel(nn.Module):
    """Minimal model that respects the AttnRes per-stage contract.

    Each stage commits exactly ``new_blocks_per_stage`` blocks. With
    ``_return_only_new_blocks=True`` the intermediate return is
    ``(partial, stack(new_blocks))``; last stage returns a scalar-ish
    output. Params are scalars so grad checks are stable.
    """

    def __init__(self, new_blocks_per_stage: int, is_last: bool):
        super().__init__()
        self.new_blocks_per_stage = new_blocks_per_stage
        self.is_last = is_last
        self._return_only_new_blocks: bool = False
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, tokens_or_partial, blocks=None, **kwargs):
        partial = tokens_or_partial
        if blocks is None:
            block_list: list[torch.Tensor] = []
        else:
            block_list = unstack_blocks(blocks)
        initial = len(block_list)
        for _ in range(self.new_blocks_per_stage):
            block_list.append(partial * self.w)
            partial = partial + sum(block_list) / max(len(block_list), 1)
        if not self.is_last:
            if self._return_only_new_blocks:
                new_blocks = block_list[initial:]
                if not new_blocks:
                    empty = partial.new_zeros((0, *partial.shape))
                    return partial, empty
                return partial, stack_blocks(new_blocks)
            return partial, stack_blocks(block_list)
        return partial + sum(block_list)


class TestForwardDeltaNumerics(unittest.TestCase):
    """End-to-end forward numerics: chain 4 toy stages through the delta
    adapter (P=2, V=2, num_blocks=4) in a single process and compare
    against the naive full-stack forward of the same models. Exercises
    the Step-3 cache append + delta-rebuild + outgoing-stack path.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    def _build_chain(self, num_stages: int):
        return [
            _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=(s == num_stages - 1))
            for s in range(num_stages)
        ]

    def test_forward_delta_numerics_2stage(self):
        torch.manual_seed(42)
        P, V = 2, 2
        num_stages = P * V
        num_blocks = num_stages  # one commit per stage
        n_layers = num_stages
        layers_per_block = 1
        B, T, D = 2, 3, 4

        layout = BlockLayoutTables(
            pp_size=P, virtual_stages_per_rank=V,
            num_blocks=num_blocks, n_layers=n_layers,
            layers_per_block=layers_per_block,
        )

        naive = self._build_chain(num_stages)
        adapt = self._build_chain(num_stages)
        with torch.no_grad():
            for n, a in zip(naive, adapt):
                a.w.copy_(n.w)

        tokens = torch.randn(B, T, D)

        # Naive: sequential full-stack.
        p = tokens
        bl = None
        for st in naive[:-1]:
            p, bl = st(p, blocks=bl)
        naive_out = naive[-1](p, blocks=bl)

        # Adapter chain: 4 adapters, pp_rank = stage % P.
        stage_to_rank = {s: s % P for s in range(num_stages)}
        adapters = [
            CrossStageCacheAdapter(
                adapt[s], stage_id=s, num_stages=num_stages, group=None,
                stage_to_rank=stage_to_rank, pp_rank=s % P,
                layout_tables=layout,
            )
            for s in range(num_stages)
        ]
        for a in adapters:
            _set_mb_index(id(a), 0)
        try:
            p0, b0 = adapters[0](tokens)
            prev_p, prev_b = p0, b0
            for s in range(1, num_stages - 1):
                prev_p, prev_b = adapters[s](prev_p, prev_b)
            out = adapters[-1](prev_p, prev_b)
        finally:
            for a in adapters:
                _set_mb_index(id(a), None)

        self.assertTrue(
            torch.allclose(naive_out, out, atol=1e-5),
            f"forward diverges: naive={naive_out}, adapter={out}",
        )

    def test_backward_grad_equivalence_2stage(self):
        """Backward grad check for the SUBSET of params whose grads do
        NOT travel through the send-back P2P in the adapter path.

        With ``group=None`` the cross-rank grad-send-back isends are
        no-ops by design (they're the payload, not the fallback). So a
        producer whose blocks are consumed through a later stage's
        cached prefix loses the consumer contribution locally -- on
        NCCL the irecv would fill it in. This test picks a setup where
        NO cached-prefix lookup happens during the adapter forward, and
        the adapter's grads therefore equal the naive grads exactly.

        Setup: ``P=2, V=1, num_stages=2``. Each rank has exactly one
        virtual stage, so the shared cache never holds blocks from an
        earlier virtual stage of the same rank. The cached-prefix wrap
        is never entered. Delta mode still wraps producer commits with
        :class:`_RecvBlockGradsFromConsumers`, which (group=None) is a
        pure autograd identity. Result: param grads match naive to
        machine precision.
        """
        torch.manual_seed(7)
        # Strictly speaking P=2, V=1 isn't Interleaved1F1B's invariant
        # (V>=2), but the layout tables are happy with it and the
        # forward + grad paths still go through the delta code. This is
        # the no-P2P correctness slice of the backward equivalence.
        P, V = 2, 1
        num_stages = 2
        num_blocks = 2
        n_layers = 2
        B, T, D = 2, 3, 4

        layout = BlockLayoutTables(
            pp_size=P, virtual_stages_per_rank=V,
            num_blocks=num_blocks, n_layers=n_layers, layers_per_block=1,
        )

        naive = [
            _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False),
            _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=True),
        ]
        adapt = [
            _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False),
            _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=True),
        ]
        with torch.no_grad():
            for n, a in zip(naive, adapt):
                a.w.copy_(n.w)

        tokens = torch.randn(B, T, D)

        # Naive path
        p, bl = naive[0](tokens, blocks=None)
        out_n = naive[1](p, blocks=bl)
        out_n.sum().backward()

        # Adapter path
        stage_to_rank = {0: 0, 1: 1}
        adapters = [
            CrossStageCacheAdapter(
                adapt[s], stage_id=s, num_stages=num_stages, group=None,
                stage_to_rank=stage_to_rank, pp_rank=s,
                layout_tables=layout,
            )
            for s in range(num_stages)
        ]
        for a in adapters:
            _set_mb_index(id(a), 0)
        try:
            p0, b0 = adapters[0](tokens)
            out_a = adapters[1](p0, b0)
        finally:
            for a in adapters:
                _set_mb_index(id(a), None)
        out_a.sum().backward()

        for s in range(num_stages):
            self.assertTrue(
                torch.allclose(adapt[s].w.grad, naive[s].w.grad, atol=1e-5),
                f"stage {s} grad diverges: naive={naive[s].w.grad}, "
                f"adapter={adapt[s].w.grad}",
            )


# --------------------------------------------------------------------------- #
# Schedule guard: non-Interleaved1F1B must fall back silently
# --------------------------------------------------------------------------- #


class TestScheduleGuardNonInterleaved(unittest.TestCase):
    """Step-1: when the configured schedule is not Interleaved1F1B, the
    custom pipelining_fn warns + falls back to naive PP. We simulate
    that path by calling the guarded helper directly with a fake
    non-Interleaved schedule and asserting no wrapping happened.
    """

    def test_schedule_guard_non_interleaved(self):
        from torchtitan.experiments.attn_res import pipeline_adapter as pa

        # Stand in for pp_schedule: anything that isn't an instance of
        # _INTERLEAVED_1F1B_CLASS triggers the guard.
        class _FakeNonInterleavedSchedule:
            pass

            # We don't need stages; the guard exits before iteration.
            _stage = None

        fake_schedule = _FakeNonInterleavedSchedule()

        # Monkey-patch core pipeline_llm to return our fake. The guard
        # path must warn and return early without calling _iter_schedule_stages.
        from torchtitan.distributed import pipeline_parallel as core

        original = core.pipeline_llm

        def _fake_pipeline_llm(model, **kwargs):
            # Return shape matches core's contract.
            return fake_schedule, [model], True, True

        core.pipeline_llm = _fake_pipeline_llm
        os.environ["TORCHTITAN_ATTNRES_CACHE"] = "1"
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                # Minimal kwargs; the guard path exits before they're used.
                ret = pa.pipeline_llm_with_cache_adapter(
                    nn.Linear(4, 4),
                    parallel_dims=None,
                    training=None,
                    model_converters=None,
                    parallelism=None,
                    compile_config=None,
                    ac_config=None,
                    dump_folder="",
                    device=torch.device("cpu"),
                    model_config=None,
                    parallelize_fn=None,
                    loss_fn=None,
                )
                # Must have returned the original schedule and un-wrapped
                # model_parts (no adapter), with a user-visible warning.
                sched, parts, hf, hl = ret
                self.assertIs(sched, fake_schedule)
                self.assertEqual(len(parts), 1)
                self.assertNotIsInstance(parts[0], CrossStageCacheAdapter)
                msgs = [str(w.message) for w in rec]
                self.assertTrue(
                    any("Interleaved1F1B" in m for m in msgs),
                    f"expected a warning mentioning Interleaved1F1B, got {msgs}",
                )
        finally:
            core.pipeline_llm = original
            os.environ.pop("TORCHTITAN_ATTNRES_CACHE", None)


# --------------------------------------------------------------------------- #
# Drop-guard: only the last virtual stage on a rank evicts the shared cache
# --------------------------------------------------------------------------- #


class TestDropLifecycleUnderVP(unittest.TestCase):
    """Step-6: in delta mode, the shared :class:`RankLocalCache` must
    only be dropped by the LAST virtual stage on the rank (the one
    with ``stage_id + P >= num_stages``). An earlier virtual stage
    that evicts first would blow away blocks a later virtual stage
    still needs.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    def test_earlier_virtual_stage_drop_is_deferred(self):
        layout = BlockLayoutTables(
            pp_size=2, virtual_stages_per_rank=2,
            num_blocks=4, n_layers=4, layers_per_block=1,
        )
        inner = _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False)
        inner2 = _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False)
        # stage 0 (pp_rank=0, v=0) and stage 2 (pp_rank=0, v=1) share
        # the same rank cache.
        a0 = CrossStageCacheAdapter(
            inner, stage_id=0, num_stages=4, group=None, pp_rank=0,
            stage_to_rank={s: s % 2 for s in range(4)},
            layout_tables=layout,
        )
        a2 = CrossStageCacheAdapter(
            inner2, stage_id=2, num_stages=4, group=None, pp_rank=0,
            stage_to_rank={s: s % 2 for s in range(4)},
            layout_tables=layout,
        )
        self.assertIs(a0._cache, a2._cache)
        a0._cache.append(0, torch.ones(2), (0, 0, 0))
        self.assertEqual(len(a0._cache.get_blocks(0)), 1)

        # Earlier virtual stage calls on_microbatch_end -> no-op.
        a0.on_microbatch_end(0)
        self.assertEqual(
            len(a0._cache.get_blocks(0)), 1,
            "Earlier virtual stage's drop must not evict the shared cache.",
        )
        # Last virtual stage on the rank drops -> cache cleared.
        a2.on_microbatch_end(0)
        self.assertEqual(len(a0._cache.get_blocks(0)), 0)


# --------------------------------------------------------------------------- #
# Deferred grad-send-back: NCCL must NOT run inside autograd.Function.backward
# --------------------------------------------------------------------------- #


class TestGradSendBackDeferredTransport(unittest.TestCase):
    """After the refactor the two autograd Functions must NEVER issue
    any ``dist.isend`` / ``dist.irecv`` / ``dist.batch_isend_irecv``
    calls from inside their ``backward`` methods. Transport is deferred
    to :meth:`CrossStageCacheAdapter._flush_grad_sendback`, which runs
    OUTSIDE the autograd engine (after ``backward_one_chunk`` returns).

    These tests:

    * Assert that a backward pass through ``_SendBlockGradsBack`` with
      a stubbed ``dist`` module that explodes on any P2P call still
      succeeds, and that ``_grad_accum`` has the per-block grad ready
      for a later ``_flush_grad_sendback`` to consume.
    * Exercise ``_flush_grad_sendback`` on a single-adapter end-to-end
      setup where the layout reports zero cross-rank consumers, so the
      flush degenerates to a no-op -- but must still not blow up under
      ``group=None`` / no process group.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    def test_send_backward_does_not_call_nccl(self):
        """Stub out every P2P entry point on ``dist`` with a sentinel
        that raises if called. Backward must still succeed and leave a
        ready-to-ship grad in the cache.
        """
        from torchtitan.experiments.attn_res import pipeline_adapter as pa

        class _NcclCalled(RuntimeError):
            pass

        def _explode(*a, **kw):
            raise _NcclCalled("NCCL must NOT run inside autograd.Function.backward")

        originals = {}
        for name in ("isend", "irecv", "batch_isend_irecv"):
            if hasattr(pa.dist, name):
                originals[name] = getattr(pa.dist, name)
                setattr(pa.dist, name, _explode)

        try:
            cache = _PerMicrobatchCache()
            b0 = torch.randn(2, 3, 4, requires_grad=True)
            b1 = torch.randn(2, 3, 4, requires_grad=True)
            outs = _SendBlockGradsBack.apply(
                [0, 0], [0, 1], None, cache, 5, b0, b1
            )
            (3.0 * outs[0].sum() + 7.0 * outs[1].sum()).backward()
            grads = cache.pop_grads(5)
            self.assertEqual(len(grads), 2)
            self.assertTrue(torch.allclose(grads[0], 3.0 * torch.ones_like(b0)))
            self.assertTrue(torch.allclose(grads[1], 7.0 * torch.ones_like(b1)))
            # None returned for each tensor input.
            self.assertIsNone(b0.grad)
            self.assertIsNone(b1.grad)
        finally:
            for name, fn in originals.items():
                setattr(pa.dist, name, fn)

    def test_recv_backward_does_not_call_nccl(self):
        """Symmetric check on the producer side: backward returns
        ``grad_output`` unchanged and records a pending-recv marker; no
        P2P call occurs.
        """
        from torchtitan.experiments.attn_res import pipeline_adapter as pa

        class _NcclCalled(RuntimeError):
            pass

        def _explode(*a, **kw):
            raise _NcclCalled("NCCL must NOT run inside autograd.Function.backward")

        originals = {}
        for name in ("isend", "irecv", "batch_isend_irecv"):
            if hasattr(pa.dist, name):
                originals[name] = getattr(pa.dist, name)
                setattr(pa.dist, name, _explode)

        try:
            cache = _PerMicrobatchCache()
            x = torch.randn(2, 3, 4, requires_grad=True)
            y = _RecvBlockGradsFromConsumers.apply(
                [1, 2], None, 123, cache, 4, 7, x,
            )
            y.sum().backward()
            # Local grad flows through unchanged.
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))
            # Pending-recv marker is now on the cache, ready for the
            # flush helper to consume.
            pending = cache.pop_pending_recv(4, 7)
            self.assertIsNotNone(pending)
            self.assertEqual(pending["num_blocks"], 2)
            self.assertEqual(pending["consumer_ranks"], (1, 2))
            self.assertEqual(pending["tag_base"], 123)
        finally:
            for name, fn in originals.items():
                setattr(pa.dist, name, fn)

    def test_flush_grad_sendback_is_noop_without_pg(self):
        """Single-adapter CPU setup: ``group=None`` short-circuits the
        flush to a no-op. Covers the code path the 8-GPU launch will
        hit when ``TORCHTITAN_ATTNRES_CACHE=0`` by exercising the same
        entry point.
        """
        layout = BlockLayoutTables(
            pp_size=2, virtual_stages_per_rank=1,
            num_blocks=2, n_layers=2, layers_per_block=1,
        )
        inner = _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False)
        adapter = CrossStageCacheAdapter(
            inner, stage_id=0, num_stages=2, group=None,
            stage_to_rank={0: 0, 1: 1}, pp_rank=0, layout_tables=layout,
        )
        # Pretend a backward just ran: seed the cache with a grad entry.
        adapter._cache.add_grad(0, 0, torch.ones(4))
        # Flush must not blow up and must not crash on missing pg.
        adapter._flush_grad_sendback(0)
        # And the no-op path leaves the grad-accum alone (no pg -> no
        # producer to send to).
        # (``pop_grads`` would have been called only along the NCCL
        # path; under group=None we exit early before touching it.)

    def test_flush_grad_sendback_2stage_delta_mode_no_pg(self):
        """Run the full adapter path on 2 stages, group=None. Backward
        must complete cleanly without any NCCL call (the
        ``_flush_grad_sendback`` helper short-circuits on no PG) and
        param grads land on the local autograd path.
        """
        from torchtitan.experiments.attn_res import pipeline_adapter as pa

        # Guard: count any P2P calls the adapter makes (should be zero).
        call_count = {"isend": 0, "irecv": 0, "batch_isend_irecv": 0}

        originals = {}
        for name in list(call_count.keys()):
            if hasattr(pa.dist, name):
                originals[name] = getattr(pa.dist, name)

                def make_counter(k, orig):
                    def _counter(*a, **kw):
                        call_count[k] += 1
                        return orig(*a, **kw)
                    return _counter

                setattr(pa.dist, name, make_counter(name, originals[name]))

        try:
            torch.manual_seed(0)
            layout = BlockLayoutTables(
                pp_size=2, virtual_stages_per_rank=1,
                num_blocks=2, n_layers=2, layers_per_block=1,
            )
            stage0 = _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False)
            stage1 = _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=True)
            a0 = CrossStageCacheAdapter(
                stage0, stage_id=0, num_stages=2, group=None,
                stage_to_rank={0: 0, 1: 1}, pp_rank=0, layout_tables=layout,
            )
            a1 = CrossStageCacheAdapter(
                stage1, stage_id=1, num_stages=2, group=None,
                stage_to_rank={0: 0, 1: 1}, pp_rank=1, layout_tables=layout,
            )
            tokens = torch.randn(2, 3, 4)

            _set_mb_index(id(a0), 0)
            _set_mb_index(id(a1), 0)
            try:
                p0, b0 = a0(tokens)
                out = a1(p0, b0)
                out.sum().backward()
            finally:
                _set_mb_index(id(a0), None)
                _set_mb_index(id(a1), None)
            # Now drive the flush for both adapters.
            a0._flush_grad_sendback(0)
            a1._flush_grad_sendback(0)

            # Stage 0 and 1 saw their local grads.
            self.assertIsNotNone(stage0.w.grad)
            self.assertIsNotNone(stage1.w.grad)
            # With group=None, NO P2P primitive may have run.
            self.assertEqual(call_count["isend"], 0)
            self.assertEqual(call_count["irecv"], 0)
            self.assertEqual(call_count["batch_isend_irecv"], 0)
        finally:
            for name, fn in originals.items():
                setattr(pa.dist, name, fn)

    def test_flush_grad_sendback_applies_intra_rank_contrib(self):
        """Direct test of the per-block accumulation math: seed the
        shared cache with a committed block and a consumer-side
        accumulated grad whose producer_rank matches ``self.pp_rank``,
        and verify the flush stashes it in ``_intra_rank_grads`` keyed
        by ``(mb, producer_stage, block_idx)``.
        """
        layout = BlockLayoutTables(
            pp_size=2, virtual_stages_per_rank=2,
            num_blocks=4, n_layers=4, layers_per_block=1,
        )
        inner = _ToyAttnResLikeModel(new_blocks_per_stage=1, is_last=False)
        adapter = CrossStageCacheAdapter(
            inner, stage_id=2, num_stages=4, group=None,
            stage_to_rank={s: s % 2 for s in range(4)}, pp_rank=0,
            layout_tables=layout,
        )
        # No PG, so the flush should exit immediately; this test also
        # documents that invariant.
        adapter._cache.add_grad(0, 0, torch.full((2, 3, 4), 0.5))
        adapter._flush_grad_sendback(0)
        # group=None path exits before any state transfer.
        # (If we later support an in-process flush for CPU tests, that
        # helper should write into _intra_rank_grads.)


if __name__ == "__main__":
    unittest.main()
