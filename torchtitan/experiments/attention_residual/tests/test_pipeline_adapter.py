# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only unit tests for :mod:`pipeline_adapter`.

Covers:
  * Microbatch-index threading through the monkey-patched
    ``forward_one_chunk`` / ``backward_one_chunk`` hooks.
  * Shared :class:`RankLocalCache` semantics across virtual stages.
  * Forward-delta numerics against a naive full-stack reference.
  * Backward grad equivalence for a 2-stage chain (canary that the
    pure-autograd-through-PP-SEND_B design routes grads correctly).
  * Schedule guard: non-Interleaved1F1B falls back to a warn.
  * VP drop-guard: only the last virtual stage on a rank evicts.

These tests do NOT spin up NCCL; they exercise the CPU / no-PG branch
explicitly (``group=None``). An 8-GPU correctness check is the separate
A/B smoke under ``phase3/launch_8gpu_adapter.sh``.
"""

import os
import threading
import unittest
import warnings
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from torchtitan.experiments.attention_residual.attn_res import stack_blocks, unstack_blocks
from torchtitan.experiments.attention_residual.pipeline_adapter import (
    _current_mb_index,
    _get_or_create_rank_cache,
    _install_augment_hook,
    _install_mb_index_patch,
    _LocalCacheCapture,
    _reset_rank_caches_for_testing,
    _set_mb_index,
    BlockLayoutTables,
    CrossStageCacheAdapter,
    RankLocalCache,
)


class TestMbIndexThreading(unittest.TestCase):
    """The monkey-patched stage forward/backward must make the schedule's
    microbatch index visible to the adapter via a thread-local, so
    cached-block lookups succeed by integer key rather than by
    ``id(tensor)`` (which would change across a P2P boundary).
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

    def test_backward_patch_marks_seen_and_defers_drop(self):
        """The patched ``backward_one_chunk`` MUST NOT drop the cache
        mid-step: rank 0's backward for mb=0 happens at a different
        schedule tick than rank 7's. The patch records the mb in
        ``_seen_mbs`` and leaves everything else intact;
        ``_drop_all_seen_and_clear`` fires at ``pp_schedule.step`` return.
        """
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

        stage.backward_one_chunk(5)

        # Mid-step: mb=5 is SEEN, but cache contents remain.
        self.assertIn(5, adapter._cache._seen_mbs)
        self.assertEqual(len(adapter._cache.get_blocks(5)), 1)

        # Simulate pp_schedule.step's return: drain mb=5.
        adapter._drop_all_seen_and_clear()
        self.assertEqual(adapter._cache.get_blocks(5), [])
        self.assertNotIn(5, adapter._cache._seen_mbs)


class _InnerToy(nn.Module):
    """Tiny inner model with a mix of params and a buffer -- stands in
    for ``AttnResModel`` for the state_dict key-layout tests.
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
    """Per-process, rank-keyed :class:`RankLocalCache` registry.
    Invariants:

      * Two adapters in the same process with the same ``pp_rank`` hold
        THE SAME ``RankLocalCache`` object (VP=2 sharing case).
      * Different ``pp_rank`` -> different cache objects.
      * ``append``/``drop`` compose correctly when multiple adapters on
        the same rank drive them.
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
        self.assertEqual(cache.get_blocks(0), [b0])
        self.assertEqual(cache.get_meta(0), [(1, 1, 0)])

    def test_drop_is_idempotent_across_adapters(self):
        """Two adapters on the same rank both calling the step-end
        drop back-to-back is fine: second call no-ops on an
        already-empty mb.
        """
        a1 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=0, num_stages=16, group=None, pp_rank=0
        )
        a2 = CrossStageCacheAdapter(
            nn.Linear(4, 4), stage_id=8, num_stages=16, group=None, pp_rank=0
        )
        self.assertIs(a1._cache, a2._cache)

        a1._cache.append(5, torch.ones(2), (0, 0, 0))
        # Record that mb=5 saw backward on this rank.
        a1.on_microbatch_end(5)
        self.assertIn(5, a1._cache._seen_mbs)

        # With num_stages=16 and implicit P=num_stages, both a1 and a2
        # are drop-eligible (naive mode: stage_id + num_stages >=
        # num_stages always). a1 drops first; a2's drop is a no-op.
        a1._drop_all_seen_and_clear()
        self.assertEqual(a1._cache.get_blocks(5), [])
        a2._drop_all_seen_and_clear()  # must not raise
        self.assertEqual(a2._cache.get_blocks(5), [])

    def test_multiple_adapters_on_same_rank_drive_shared_cache(self):
        """append interleaved across two adapters on the same rank
        sees a coherent shared state.
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

        # consumer_stages_of (kept on the layout for telemetry): the
        # stages that pull the producer's block from CACHE, not from
        # the delta buffer. For block b committed at stage 2b, cache
        # consumers are every v=1 stage with b already in its
        # rank_cache_at_entry; blocks committed during v=1 (stages 8,
        # 10, 12, 14) have NO cache consumers.
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
    """End-to-end forward numerics: chain toy stages through the delta
    adapter in a single process and compare against the naive full-stack
    forward of the same models. Exercises the cache append +
    delta-rebuild + outgoing-stack path.
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

    def test_backward_grad_equivalence_4stage_vp2(self):
        """V=2 canary for the _LocalCacheAugment / _LocalCacheCapture
        dance: with P=2, V=2, num_stages=4, rank 0 hosts stages {0, 2}.
        Stage 2's forward reads block 0 (committed by stage 0 on the
        same rank) back from the shared cache -- this is the exact
        "same-rank own-commit cache read" path the new Functions cover.
        Param grads must match naive autograd.
        """
        torch.manual_seed(42)
        P, V = 2, 2
        num_stages = P * V
        num_blocks = num_stages
        n_layers = num_stages
        B, T, D = 2, 3, 4

        layout = BlockLayoutTables(
            pp_size=P, virtual_stages_per_rank=V,
            num_blocks=num_blocks, n_layers=n_layers, layers_per_block=1,
        )

        naive = self._build_chain(num_stages)
        adapt = self._build_chain(num_stages)
        with torch.no_grad():
            for n, a in zip(naive, adapt):
                a.w.copy_(n.w)

        tokens = torch.randn(B, T, D)

        # Naive path
        p = tokens
        bl = None
        for st in naive[:-1]:
            p, bl = st(p, blocks=bl)
        naive_out = naive[-1](p, blocks=bl)
        naive_out.sum().backward()

        # Adapter path
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
        out.sum().backward()

        for s in range(num_stages):
            self.assertTrue(
                torch.allclose(adapt[s].w.grad, naive[s].w.grad, atol=1e-5),
                f"stage {s} grad diverges: naive={naive[s].w.grad}, "
                f"adapter={adapt[s].w.grad}",
            )

    def test_backward_grad_equivalence_2stage(self):
        """Canary for the pure-autograd-through-PP-SEND_B design:
        with num_stages=2 and layer_to_stage wired so each rank has
        exactly one virtual stage, the shared cache never holds cached
        prefix blocks. Delta-mode forward builds blocks_tensor entirely
        from the recv_delta slices; backward flows through torch.cat
        into the recv tensor (which, in production, PP SEND_B drains
        back to rank 0) and into local wrapped params.

        Since we're single-process with group=None, the recv tensor IS
        the producer's output tensor (same autograd graph end-to-end),
        so param grads match naive to machine precision.
        """
        torch.manual_seed(7)
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
    """When the configured schedule is not Interleaved1F1B, the custom
    pipelining_fn warns + falls back to naive PP. We simulate that path
    by calling the guarded helper directly with a fake non-Interleaved
    schedule and asserting no wrapping happened.
    """

    def test_schedule_guard_non_interleaved(self):
        from torchtitan.experiments.attention_residual import pipeline_adapter as pa

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
    """In delta mode, the shared :class:`RankLocalCache` must only be
    dropped by the LAST virtual stage on the rank (the one with
    ``stage_id + P >= num_stages``). An earlier virtual stage that
    evicts first would blow away blocks a later virtual stage still
    needs.
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

        # Simulate a completed backward for both virtual stages on this
        # rank (both record the mb in the shared cache's _seen_mbs).
        a0.on_microbatch_end(0)
        a2.on_microbatch_end(0)
        self.assertIn(0, a0._cache._seen_mbs)

        # Earlier virtual stage calls the step-end drop -> no-op (the
        # VP drop guard keeps the shared cache alive for later virtual
        # stages on the same rank).
        a0._drop_all_seen_and_clear()
        self.assertEqual(
            len(a0._cache.get_blocks(0)), 1,
            "Earlier virtual stage's drop must not evict the shared cache.",
        )
        # Last virtual stage on the rank drops -> cache cleared.
        a2._drop_all_seen_and_clear()
        self.assertEqual(len(a0._cache.get_blocks(0)), 0)
        self.assertNotIn(0, a0._cache._seen_mbs)


# --------------------------------------------------------------------------- #
# Local-only autograd.Function dance (replaces the retain_graph monkey-patch)
# --------------------------------------------------------------------------- #


class TestLocalCacheAutogradFunctions(unittest.TestCase):
    """Unit tests for the hook-based augment + :class:`_LocalCacheCapture`.

    The producer-side ``_install_augment_hook`` and the consumer-side
    ``_LocalCacheCapture`` together replace the prior process-global
    ``retain_graph=True`` override. The gate for integrating them into
    ``_forward_delta`` / ``_finish_forward`` is that the four tests
    below all pass on CPU with pure autograd -- no PG, no schedule.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    def test_local_cache_capture_blocks_backward_propagation(self):
        """Capture.backward deposits grad in the slot and returns None for
        its tensor input. The real adapter pairs this with a DETACHED
        cache entry (so the input has no upstream grad_fn at all); this
        unit test mirrors that contract by detaching ``b`` before passing
        it to Capture, so the upstream tensor ``a`` cannot receive a
        grad even in principle.
        """
        cache = RankLocalCache()
        key = (0, 0, 0)

        a = torch.randn(3, 4, requires_grad=True)
        b = (a * 2.0).detach()
        b.requires_grad_(True)  # leaf with grad enabled — adapter pattern
        captured_in = _LocalCacheCapture.apply(b, key, cache)
        # Further ops downstream of the capture must still get their
        # grads the usual way.
        c = captured_in + 5.0
        loss = c.sum()
        loss.backward()

        # Captured slot holds what flowed into the capture's input.
        slot = cache._captured_grads.get(key)
        self.assertIsNotNone(slot)
        # dL/d(captured_in) = dL/dc * 1 = ones_like(c)
        self.assertTrue(torch.allclose(slot, torch.ones_like(b)))

        # Capture severed the path; a is upstream of the (detached) b
        # but autograd cannot reach a via the Capture.
        self.assertIsNone(a.grad)

    def test_augment_hook_adds_captured_to_incoming_grad(self):
        """``_install_augment_hook`` installs a tensor grad hook that,
        when the producer block's grad fires during the producer's own
        backward, pops the matching captured-grad slot and sums it
        into the incoming grad. Pre-populate the slot with a known
        value G and verify the upstream gradient picks up ``incoming + G``.
        """
        cache = RankLocalCache()
        key = (0, 0, 0)

        a = torch.randn(3, 4, requires_grad=True)
        b = a * 3.0  # autograd-live to a
        _install_augment_hook(b, key, cache)
        c = b * 1.0  # incoming grad path
        loss = c.sum()

        # Pre-populate the captured slot (simulate an earlier
        # consumer-side Capture having already deposited G).
        G = torch.full_like(b, 7.0)
        cache.capture_grad(key, G)

        loss.backward()

        # dL/db incoming = ones_like(c). Hook adds G. So upstream
        # d/da = 3 * (ones + G).
        expected_a_grad = 3.0 * (torch.ones_like(a) + G)
        self.assertIsNotNone(a.grad)
        self.assertTrue(torch.allclose(a.grad, expected_a_grad))
        # Slot should now be empty (pop semantics).
        self.assertNotIn(key, cache._captured_grads)

    def test_multi_consumer_hook_sums_across_captures(self):
        """V>2: producer's hook sees ``incoming + sum_of_all_captures``.

        Two independent consumer branches each wrap the SAME detached
        cache entry (simulating one producer, two later virtual stages
        on the same rank). The producer-side hook fires once on the
        producer's own backward path and reads ``capture_grad`` which
        was summed across both consumer Captures.
        """
        cache = RankLocalCache()
        key = (0, 0, 0)

        a = torch.randn(3, 4, requires_grad=True)
        b = a * 2.0  # shared producer
        _install_augment_hook(b, key, cache)

        # Producer's OWN downstream path (simulating the SEND_B grad
        # that eventually arrives at the producer block).
        own_path = b * 1.0
        own_loss = own_path.sum()

        # Two consumer branches read DETACHED copies of b from the
        # cache and wrap each in Capture. Both Captures' grads sum
        # into the same slot via cache.capture_grad.
        cached = b.detach()
        cached.requires_grad_(True)
        cap1 = _LocalCacheCapture.apply(cached, key, cache)
        cap2 = _LocalCacheCapture.apply(cached, key, cache)
        cons1_loss = (cap1 * 2.0).sum()  # dL1/d(cap1) = 2 * ones
        cons2_loss = (cap2 * 5.0).sum()  # dL2/d(cap2) = 5 * ones

        total_loss = own_loss + cons1_loss + cons2_loss
        total_loss.backward()

        # Slot must be drained by the producer's hook.
        self.assertNotIn(key, cache._captured_grads)

        # a.grad = 2.0 * (incoming_from_own_path + sum_of_both_captures)
        #        = 2.0 * (1*ones + 2*ones + 5*ones) = 16 * ones.
        expected_a_grad = torch.full_like(a, 2.0 * (1.0 + 2.0 + 5.0))
        self.assertIsNotNone(a.grad)
        self.assertTrue(
            torch.allclose(a.grad, expected_a_grad),
            f"a.grad={a.grad}, expected={expected_a_grad}",
        )

    def test_producer_param_grad_equivalence_to_naive(self):
        """End-to-end param-grad equivalence on a toy mimic of the real
        pattern: a producer emits a block that flows into BOTH (a) the
        delta sent forward to later stages (returns grad via SEND_B)
        AND (b) the shared cache, from which a later same-rank virtual
        stage reads it.

        In the wrapped path the producer block carries an
        ``_install_augment_hook`` and the cached read site uses Capture
        on a DETACHED cache entry. Run once without wrappers (baseline),
        once with the hook+Capture pair. All param grads must match.
        """
        cache = RankLocalCache()
        key = (0, 0, 0)

        torch.manual_seed(11)
        tokens = torch.randn(2, 4, requires_grad=False)

        def _forward_joint(
            w_p_, w_delta_, w_c_, use_wrappers: bool,
        ) -> torch.Tensor:
            block = tokens @ w_p_  # producer emission
            if use_wrappers:
                _install_augment_hook(block, key, cache)
            # Path 1: outgoing delta (analogue of SEND_B downstream).
            delta_path = (block @ w_delta_).sum()
            # Path 2: later same-rank virtual stage reads the cached
            # block. In the wrapped flow the cache stores a DETACHED
            # leaf; Capture intercepts the grad and stores it in a slot.
            if use_wrappers:
                cached = block.detach()
                cached.requires_grad_(True)
                cached_read = _LocalCacheCapture.apply(cached, key, cache)
            else:
                cached_read = block
            consumer_path = (cached_read @ w_c_ + cached_read * 0.3).sum()
            return delta_path + consumer_path

        # --- baseline (no wrappers): naive autograd over the joint
        # forward. All three params' grads are what we must match.
        w_p = nn.Parameter(torch.randn(4, 4))
        w_delta = nn.Parameter(torch.randn(4, 4))
        w_c = nn.Parameter(torch.randn(4, 4))
        loss_n = _forward_joint(w_p, w_delta, w_c, use_wrappers=False)
        loss_n.backward()
        naive_wp = w_p.grad.detach().clone()
        naive_wd = w_delta.grad.detach().clone()
        naive_wc = w_c.grad.detach().clone()

        # --- wrapped path: hook at emission, Capture on detached cache
        # read.
        w_p2 = nn.Parameter(w_p.detach().clone())
        w_delta2 = nn.Parameter(w_delta.detach().clone())
        w_c2 = nn.Parameter(w_c.detach().clone())
        loss_a = _forward_joint(w_p2, w_delta2, w_c2, use_wrappers=True)
        loss_a.backward()

        self.assertTrue(
            torch.allclose(w_p2.grad, naive_wp, atol=1e-5),
            f"producer w_p grad diverges: naive={naive_wp} wrapped={w_p2.grad}",
        )
        self.assertTrue(
            torch.allclose(w_delta2.grad, naive_wd, atol=1e-5),
            f"delta w_delta grad diverges: naive={naive_wd} wrapped={w_delta2.grad}",
        )
        self.assertTrue(
            torch.allclose(w_c2.grad, naive_wc, atol=1e-5),
            f"consumer w_c grad diverges: naive={naive_wc} wrapped={w_c2.grad}",
        )
        # Slot must be empty after the full backward.
        self.assertNotIn(key, cache._captured_grads)


class TestCaptureCountAudit(unittest.TestCase):
    """Validates the hook's capture-count audit against
    :meth:`BlockLayoutTables.expected_same_rank_captures`.

    Silent grad loss (a same-rank consumer's backward silently not
    firing) was previously invisible because the hook would just
    ``pop_grad`` returning ``None`` and do nothing. These tests pin
    down the observed-vs-expected comparison now wired through the
    layout tables.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    def test_no_warning_when_count_matches_expected(self):
        cache = RankLocalCache()
        key = (0, 0, 0)
        a = torch.randn(3, 4, requires_grad=True)
        b = a * 2.0
        _install_augment_hook(b, key, cache, expected_captures=2)

        cached = b.detach()
        cached.requires_grad_(True)
        cap1 = _LocalCacheCapture.apply(cached, key, cache)
        cap2 = _LocalCacheCapture.apply(cached, key, cache)
        total = (b * 1.0).sum() + (cap1 * 2.0).sum() + (cap2 * 5.0).sum()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            total.backward()

        audit_warns = [x for x in w if "capture-count mismatch" in str(x.message)]
        self.assertEqual(
            audit_warns, [], f"unexpected capture-count warning: {audit_warns}"
        )

    def test_warning_when_consumer_backward_missing(self):
        """Layout expects 1 same-rank consumer deposit; no Capture fires,
        so count=0 and the hook warns.
        """
        cache = RankLocalCache()
        key = (0, 0, 0)
        a = torch.randn(3, 4, requires_grad=True)
        b = a * 2.0
        _install_augment_hook(b, key, cache, expected_captures=1)

        # Only the producer's own backward runs; the consumer-side
        # Capture never executes -> 0 deposits where 1 was expected.
        (b * 1.0).sum().backward()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # re-issue to observe: run in a second backward to exercise
            # the warn path deterministically.
            a2 = torch.randn(3, 4, requires_grad=True)
            b2 = a2 * 2.0
            _install_augment_hook(b2, key, cache, expected_captures=1)
            (b2 * 1.0).sum().backward()
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("capture-count mismatch" in m for m in msgs),
            f"expected capture-count warning, got {msgs}",
        )

    def test_warning_when_extra_consumer_deposits(self):
        """Layout expects 1 deposit; two Captures deposit. Hook still
        sums both into producer grad (correctness preserved) but warns
        because the static layout didn't predict the extra consumer.
        """
        cache = RankLocalCache()
        key = (0, 0, 0)
        a = torch.randn(3, 4, requires_grad=True)
        b = a * 2.0
        _install_augment_hook(b, key, cache, expected_captures=1)

        cached = b.detach()
        cached.requires_grad_(True)
        cap1 = _LocalCacheCapture.apply(cached, key, cache)
        cap2 = _LocalCacheCapture.apply(cached, key, cache)
        total = (b * 1.0).sum() + cap1.sum() + cap2.sum()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            total.backward()
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("capture-count mismatch" in m for m in msgs),
            f"expected capture-count warning, got {msgs}",
        )

    def test_capture_counts_drains_on_pop(self):
        cache = RankLocalCache()
        key = (0, 0, 0)
        a = torch.randn(3, 4, requires_grad=True)
        b = a * 1.0
        _install_augment_hook(b, key, cache, expected_captures=1)
        cached = b.detach()
        cached.requires_grad_(True)
        cap = _LocalCacheCapture.apply(cached, key, cache)
        ((b * 1.0).sum() + cap.sum()).backward()
        self.assertNotIn(key, cache._captured_grads)
        self.assertNotIn(key, cache._capture_counts)

    def test_expected_same_rank_captures_canonical(self):
        """Canonical Interleaved1F1B: P=4, V=2, N=8, one block per stage.
        Every v=0 producer has exactly one same-rank v=1 consumer; no
        v=1 producer has any later same-rank consumer.
        """
        layout = BlockLayoutTables(
            pp_size=4, virtual_stages_per_rank=2, num_blocks=8,
            n_layers=8, layers_per_block=1,
        )
        for producer_stage in range(4):
            self.assertEqual(
                layout.expected_same_rank_captures(producer_stage, 0), 1,
                f"producer_stage={producer_stage}",
            )
        for producer_stage in range(4, 8):
            self.assertEqual(
                layout.expected_same_rank_captures(producer_stage, 0), 0,
                f"producer_stage={producer_stage}",
            )

    def test_expected_same_rank_captures_v3(self):
        """V=3: producer at v=0 has 2 same-rank consumers (v=1, v=2);
        producer at v=1 has 1 (v=2); producer at v=2 has 0.
        """
        layout = BlockLayoutTables(
            pp_size=2, virtual_stages_per_rank=3, num_blocks=6,
            n_layers=6, layers_per_block=1,
        )
        # v=0 producers: stages 0, 1
        for s in (0, 1):
            self.assertEqual(layout.expected_same_rank_captures(s, 0), 2)
        # v=1 producers: stages 2, 3
        for s in (2, 3):
            self.assertEqual(layout.expected_same_rank_captures(s, 0), 1)
        # v=2 producers: stages 4, 5
        for s in (4, 5):
            self.assertEqual(layout.expected_same_rank_captures(s, 0), 0)

    def test_expected_same_rank_captures_bounds(self):
        layout = BlockLayoutTables(
            pp_size=2, virtual_stages_per_rank=2, num_blocks=4,
            n_layers=4, layers_per_block=1,
        )
        self.assertEqual(layout.expected_same_rank_captures(0, 5), 0)
        self.assertEqual(layout.expected_same_rank_captures(99, 0), 0)
        self.assertEqual(layout.expected_same_rank_captures(0, -1), 0)


class TestCaptureGradCloneSemantics(unittest.TestCase):
    """Validates the defensive ``.detach().clone()`` in ``capture_grad``.

    Before the clone was added, ``_captured_grads[key]`` aliased the
    backward-produced grad tensor. If a downstream framework (FSDP2
    post-backward pipeline, future torch.compile'd backward) were to
    reuse or mutate that storage between deposit and pop, the slot
    would silently corrupt. Test: mutate the original grad after
    deposit and verify the slot is unaffected.
    """

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def test_capture_grad_decouples_from_input_storage(self):
        cache = RankLocalCache()
        key = (0, 0, 0)
        grad = torch.full((2, 3), 7.0)
        cache.capture_grad(key, grad)
        # Mutate original grad in place; slot must NOT observe the change.
        grad.fill_(999.0)
        stored = cache._captured_grads[key]
        self.assertTrue(
            torch.allclose(stored, torch.full_like(stored, 7.0)),
            f"slot leaked mutation: stored={stored}",
        )
        # pop_grad returns (grad, count) under the new API.
        popped, count = cache.pop_grad(key)
        self.assertEqual(count, 1)
        self.assertTrue(torch.allclose(popped, torch.full_like(popped, 7.0)))

    def test_capture_grad_accumulates_deposits(self):
        cache = RankLocalCache()
        key = (0, 0, 0)
        cache.capture_grad(key, torch.full((2, 3), 1.0))
        cache.capture_grad(key, torch.full((2, 3), 2.0))
        cache.capture_grad(key, torch.full((2, 3), 4.0))
        popped, count = cache.pop_grad(key)
        self.assertEqual(count, 3)
        self.assertTrue(torch.allclose(popped, torch.full_like(popped, 7.0)))
        # pop drains both slot and counter
        popped2, count2 = cache.pop_grad(key)
        self.assertIsNone(popped2)
        self.assertEqual(count2, 0)


# ----- DTensor smoke test ------------------------------------------------- #
# Only importable when torch.distributed.tensor is available. This test
# does NOT exercise real sharding; it verifies that register_hook,
# detach, Capture.apply, and grad+captured arithmetic all behave
# correctly when the operand happens to be a DTensor (the common case
# under FSDP2 + torchtitan). A 1-rank Replicate mesh is the minimal
# surface area that still hits the DTensor code paths.

try:
    import torch.distributed as _dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    _DTENSOR_IMPORTABLE = True
except (ImportError, RuntimeError):
    _DTENSOR_IMPORTABLE = False


@unittest.skipUnless(
    _DTENSOR_IMPORTABLE, "torch.distributed.tensor not importable"
)
class TestHookWithDTensor(unittest.TestCase):
    """Smoke test: the hook+detach+Capture bridge on DTensor operands.

    Under real FSDP2 the producer block output can be a DTensor (plain
    Replicate over the FSDP mesh, or Shard over TP mesh). This test
    pins down that the bridge produces param grads identical to the
    plain-tensor baseline when the top-level inputs are DTensor.
    """

    _owns_pg = False

    @classmethod
    def setUpClass(cls) -> None:
        if not _DTENSOR_IMPORTABLE:
            return
        if not _dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29599")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")
            try:
                _dist.init_process_group(backend="gloo", rank=0, world_size=1)
                cls._owns_pg = True
            except Exception as e:  # pragma: no cover — test env specific
                raise unittest.SkipTest(f"gloo PG init failed: {e!r}")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owns_pg and _dist.is_initialized():
            _dist.destroy_process_group()

    def setUp(self) -> None:
        _reset_rank_caches_for_testing()

    def tearDown(self) -> None:
        _reset_rank_caches_for_testing()

    def test_hook_and_capture_on_dtensor_match_plain_path(self):
        mesh = init_device_mesh("cpu", (1,))

        torch.manual_seed(47)
        w_p_init = torch.randn(4, 4)
        w_delta_init = torch.randn(4, 4)
        w_c_init = torch.randn(4, 4)
        tokens_init = torch.randn(2, 4)

        def _run(use_dtensor: bool):
            cache = RankLocalCache()
            key = (0, 0, 0)
            if use_dtensor:
                # Under real FSDP2 every operand that meets the producer
                # graph is a DTensor (params via FSDP wrap, activations
                # via DTensor redistribute). Wrap all three here so the
                # first matmul is DTensor @ DTensor.
                tokens = DTensor.from_local(
                    tokens_init.clone(), mesh, [Replicate()]
                )
                w_p = nn.Parameter(
                    DTensor.from_local(w_p_init.clone(), mesh, [Replicate()])
                )
                w_delta = nn.Parameter(
                    DTensor.from_local(w_delta_init.clone(), mesh, [Replicate()])
                )
                w_c = nn.Parameter(
                    DTensor.from_local(w_c_init.clone(), mesh, [Replicate()])
                )
            else:
                tokens = tokens_init.clone()
                w_p = nn.Parameter(w_p_init.clone())
                w_delta = nn.Parameter(w_delta_init.clone())
                w_c = nn.Parameter(w_c_init.clone())

            block = tokens @ w_p
            if use_dtensor:
                self.assertIsInstance(
                    block, DTensor, "producer emission should be DTensor"
                )
            _install_augment_hook(block, key, cache, expected_captures=1)

            # outgoing-delta analogue
            delta_loss = (block @ w_delta).sum()

            # cached-read analogue (same-rank consumer)
            cached = block.detach()
            if use_dtensor:
                self.assertIsInstance(
                    cached, DTensor, "detach should preserve DTensor type"
                )
            cached.requires_grad_(True)
            cached_read = _LocalCacheCapture.apply(cached, key, cache)
            if use_dtensor:
                self.assertIsInstance(
                    cached_read, DTensor,
                    "Capture.forward output should preserve DTensor type",
                )
            consumer_loss = (cached_read @ w_c + cached_read * 0.3).sum()

            (delta_loss + consumer_loss).backward()

            # Slot drained, counter drained.
            self.assertNotIn(key, cache._captured_grads)
            self.assertNotIn(key, cache._capture_counts)

            def _local(t: torch.Tensor) -> torch.Tensor:
                return t.to_local() if isinstance(t, DTensor) else t

            return (
                _local(w_p.grad).detach().clone(),
                _local(w_delta.grad).detach().clone(),
                _local(w_c.grad).detach().clone(),
            )

        plain_grads = _run(use_dtensor=False)
        dt_grads = _run(use_dtensor=True)

        for plain, dt, name in zip(
            plain_grads, dt_grads, ("w_p", "w_delta", "w_c")
        ):
            self.assertTrue(
                torch.allclose(plain, dt, atol=1e-5),
                f"{name} grad diverges between plain and DTensor paths:\n"
                f"plain={plain}\ndt={dt}",
            )


if __name__ == "__main__":
    unittest.main()
