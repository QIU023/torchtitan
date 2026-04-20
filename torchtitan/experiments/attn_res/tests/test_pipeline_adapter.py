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

from torchtitan.experiments.attn_res.attn_res import stack_blocks, unstack_blocks
from torchtitan.experiments.attn_res.pipeline_adapter import (
    _current_mb_index,
    _get_or_create_rank_cache,
    _install_mb_index_patch,
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


if __name__ == "__main__":
    unittest.main()
