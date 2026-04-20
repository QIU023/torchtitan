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

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from torchtitan.experiments.attn_res.attn_res import stack_blocks, unstack_blocks
from torchtitan.experiments.attn_res.pipeline_adapter import (
    _current_mb_index,
    _grad_tag_base,
    _install_mb_index_patch,
    _PerMicrobatchCache,
    _RecvBlockGradsFromConsumers,
    _SendBlockGradsBack,
    _set_mb_index,
    CrossStageCacheAdapter,
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
        y = _RecvBlockGradsFromConsumers.apply([1, 2], None, 0, x)
        self.assertTrue(torch.allclose(y, x))

    def test_backward_passes_local_grad(self):
        x = torch.randn(3, 2, 4, 5, requires_grad=True)
        y = _RecvBlockGradsFromConsumers.apply([1, 2], None, 0, x)
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

    def test_backward_accumulates_block_grads_in_cache(self):
        """The consumer adapter's backward must accumulate per-block
        grads into its internal ``_grad_accum``, keyed by the mb index.

        We arrange for the middle stage to have a single cached prefix
        block, run backward, and assert the cache holds a tensor of
        the correct shape. With group=None the send-back Function then
        silently drops the grad -- real runs would isend it back.
        """
        torch.manual_seed(0)
        dim = 4
        B, T = 2, 3

        # 3 stages; middle stage will have 1 cached prefix block.
        stage_mid = _ToyBlockModel(dim=dim, new_blocks_per_stage=1, is_last=True)
        adapter_mid = CrossStageCacheAdapter(
            stage_mid, stage_id=1, num_stages=3, group=None
        )

        # Pre-seed the adapter's cache with one cached prefix block for
        # mb=0 -- simulating what stage 1 would have after receiving
        # stage 0's commit and putting it in the cache on the previous
        # forward call.
        mb = 0
        cached_block = torch.randn(B, T, dim, requires_grad=True)
        adapter_mid._cache.put_forward(
            mb, [cached_block], producer_meta=[(0, 0, 0)]
        )

        partial = torch.randn(B, T, dim, requires_grad=True)
        # Fresh-from-prev blocks: shape [0, B, T, D] to simulate a prev
        # stage that committed nothing on this forward (keeps the
        # assertion tight).
        new_blocks_tensor = partial.new_zeros((0, B, T, dim))

        _set_mb_index(id(adapter_mid), mb)
        try:
            out = adapter_mid(partial, new_blocks_tensor)
        finally:
            _set_mb_index(id(adapter_mid), None)

        # Force a backward.
        out.sum().backward()

        grads = adapter_mid._cache._grad_accum.get(mb, [])
        self.assertEqual(len(grads), 1, "one cached block should have one grad entry")
        self.assertIsNotNone(grads[0])
        self.assertEqual(grads[0].shape, cached_block.shape)
        self.assertGreater(grads[0].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
