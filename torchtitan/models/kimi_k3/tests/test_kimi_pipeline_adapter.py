# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Kimi Linear's PP adapter plumbing.

Focused on the parts that are Kimi-specific (FQN name remapping +
AttnRes-presence detection via ``num_blocks`` attr). The heavy lift —
``CrossStageCacheAdapter`` / ``RankLocalCache`` / the hook+detach
bridge — is tested in ``torchtitan/models/kimi_k3/tests/`` and
reused verbatim.
"""

from __future__ import annotations

import unittest

from torchtitan.models.kimi_k3.pipeline_adapter import (
    _KIMI_ATTN_RES_LAST_STAGE_FQNS,
    _kimi_llm_fqns,
)


class TestKimiFQNRemapping(unittest.TestCase):
    def test_embed_tokens_and_lm_head_replacements(self):
        """``tok_embeddings`` → ``embed_tokens``, ``output`` → ``lm_head``."""
        # 2 stages, 4 layers, default weights.
        fqns = _kimi_llm_fqns(num_stages=2, num_layers=4)
        # Stage 0 should start with embed_tokens, stage 1 ends with lm_head.
        self.assertEqual(fqns[0][0], "embed_tokens")
        self.assertIn("lm_head", fqns[-1])
        self.assertNotIn("tok_embeddings", fqns[0])
        self.assertNotIn("output", fqns[-1])

    def test_layers_preserved(self):
        """Layer FQNs (``layers.N``) pass through untouched."""
        fqns = _kimi_llm_fqns(num_stages=2, num_layers=4)
        flat = [name for stage in fqns for name in stage]
        for i in range(4):
            self.assertIn(f"layers.{i}", flat)

    def test_stage_count(self):
        """Requested stage count matches output length."""
        for n in (1, 2, 4, 8):
            fqns = _kimi_llm_fqns(
                num_stages=n,
                num_layers=max(n, 4),
            )
            self.assertEqual(len(fqns), n)

    def test_attn_res_extra_fqns_constant(self):
        """Last-stage AttnRes extras are exactly the two final modules."""
        self.assertEqual(
            _KIMI_ATTN_RES_LAST_STAGE_FQNS,
            ("final_attn_res_proj", "final_attn_res_norm"),
        )


class TestPipeliningFnInModelSpec(unittest.TestCase):
    def test_all_flavors_wire_pipelining_fn(self):
        """Every registered flavor's ModelSpec points at
        ``pipeline_kimi_k3_with_cache_adapter``. Runtime detection
        (baseline vs AttnRes) happens inside that function via
        ``num_blocks`` attr check, not at registration time.
        """
        from torchtitan.models.kimi_k3 import flavor_names, model_registry
        from torchtitan.models.kimi_k3.pipeline_adapter import (
            pipeline_kimi_k3_with_cache_adapter,
        )

        for flavor in flavor_names():
            spec = model_registry(flavor)
            self.assertEqual(
                spec.pipelining_fn,
                pipeline_kimi_k3_with_cache_adapter,
                f"{flavor}: pipelining_fn not wired",
            )


class TestContiguousSplitGuard(unittest.TestCase):
    """The layer->stage discovery verifies the layout it cannot replace.

    ``stages`` is the local rank's stages, so the discovery can never see every
    layer and the map it builds is always partial. What it can do is check the
    contiguous default against the layers this rank actually holds.
    """

    @staticmethod
    def _stage(stage_index: int, layer_ids):
        from torch import nn

        submod = nn.Module()
        if layer_ids is not None:
            submod.layers = nn.ModuleDict({str(i): nn.Identity() for i in layer_ids})
        stage = nn.Module()
        stage.submod = submod
        stage.stage_index = stage_index
        return stage

    def _infer(self, stages):
        from torchtitan.models.kimi_k3.layout import (
            _infer_block_layout_tables_from_stages,
        )

        # 8 layers over 2 stages -> 4 per stage; blocks of 4 -> 2 blocks.
        return _infer_block_layout_tables_from_stages(
            stages, pp_size=2, num_blocks=2, n_layers=8, layers_per_block=4
        )

    def test_a_contiguous_rank_is_accepted(self):
        tables = self._infer([self._stage(1, [4, 5, 6, 7])])
        self.assertEqual(tables.num_blocks, 2)

    def test_a_non_contiguous_split_raises_instead_of_mislaying_blocks(self):
        # Stage 1 holding the first four layers contradicts the default, which
        # would route block deltas to the wrong stage.
        with self.assertRaises(ValueError) as ctx:
            self._infer([self._stage(1, [0, 1, 2, 3])])
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_stages_without_layers_leave_nothing_to_verify(self):
        tables = self._infer([self._stage(0, None)])
        self.assertEqual(tables.num_blocks, 2)


class TestStepEndSweep(unittest.TestCase):
    """What the step-end sweep evicts.

    Only backward marks a microbatch as seen, so a sweep keyed on the seen-set
    alone cannot reach anything a forward-only pass cached.
    """

    @staticmethod
    def _adapter():
        from torch import nn

        from torchtitan.models.kimi_k3.pipeline_adapter import CrossStageCacheAdapter

        return CrossStageCacheAdapter(nn.Identity(), stage_id=0, num_stages=1)

    def test_a_forward_only_microbatch_is_evicted(self):
        import torch

        adapter = self._adapter()
        adapter._cache.append(0, torch.zeros(2), (0, 0, 0))
        # Evaluation reaches exactly this state: cached blocks, nothing marked.
        self.assertEqual(adapter._cache._seen_mbs, set())
        adapter._drop_all_cached_and_clear()
        self.assertEqual(adapter._cache.get_blocks(0), [])

    def test_a_backward_marked_microbatch_is_still_evicted(self):
        import torch

        adapter = self._adapter()
        adapter._cache.append(1, torch.zeros(2), (0, 0, 0))
        adapter.on_microbatch_end(1)
        adapter._drop_all_cached_and_clear()
        self.assertEqual(adapter._cache.get_blocks(1), [])
        self.assertEqual(adapter._cache._seen_mbs, set())


if __name__ == "__main__":
    unittest.main()
