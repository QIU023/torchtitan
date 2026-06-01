# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Kimi Linear's PP adapter plumbing.

Focused on the parts that are Kimi-specific (FQN name remapping +
AttnRes-presence detection via ``num_blocks`` attr). The heavy lift —
``CrossStageCacheAdapter`` / ``RankLocalCache`` / the hook+detach
bridge — is tested in ``torchtitan/experiments/attn_res/tests/`` and
reused verbatim.
"""

from __future__ import annotations

import unittest

from torchtitan.experiments.attention_residual.kimi_linear.pipeline_adapter import (
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
        ``pipeline_kimi_linear_with_cache_adapter``. Runtime detection
        (baseline vs AttnRes) happens inside that function via
        ``num_blocks`` attr check, not at registration time.
        """
        from torchtitan.experiments.attention_residual.kimi_linear import (
            flavor_names, model_registry,
        )
        from torchtitan.experiments.attention_residual.kimi_linear.pipeline_adapter import (
            pipeline_kimi_linear_with_cache_adapter,
        )
        for flavor in flavor_names():
            spec = model_registry(flavor)
            self.assertEqual(
                spec.pipelining_fn, pipeline_kimi_linear_with_cache_adapter,
                f"{flavor}: pipelining_fn not wired",
            )


if __name__ == "__main__":
    unittest.main()
