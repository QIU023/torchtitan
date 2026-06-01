# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for Phase 4c ModelSpec integration.

Covers:
* ``KimiLinearSpec.build()`` dispatches to baseline vs AttnRes variant.
* ``model_registry(flavor)`` returns a valid :class:`ModelSpec` for each
  of the 15 scaling-law flavors.
* ``Trainer.Config`` factory resolves for at least one flavor.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.experiments.attention_residual.kimi_linear import (
    flavor_names,
    KimiLinearAttnResModel,
    KimiLinearConfig,
    KimiLinearModel,
    KimiLinearSpec,
    model_registry,
)
from torchtitan.experiments.attention_residual.kimi_linear.config_registry import (
    kimi_linear_194m_baseline,
    kimi_linear_528m_block_attn_res,
    build_kimi_linear_config,
    SCALING_LAW_TABLE,
)
from torchtitan.protocols.model_spec import ModelSpec


class TestKimiLinearSpec(unittest.TestCase):
    def test_baseline_build(self):
        kcfg = build_kimi_linear_config("194m")
        spec = KimiLinearSpec(kimi_config=kcfg, num_blocks=None)
        model = spec.build()
        self.assertIsInstance(model, KimiLinearModel)

    def test_attn_res_build(self):
        kcfg = build_kimi_linear_config("194m")
        spec = KimiLinearSpec(kimi_config=kcfg, num_blocks=12)
        model = spec.build()
        self.assertIsInstance(model, KimiLinearAttnResModel)
        self.assertEqual(model.num_blocks, 12)

    def test_nparams_and_flops(self):
        kcfg = build_kimi_linear_config("194m")
        spec = KimiLinearSpec(kimi_config=kcfg, num_blocks=None)
        model = spec.build()
        n_params, flops = spec.get_nparams_and_flops(model, seq_len=8192)
        self.assertGreater(n_params, 100_000_000)  # ~580M total MoE params
        # flops is per-TOKEN (not per-step). At 194M activated with
        # MLA attn term, expect ~0.5-5 GFLOPs per token for this size.
        self.assertGreater(flops, 1_000_000)
        self.assertLess(flops, 100_000_000_000)
        # Sanity: must NOT match the old over-counting formula
        # (which was 6 * total_params * seq_len ~= 28e12 for this size).
        self.assertLess(flops, 6 * n_params * 8192 // 10)


class TestModelRegistry(unittest.TestCase):
    def test_all_flavors_build(self):
        flavors = flavor_names()
        # flavor_names() = every scaling-law size × 3 AttnRes variants
        # (baseline / block_attn_res / full_attn_res). Derive the expected
        # count from the table so adding a size row can't silently drift it.
        self.assertEqual(len(flavors), len(SCALING_LAW_TABLE) * 3)
        for flavor in flavors:
            spec = model_registry(flavor)
            self.assertIsInstance(spec, ModelSpec)
            self.assertEqual(spec.name, "kimi_linear")
            self.assertEqual(spec.flavor, flavor)
            # pipelining_fn is wired as of Phase 4d (runtime-dispatches
            # to cache adapter when AttnRes+Interleaved1F1B, else PP passthrough).
            self.assertIsNotNone(spec.pipelining_fn)
            self.assertIsNotNone(spec.parallelize_fn)
            self.assertIsNotNone(spec.build_loss_fn)

    def test_reject_unknown_flavor(self):
        with self.assertRaises(ValueError):
            model_registry("kimi_linear_999q_baseline")

    def test_reject_malformed_flavor(self):
        with self.assertRaises(ValueError):
            model_registry("not_kimi_linear_194m_baseline")


class TestTrainerConfigFactory(unittest.TestCase):
    def test_194m_baseline_builds(self):
        cfg = kimi_linear_194m_baseline()
        self.assertIsNotNone(cfg.model_spec)
        self.assertEqual(cfg.model_spec.flavor, "kimi_linear_194m_baseline")
        # LR from paper Table 2
        self.assertAlmostEqual(cfg.optimizer.lr, 2.99e-3, places=5)

    def test_528m_block_attn_res_builds(self):
        cfg = kimi_linear_528m_block_attn_res()
        self.assertEqual(cfg.model_spec.flavor, "kimi_linear_528m_block_attn_res")
        self.assertAlmostEqual(cfg.optimizer.lr, 2.02e-3, places=5)


if __name__ == "__main__":
    unittest.main()
