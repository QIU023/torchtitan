# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stable LatentMoE entry/exit -- K3 tech report sec 2.3, Eq. 11.

    u = sum_{i in Tk(x)} p_i * E_i^routed(W_down x)
    y = sum_j E_j^shared(x) + W_up RMSNorm(u)

Official widths: hidden 7168, routed_expert_hidden_size 3584 (the latent l),
moe_intermediate_size 3072 (inside each routed expert), num_shared_experts 2.
The routed dispatch itself is GPU-only, so what is covered here is the shared
entry/exit math and the fail-loud guard on the unwired training path.
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3.model import (
    KimiLatentMoEProjection,
    KimiLinearConfig,
    KimiMoE,
)


class TestLatentProjection(unittest.TestCase):
    def test_official_widths(self):
        proj = KimiLatentMoEProjection(7168, 3584)
        self.assertEqual(proj.down.weight.shape, (3584, 7168))
        self.assertEqual(proj.up.weight.shape, (7168, 3584))
        self.assertEqual(proj.norm.normalized_shape, (3584,))

    def test_round_trip_shapes(self):
        proj = KimiLatentMoEProjection(64, 32)
        x = torch.randn(2, 5, 64)
        u = proj.to_latent(x)
        self.assertEqual(u.shape, (2, 5, 32))
        self.assertEqual(proj.from_latent(u).shape, (2, 5, 64))

    def test_norm_sits_before_up(self):
        torch.manual_seed(0)
        proj = KimiLatentMoEProjection(64, 32)
        u = torch.randn(2, 5, 32) * 100  # scale the aggregate up
        torch.testing.assert_close(proj.from_latent(u), proj.up(proj.norm(u)))

    def test_norm_makes_exit_scale_insensitive(self):
        # the point of sec 2.3.1: u's scale varies with the selected experts
        torch.manual_seed(0)
        proj = KimiLatentMoEProjection(64, 32)
        u = torch.randn(2, 5, 32)
        a = proj.from_latent(u)
        b = proj.from_latent(u * 50.0)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_norm_can_be_disabled(self):
        proj = KimiLatentMoEProjection(64, 32, use_norm=False)
        self.assertIsNone(proj.norm)
        u = torch.randn(2, 5, 32)
        torch.testing.assert_close(proj.from_latent(u), proj.up(u))

    def test_projections_are_shared_not_per_expert(self):
        # one down/up pair per layer -- applied once per token, which is what
        # keeps 896-expert dispatch affordable (traffic is O(l), not O(d))
        proj = KimiLatentMoEProjection(64, 32)
        names = {n for n, _ in proj.named_parameters()}
        self.assertEqual(names, {"down.weight", "up.weight", "norm.weight"})


class TestLatentMoEGuard(unittest.TestCase):
    def _cfg(self, latent):
        return KimiLinearConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            num_experts=8,
            num_experts_per_token=2,
            moe_intermediate_size=32,
            routed_expert_hidden_size=latent,
        )

    def test_unwired_latent_path_fails_loudly(self):
        with self.assertRaises(NotImplementedError) as cm:
            KimiMoE(self._cfg(48))
        msg = str(cm.exception)
        self.assertIn("full-width token", msg)
        self.assertIn("K3_RECONCILIATION", msg)

    def test_none_keeps_the_conventional_path_constructible(self):
        # not asserting a forward (routed dispatch is GPU-only), only that the
        # non-latent config still builds as it did before the release
        KimiMoE(self._cfg(None))


if __name__ == "__main__":
    unittest.main()
