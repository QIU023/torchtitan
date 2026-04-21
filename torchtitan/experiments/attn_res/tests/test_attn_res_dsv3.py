# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the DSv3-shaped MoE + MLA + AttnRes flavors.

Covers build, per-layer dense / MoE mix (DSv3 first-N-dense convention),
zero-init of pseudo-queries, forward+backward, and the
``model_registry`` dispatch that picks ``parallelize_deepseekv3`` vs
``parallelize_llama`` based on whether the config has any MoE layers.
CPU only; uses the small ``dsv3_debugmodel_attn_res`` flavor (6 layers,
1 dense + 5 MoE, 8 experts, dim=256, N=3 AttnRes blocks).
"""

import unittest

import torch
import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.attn_res import attn_res_configs, model_registry
from torchtitan.experiments.attn_res.model import (
    AttnResModel,
    AttnResTransformerBlock,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.model import Attention as DSv3MLAAttention
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter


class TestDSv3AttnResLayers(unittest.TestCase):
    """Config-level tests: layer list shape, MoE-vs-dense split, MLA attn."""

    def setUp(self):
        self.config = attn_res_configs["dsv3_debugmodel_attn_res"]()

    def test_layer_count_and_model_shape(self):
        self.assertIsInstance(self.config, AttnResModel.Config)
        self.assertEqual(self.config.dim, 256)
        self.assertEqual(len(self.config.layers), 6)
        self.assertEqual(self.config.attn_res.num_blocks, 3)
        self.assertTrue(self.config.attn_res.enabled)

    def test_first_layer_dense_rest_moe(self):
        """DSv3 pattern: layer 0 dense, layers 1-5 MoE."""
        layers = self.config.layers
        self.assertIsNone(layers[0].moe, "layer 0 should be dense (moe is None)")
        self.assertIsNotNone(layers[0].feed_forward)
        for i in range(1, 6):
            self.assertIsNotNone(
                layers[i].moe, f"layer {i} should have MoE"
            )
            self.assertIsNone(
                layers[i].feed_forward,
                f"layer {i} should NOT have dense feed_forward",
            )

    def test_every_layer_has_mla_attention(self):
        for layer in self.config.layers:
            self.assertIsInstance(
                layer.attention,
                DSv3MLAAttention.Config,
                "DSv3-flavor AttnRes must use MLA attention on every layer",
            )

    def test_every_layer_has_attn_res_params(self):
        """AttnRes is orthogonal to MoE/dense: every layer gets the four params."""
        for layer in self.config.layers:
            self.assertIsNotNone(layer.attn_res_proj)
            self.assertIsNotNone(layer.mlp_res_proj)
            self.assertIsNotNone(layer.attn_res_norm)
            self.assertIsNotNone(layer.mlp_res_norm)

    def test_final_attn_res_params_on_model(self):
        self.assertIsNotNone(self.config.final_attn_res_proj)
        self.assertIsNotNone(self.config.final_attn_res_norm)


class TestDSv3AttnResModel(unittest.TestCase):
    """Build + forward + backward smoke on the DSv3 AttnRes debug model."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = attn_res_configs["dsv3_debugmodel_attn_res"]()
        self.model = self.config.build()
        self.model.init_states()

    def test_build_produces_correct_block_types(self):
        """Each built layer is an AttnResTransformerBlock; layer 0 has
        ``feed_forward``, layers 1-5 have ``moe``.
        """
        for i, layer in enumerate(self.model.layers.values()):
            self.assertIsInstance(layer, AttnResTransformerBlock)
            if i == 0:
                self.assertFalse(layer.moe_enabled)
                self.assertTrue(hasattr(layer, "feed_forward"))
                self.assertFalse(hasattr(layer, "moe"))
            else:
                self.assertTrue(layer.moe_enabled)
                self.assertIsInstance(layer.moe, MoE)
                self.assertFalse(hasattr(layer, "feed_forward"))

    def test_pseudo_queries_zero_after_init(self):
        """Every AttnRes projection (per-layer + final) is zero-initialized.

        This is the training-stability invariant: the softmax starts
        uniform, so the model's first-step output equals a plain
        average of sources (= standard residual up to normalization).
        """
        for layer in self.model.layers.values():
            self.assertTrue(
                torch.all(layer.attn_res_proj.weight == 0),
                "per-layer attn_res_proj weight must be zero at init",
            )
            self.assertTrue(
                torch.all(layer.mlp_res_proj.weight == 0),
                "per-layer mlp_res_proj weight must be zero at init",
            )
        self.assertTrue(torch.all(self.model.final_attn_res_proj.weight == 0))

    def test_forward_shape(self):
        B, T = 2, 8
        tokens = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(tokens)
        self.assertEqual(logits.shape, torch.Size([B, T, self.config.vocab_size]))

    def test_forward_finite(self):
        """MoE + MLA + AttnRes composition does not NaN on step 0."""
        B, T = 2, 8
        tokens = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(tokens)
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_backward_grads_reach_attn_res_params(self):
        B, T = 2, 8
        tokens = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(tokens)
        loss = logits.sum()
        loss.backward()
        # AttnRes params on every layer receive a grad (sources differ
        # after layer 0, so softmax-through-query grad is non-trivial).
        for i, layer in enumerate(self.model.layers.values()):
            self.assertIsNotNone(
                layer.attn_res_proj.weight.grad,
                f"layer {i} attn_res_proj grad missing",
            )
            self.assertIsNotNone(
                layer.mlp_res_proj.weight.grad,
                f"layer {i} mlp_res_proj grad missing",
            )
        self.assertIsNotNone(self.model.final_attn_res_proj.weight.grad)

    def test_forward_backward_grads_reach_moe_router(self):
        """Backward reaches the router gate on at least one MoE layer."""
        B, T = 2, 8
        tokens = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(tokens)
        logits.sum().backward()
        # Layers 1..5 are MoE; pick one and check its router.
        moe_layer = self.model.layers["1"]
        self.assertTrue(moe_layer.moe_enabled)
        # Router gate weight grad is the canonical MoE grad signal.
        router_gate = moe_layer.moe.router.gate
        self.assertIsNotNone(router_gate.weight.grad)


class TestModelRegistryDispatch(unittest.TestCase):
    """``model_registry`` picks DSv3 machinery for MoE flavors and the
    Llama3 machinery for dense flavors, selected from the Config itself
    (whether any layer has ``moe is not None``)."""

    def test_dsv3_flavor_uses_deepseekv3_parallelize(self):
        spec = model_registry("dsv3_debugmodel_attn_res")
        self.assertIs(spec.parallelize_fn, parallelize_deepseekv3)
        self.assertIs(spec.state_dict_adapter, DeepSeekV3StateDictAdapter)
        self.assertIs(
            spec.post_optimizer_build_fn, register_moe_load_balancing_hook
        )

    def test_dense_flavor_uses_llama_parallelize(self):
        spec = model_registry("debugmodel_attn_res")
        self.assertIs(spec.parallelize_fn, parallelize_llama)
        self.assertIs(spec.state_dict_adapter, Llama3StateDictAdapter)
        self.assertIsNone(spec.post_optimizer_build_fn)

    def test_dsv3_16b_flavor_uses_deepseekv3_parallelize(self):
        """Also verify the training-scale MoE flavor resolves correctly."""
        spec = model_registry("dsv3_16b_attn_res")
        self.assertIs(spec.parallelize_fn, parallelize_deepseekv3)
        self.assertIs(
            spec.post_optimizer_build_fn, register_moe_load_balancing_hook
        )

    def test_all_flavors_resolve(self):
        """Every flavor in the registry can at least build its ModelSpec.

        No forward runs here -- just verifies the config builders and
        registry plumbing are self-consistent.
        """
        from torchtitan.experiments.attn_res import attn_res_configs

        for flavor in attn_res_configs.keys():
            with self.subTest(flavor=flavor):
                spec = model_registry(flavor)
                self.assertIsNotNone(spec.model)
                self.assertIsNotNone(spec.parallelize_fn)
                self.assertIsNotNone(spec.pipelining_fn)


class TestGetNparamsAndFlopsDispatch(unittest.TestCase):
    """``AttnResModel.Config.get_nparams_and_flops`` picks MoE vs dense
    helper based on whether any layer has ``moe``."""

    def test_dsv3_config_goes_through_moe_helper(self):
        """MoE helper for the MoE flavor; just checks it does not error.

        The numeric return values depend on torchtitan's MoE flop model
        so we only assert the call completes and returns sensible
        positive ints.
        """
        config = attn_res_configs["dsv3_debugmodel_attn_res"]()
        model = config.build()
        model.init_states()
        nparams, nflops_per_token = config.get_nparams_and_flops(model, seq_len=128)
        self.assertGreater(nparams, 0)
        self.assertGreater(nflops_per_token, 0)

    def test_dense_config_goes_through_dense_helper(self):
        config = attn_res_configs["debugmodel_attn_res"]()
        model = config.build()
        model.init_states()
        nparams, nflops_per_token = config.get_nparams_and_flops(model, seq_len=128)
        self.assertGreater(nparams, 0)
        self.assertGreater(nflops_per_token, 0)


if __name__ == "__main__":
    unittest.main()
