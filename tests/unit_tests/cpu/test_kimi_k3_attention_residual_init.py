# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.model import _apply_attention_residual

EPS = 1e-5
TOKENS, BLOCKS, DIM = 16, 3, 8


def _aggregate(prefix, stack, projection, norm_weight):
    return _apply_attention_residual(
        prefix,
        stack,
        types.SimpleNamespace(weight=projection),
        types.SimpleNamespace(weight=norm_weight, eps=EPS),
    )


class TestKimiK3AttentionResidualInit(unittest.TestCase):
    def test_zero_projection_gives_uniform_depth_weights(self):
        # Zero initialisation makes the initial depth weights uniform, so the
        # aggregation returns the mean of its sources exactly.
        generator = torch.Generator().manual_seed(0)
        prefix = torch.randn(TOKENS, DIM, generator=generator)
        stack = torch.randn(TOKENS, BLOCKS, DIM, generator=generator)
        actual = _aggregate(prefix, stack, torch.zeros(1, DIM), torch.ones(DIM))
        expected = torch.cat((stack, prefix.unsqueeze(1)), dim=1).mean(dim=1)
        torch.testing.assert_close(actual, expected)

    def test_zero_projection_moves_the_projection_but_not_the_norm(self):
        # The norm weight reaches the loss only through its product with the
        # projection, so at zero initialisation it has no gradient. It gains one
        # as soon as the projection leaves zero.
        generator = torch.Generator().manual_seed(0)
        prefix = torch.randn(TOKENS, DIM, generator=generator)
        stack = torch.randn(TOKENS, BLOCKS, DIM, generator=generator)
        projection = torch.zeros(1, DIM, requires_grad=True)
        norm_weight = torch.ones(DIM, requires_grad=True)
        output = _aggregate(prefix, stack, projection, norm_weight)
        output.backward(torch.randn(TOKENS, DIM, generator=generator))
        assert projection.grad is not None and norm_weight.grad is not None
        self.assertGreater(projection.grad.abs().max().item(), 0.0)
        self.assertEqual(norm_weight.grad.abs().max().item(), 0.0)

    def test_residual_projections_are_zero_initialised(self):
        model = model_registry("debugmodel")
        self.assertIsNone(model.layers[0].attention_res_proj)
        for projection in (
            model.layers[0].ffn_res_proj,
            model.layers[1].attention_res_proj,
            model.output_res_proj,
        ):
            assert projection is not None and projection.param_init is not None
            self.assertIs(projection.param_init["weight"], torch.nn.init.zeros_)


if __name__ == "__main__":
    unittest.main()
