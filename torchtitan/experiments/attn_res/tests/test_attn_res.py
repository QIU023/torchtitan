# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Block Attention Residuals.

Covers the core ``block_attn_res`` primitive, the ``AttnResProjection``
config/build path, the stack/unstack helpers, and end-to-end forward and
backward on a debug-sized ``AttnResLlama3Model``. CPU only -- no GPU or
distributed setup required.
"""

import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.experiments.attn_res import attn_res_configs
from torchtitan.experiments.attn_res.attn_res import (
    AttnResProjection,
    block_attn_res,
    stack_blocks,
    unstack_blocks,
)
from torchtitan.models.common.rmsnorm import RMSNorm


def _zero_proj(dim: int) -> AttnResProjection:
    """Helper: build a zero-initialized AttnResProjection."""
    config = AttnResProjection.Config(dim=dim, param_init={"weight": nn.init.zeros_})
    proj = config.build()
    proj.init_states()
    return proj


def _unit_norm(dim: int) -> RMSNorm:
    """Helper: build an RMSNorm with weight=ones."""
    config = RMSNorm.Config(normalized_shape=dim, param_init={"weight": nn.init.ones_})
    norm = config.build()
    norm.init_states()
    return norm


class TestBlockAttnResFunction(unittest.TestCase):
    """Tests for the core block_attn_res softmax-over-depth primitive."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T, self.D = 2, 3, 8

    def test_single_partial_is_identity(self):
        """N=0 blocks + 1 partial -> output equals partial (softmax over 1 item)."""
        proj = _zero_proj(self.D)
        norm = _unit_norm(self.D)
        partial = torch.randn(self.B, self.T, self.D)
        out = block_attn_res([], partial, proj, norm)
        self.assertTrue(torch.allclose(out, partial, atol=1e-6))

    def test_zero_query_is_uniform_average(self):
        """Zero pseudo-query -> output is the uniform average of (blocks + partial).

        This is THE invariant that lets us start training equivalent to
        standard residuals: with w_l = 0, softmax(0) = uniform, so each
        source contributes 1/(N+1).
        """
        proj = _zero_proj(self.D)
        norm = _unit_norm(self.D)
        b0 = torch.randn(self.B, self.T, self.D)
        b1 = torch.randn(self.B, self.T, self.D)
        partial = torch.randn(self.B, self.T, self.D)
        out = block_attn_res([b0, b1], partial, proj, norm)
        expected = (b0 + b1 + partial) / 3.0
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_nonzero_query_diverges_from_uniform(self):
        """A non-zero pseudo-query makes block_attn_res responsive to keys."""
        proj = _zero_proj(self.D)
        nn.init.normal_(proj.weight, std=0.1)
        norm = _unit_norm(self.D)
        b0 = torch.randn(self.B, self.T, self.D)
        b1 = torch.randn(self.B, self.T, self.D)
        partial = torch.randn(self.B, self.T, self.D)
        uniform = (b0 + b1 + partial) / 3.0
        out = block_attn_res([b0, b1], partial, proj, norm)
        self.assertFalse(torch.allclose(out, uniform, atol=1e-3))

    def test_softmax_weights_sum_to_one(self):
        """Softmax over depth means total weight across sources = 1 per token."""
        proj = _zero_proj(self.D)
        nn.init.normal_(proj.weight, std=0.5)
        norm = _unit_norm(self.D)
        # If we set all values to the same constant, output should equal it.
        const = torch.ones(self.B, self.T, self.D) * 3.14
        out = block_attn_res(
            [const.clone(), const.clone()], const.clone(), proj, norm
        )
        self.assertTrue(torch.allclose(out, const, atol=1e-5))

    def test_gradients_flow(self):
        """Gradients reach blocks, partial, pseudo-query, and norm weight."""
        proj = _zero_proj(self.D)
        norm = _unit_norm(self.D)
        b0 = torch.randn(self.B, self.T, self.D, requires_grad=True)
        b1 = torch.randn(self.B, self.T, self.D, requires_grad=True)
        partial = torch.randn(self.B, self.T, self.D, requires_grad=True)
        out = block_attn_res([b0, b1], partial, proj, norm)
        out.sum().backward()
        self.assertIsNotNone(b0.grad)
        self.assertIsNotNone(b1.grad)
        self.assertIsNotNone(partial.grad)
        self.assertIsNotNone(proj.weight.grad)
        self.assertIsNotNone(norm.weight.grad)

    def test_pseudo_query_grad_nonzero(self):
        """Gradient on the pseudo-query is non-zero when sources differ.

        When b0 != b1 != partial, the softmax is non-trivial and pushes a
        signal through the query. Guards against an accidental
        detach/stop-gradient on the pseudo-query path.
        """
        proj = _zero_proj(self.D)
        norm = _unit_norm(self.D)
        b0 = torch.randn(self.B, self.T, self.D)
        b1 = torch.randn(self.B, self.T, self.D)
        partial = torch.randn(self.B, self.T, self.D)
        out = block_attn_res([b0, b1], partial, proj, norm)
        out.sum().backward()
        self.assertGreater(proj.weight.grad.abs().sum().item(), 0.0)


class TestAttnResProjection(unittest.TestCase):
    """Tests for the AttnResProjection Config/Module."""

    def test_build_and_zero_init(self):
        config = AttnResProjection.Config(
            dim=16, param_init={"weight": nn.init.zeros_}
        )
        proj = config.build()
        proj.init_states()
        self.assertEqual(proj.weight.shape, torch.Size([1, 16]))
        self.assertTrue(torch.all(proj.weight == 0))
        self.assertIsNone(proj.bias)

    def test_init_states_respects_param_init(self):
        """If param_init is overridden, init_states uses the override."""
        config = AttnResProjection.Config(
            dim=8, param_init={"weight": partial(nn.init.constant_, val=0.5)}
        )
        proj = config.build()
        proj.init_states()
        self.assertTrue(torch.all(proj.weight == 0.5))


class TestStackUnstackBlocks(unittest.TestCase):
    """Tests for stack_blocks / unstack_blocks round-trip."""

    def test_roundtrip(self):
        B, T, D = 2, 3, 8
        blocks = [torch.randn(B, T, D) for _ in range(4)]
        stacked = stack_blocks(blocks)
        self.assertEqual(stacked.shape, torch.Size([4, B, T, D]))
        unstacked = unstack_blocks(stacked)
        self.assertEqual(len(unstacked), 4)
        for orig, recon in zip(blocks, unstacked):
            self.assertTrue(torch.equal(orig, recon))

    def test_roundtrip_preserves_grad(self):
        """Round-trip must keep autograd connections to the source tensors."""
        B, T, D = 2, 3, 4
        b0 = torch.randn(B, T, D, requires_grad=True)
        b1 = torch.randn(B, T, D, requires_grad=True)
        stacked = stack_blocks([b0, b1])
        unstacked = unstack_blocks(stacked)
        loss = sum(t.sum() for t in unstacked)
        loss.backward()
        self.assertIsNotNone(b0.grad)
        self.assertIsNotNone(b1.grad)


class TestAttnResLlama3Model(unittest.TestCase):
    """End-to-end tests of a debug-sized AttnResLlama3Model.

    Builds the ``debugmodel_attn_res`` config (6 layers / 3 blocks), runs a
    forward pass on random tokens, and checks the output shape, zero-init of
    pseudo-queries, backward, and the PP-intermediate-stage tuple return.
    """

    def _build_model(self):
        config = attn_res_configs["debugmodel_attn_res"]()
        model = config.build()
        model.init_states()
        return model, config

    def test_build_and_forward(self):
        model, config = self._build_model()
        B, T = 2, 16
        tokens = torch.randint(0, config.vocab_size, (B, T))
        logits = model(tokens)
        self.assertEqual(logits.shape, torch.Size([B, T, config.vocab_size]))

    def test_pseudo_queries_are_zero_after_init(self):
        """All per-layer and final pseudo-queries must be zero right after init_states."""
        model, _ = self._build_model()
        for layer in model.layers.values():
            self.assertTrue(torch.all(layer.attn_res_proj.weight == 0))
            self.assertTrue(torch.all(layer.mlp_res_proj.weight == 0))
        self.assertTrue(torch.all(model.final_attn_res_proj.weight == 0))

    def test_forward_backward(self):
        model, config = self._build_model()
        B, T = 2, 8
        tokens = torch.randint(0, config.vocab_size, (B, T))
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()
        for layer in model.layers.values():
            self.assertIsNotNone(layer.attn_res_proj.weight.grad)
            self.assertIsNotNone(layer.mlp_res_proj.weight.grad)
        self.assertIsNotNone(model.final_attn_res_proj.weight.grad)

    def test_pp_intermediate_stage_returns_tuple(self):
        """When tok_embeddings / output are pruned (PP middle stage), forward
        should return (partial_block, stacked_blocks) so PipelineStage can
        send both tensors to the next stage.
        """
        config = attn_res_configs["debugmodel_attn_res"]()
        model = config.build()
        model.init_states()

        # Simulate a PP middle stage by stripping embedding/output/norm.
        model.tok_embeddings = None
        model.norm = None
        model.output = None

        B, T, D = 2, 8, config.dim
        partial = torch.randn(B, T, D)
        blocks_tensor = torch.randn(1, B, T, D)
        out = model(partial, blocks=blocks_tensor)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        new_partial, new_blocks = out
        self.assertEqual(new_partial.shape, torch.Size([B, T, D]))
        self.assertEqual(new_blocks.shape[1:], torch.Size([B, T, D]))
        # Stage has 6 layers / 3 blocks = 2 layers per block. Every even
        # layer_id is a block start, so the stage commits blocks at
        # layer_id = 0, 2, 4 -> 3 new commits added to the initial 1 block.
        self.assertEqual(new_blocks.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
