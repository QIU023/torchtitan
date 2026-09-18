# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch

from torchtitan.models.kimi_k3.model import (
    _apply_attention_residual,
    _AttentionResidualAggregation,
)

EPS = 1e-5
TOKENS, BLOCKS, DIM = 16, 3, 8
WIDE_DIM = 1024


def _reference(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
) -> torch.Tensor:
    """The same aggregation written as plain autograd ops."""
    values_TND = (
        block_residual_TND
        if partial_block_TD is None
        else torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)
    )
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + EPS)
    score_weight_D = norm_weight_D.float() * projection_1D.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    return torch.matmul(probs_T1N, values_float).squeeze(1).to(values_TND.dtype)


def _inputs(blocks: int, with_partial: bool, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def make(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).requires_grad_()

    partial_block_TD = make(TOKENS, DIM) if with_partial else None
    return (
        partial_block_TD,
        make(TOKENS, blocks, DIM),
        make(1, DIM),
        make(DIM),
        torch.randn(TOKENS, DIM, generator=generator),
    )


class TestKimiK3AttentionResidual(unittest.TestCase):
    def _compare(self, blocks: int, with_partial: bool) -> None:
        partial, stack, projection, norm_weight, grad_output = _inputs(
            blocks, with_partial
        )
        leaves = [x for x in (partial, stack, projection, norm_weight) if x is not None]
        outputs, grads = [], []
        for aggregate in (
            lambda: _reference(partial, stack, projection, norm_weight),
            lambda: _apply_attention_residual(
                partial,
                stack,
                types.SimpleNamespace(weight=projection),
                types.SimpleNamespace(weight=norm_weight, eps=EPS),
            ),
        ):
            for leaf in leaves:
                leaf.grad = None
            aggregate().backward(grad_output)
            outputs.append(aggregate().detach())
            grads.append([leaf.grad.detach().clone() for leaf in leaves])
        torch.testing.assert_close(outputs[0], outputs[1])
        for expected, actual in zip(*grads):
            torch.testing.assert_close(expected, actual)

    def test_matches_plain_autograd(self):
        for blocks in (1, BLOCKS):
            for with_partial in (True, False):
                with self.subTest(blocks=blocks, with_partial=with_partial):
                    self._compare(blocks, with_partial)

    def test_registered_for_spmd_type_checking(self):
        # An autograd Function the checker does not know about raises under the
        # strict type checking the multimodal cell runs with.
        from spmd_types._local_registration import _LOCAL_AUTOGRAD_FUNCTIONS

        self.assertIn(_AttentionResidualAggregation, _LOCAL_AUTOGRAD_FUNCTIONS)

    def test_bf16_weights_keep_fp32_score_precision(self):
        # The score weight is a product of two parameters. Rounding that
        # product to bfloat16 before the upcast, rather than upcasting both
        # factors first, moves the output by half a percent.
        generator = torch.Generator().manual_seed(0)

        def make(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, dtype=torch.bfloat16)

        partial, stack = make(TOKENS, WIDE_DIM), make(TOKENS, BLOCKS, WIDE_DIM)
        # Scaled so the depth softmax is not saturated: a saturated one is
        # insensitive to the score weight and hides the precision loss.
        projection = (make(1, WIDE_DIM).float() * WIDE_DIM ** -0.5).bfloat16()
        norm_weight = make(WIDE_DIM)
        expected = _reference(partial, stack, projection, norm_weight)
        actual = _apply_attention_residual(
            partial,
            stack,
            types.SimpleNamespace(weight=projection),
            types.SimpleNamespace(weight=norm_weight, eps=EPS),
        )
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-3, atol=2e-3)

    def test_empty_stack_with_a_partial_block(self):
        # A stage that holds no block yet still aggregates its own prefix sum.
        self._compare(0, True)


if __name__ == "__main__":
    unittest.main()
