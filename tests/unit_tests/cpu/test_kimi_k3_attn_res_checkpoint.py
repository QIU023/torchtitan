# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The attention residual is checkpointed: same values, recomputed in backward.

The residual math upcasts the whole block stack to fp32 twice per call, so
saving those intermediates would make each layer's activation footprint grow
with the stack. Wrapping recomputes them in backward instead. Both halves are
asserted here: the values and gradients are unchanged, and the body runs a
second time during backward rather than reading intermediates back.
"""

import unittest

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3 import model
from torchtitan.models.kimi_k3.model import (
    _apply_attention_residual,
    _attention_residual_math,
)

_TOKENS = 8
_DIM = 16
_BLOCKS = 3


def _operands(requires_grad: bool):
    torch.manual_seed(0)
    prefix_sum = torch.randn(_TOKENS, _DIM, requires_grad=requires_grad)
    blocks = torch.randn(_TOKENS, _BLOCKS, _DIM, requires_grad=requires_grad)
    projection = Linear.Config(in_features=_DIM, out_features=1, bias=False).build()
    norm = RMSNorm.Config(normalized_shape=_DIM, eps=1e-5).build()
    with torch.no_grad():
        projection.weight.normal_()
        norm.weight.normal_()
    return prefix_sum, blocks, projection, norm


class TestAttentionResidualCheckpoint(unittest.TestCase):
    def test_values_and_gradients_are_unchanged(self):
        wrapped_operands = _operands(requires_grad=True)
        plain_operands = _operands(requires_grad=True)

        wrapped = _apply_attention_residual(*wrapped_operands)
        plain = _attention_residual_math(*plain_operands)
        torch.testing.assert_close(wrapped, plain, rtol=0, atol=0)

        wrapped.sum().backward()
        plain.sum().backward()
        for name, actual, expected in zip(
            ("prefix_sum", "blocks"), wrapped_operands[:2], plain_operands[:2]
        ):
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=0, atol=0, msg=f"grad {name}"
            )

    def test_the_math_is_recomputed_in_backward(self):
        """Recomputed, not saved: the body runs a second time during backward."""
        calls = []
        original = model._attention_residual_math

        def counting(*args):
            calls.append(len(calls))
            return original(*args)

        model._attention_residual_math = counting
        try:
            output = model._apply_attention_residual(*_operands(requires_grad=True))
            self.assertEqual(len(calls), 1, "forward should run the body once")
            output.sum().backward()
            self.assertEqual(
                len(calls), 2, "backward should recompute the body, not read it back"
            )
        finally:
            model._attention_residual_math = original

    def test_inference_skips_the_wrapper(self):
        """No autograd, nothing to recompute: the values still match."""
        with torch.no_grad():
            wrapped = _apply_attention_residual(*_operands(requires_grad=False))
            plain = _attention_residual_math(*_operands(requires_grad=False))
        torch.testing.assert_close(wrapped, plain, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
