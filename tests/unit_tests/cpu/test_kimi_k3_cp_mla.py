# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3's packed MLA context-parallel kernels and their transform, on CPU."""

import unittest
from dataclasses import fields
from unittest import mock

import spmd_types as spmd
import torch

from torchtitan.config.transform import MLAContextParallelTransform
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.models.kimi_k3.cp_mla import (
    MLAKVAllGatherCPFlexInnerAttention,
    MLAUlyssesCPFlexInnerAttention,
)
from torchtitan.models.kimi_k3.model import KimiK3Model

T, H, NOPE, ROPE, V = 8, 4, 8, 4, 6


def _mla_qkv(dtype=torch.float32):
    """q, k, v the way MLA hands them over: one rope vector expanded onto every head."""
    q_THK = torch.randn(T, H, NOPE + ROPE, dtype=dtype)
    k_nope_THN = torch.randn(T, H, NOPE, dtype=dtype)
    k_rope_TR = torch.randn(T, ROPE, dtype=dtype)
    k_THK = torch.cat((k_nope_THN, k_rope_TR.unsqueeze(1).expand(-1, H, -1)), dim=-1)
    v_THV = torch.randn(T, H, V, dtype=dtype)
    return q_THK, k_THK, v_THV


def _run(kernel, q, k, v):
    """Run ``kernel`` with pass-through collectives; return the calls and what
    reached the flex attention."""
    calls, seen = [], {}

    def record(x, group, *, src, dst, backward_options=None):
        calls.append((x, src, dst, backward_options))
        return x

    def capture(self, q, k, v, **kwargs):
        seen.update(q=q, k=k, v=v)
        return q

    with mock.patch(
        "torchtitan.models.kimi_k3.cp_mla.spmd_mesh_group", return_value=object()
    ), mock.patch.object(spmd, "redistribute", record), mock.patch.object(
        FlexInnerAttention, "forward", capture
    ):
        out = kernel.forward(q, k, v)
    return calls, seen, out


class TestPackedMLAKernels(unittest.TestCase):
    def test_ulysses_moves_one_packed_tensor_and_the_rope_slice_once(self):
        q, k, v = _mla_qkv()
        kernel = MLAUlyssesCPFlexInnerAttention(
            MLAUlyssesCPFlexInnerAttention.Config(rope_head_dim=ROPE)
        )
        calls, seen, out = _run(kernel, q, k, v)

        self.assertEqual(3, len(calls))
        packed, src, dst, _ = calls[0]
        self.assertEqual((T, H, (NOPE + ROPE) + NOPE + V), tuple(packed.shape))
        self.assertEqual((spmd.S(0), spmd.S(1)), (src, dst))
        rope, src, dst, backward_options = calls[1]
        self.assertEqual((T, ROPE), tuple(rope.shape))
        self.assertEqual((spmd.S(0), spmd.R), (src, dst))
        self.assertEqual({"op_dtype": torch.float32}, backward_options)
        _, src, dst, _ = calls[2]
        self.assertEqual((spmd.S(1), spmd.S(0)), (src, dst))
        # With pass-through collectives the kernel hands the flex attention
        # exactly what MLA produced: the split, the pack and the expansion
        # are each other's inverse.
        for name, original in (("q", q), ("k", k), ("v", v)):
            self.assertTrue(torch.equal(seen[name], original), name)
        self.assertTrue(torch.equal(out, q))

    def test_all_gather_moves_the_packed_kv_and_the_rope_slice(self):
        q, k, v = _mla_qkv(torch.bfloat16)
        kernel = MLAKVAllGatherCPFlexInnerAttention(
            MLAKVAllGatherCPFlexInnerAttention.Config(rope_head_dim=ROPE)
        )
        calls, seen, _ = _run(kernel, q, k, v)

        self.assertEqual(2, len(calls))
        packed, src, dst, backward_options = calls[0]
        self.assertEqual((T, H, NOPE + V), tuple(packed.shape))
        self.assertEqual((spmd.S(0), spmd.R), (src, dst))
        # The backward reduce-scatter takes the kernel's reduce dtype.
        self.assertEqual({"op_dtype": torch.float32}, backward_options)
        rope, src, dst, backward_options = calls[1]
        self.assertEqual((T, ROPE), tuple(rope.shape))
        self.assertEqual((spmd.S(0), spmd.R), (src, dst))
        self.assertEqual({"op_dtype": torch.float32}, backward_options)
        for name, original in (("q", q), ("k", k), ("v", v)):
            self.assertTrue(torch.equal(seen[name], original), name)

    def test_the_rope_width_is_required(self):
        for kernel in (MLAKVAllGatherCPFlexInnerAttention, MLAUlyssesCPFlexInnerAttention):
            with self.assertRaisesRegex(ValueError, "MLAContextParallelTransform"):
                kernel(kernel.Config())


class TestMLAContextParallelTransform(unittest.TestCase):
    @staticmethod
    def _mla(config):
        model = config.model_spec.model
        assert isinstance(model, KimiK3Model.Config)
        return [layer.attention for layer in model.layers if layer.attention is not None]

    def test_every_mla_layer_gets_the_kernel_and_its_rope_width(self):
        for kernel in (MLAKVAllGatherCPFlexInnerAttention, MLAUlyssesCPFlexInnerAttention):
            config = kimi_k3_debugmodel()
            before = [attention.inner_attention for attention in self._mla(config)]
            self.assertTrue(before)
            MLAContextParallelTransform(inner_attention=kernel).transform(
                config.model_spec.model
            )
            for attention, original in zip(self._mla(config), before):
                inner = attention.inner_attention
                self.assertIsInstance(inner, kernel.Config)
                self.assertEqual(attention.qk_rope_head_dim, inner.rope_head_dim)
                # Every field the layer had rides through the retype.
                for f in fields(original):
                    self.assertEqual(getattr(original, f.name), getattr(inner, f.name))

    def test_a_kernel_without_context_parallelism_is_refused(self):
        with self.assertRaises(ValueError):
            MLAContextParallelTransform(inner_attention=FlexInnerAttention)


if __name__ == "__main__":
    unittest.main()
