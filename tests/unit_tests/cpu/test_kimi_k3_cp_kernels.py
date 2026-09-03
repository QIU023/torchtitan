# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Under CP each Kimi K3 attention layer gets a kernel that owns its collectives."""

import unittest
from dataclasses import replace

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.kimi_k3 import kimi_k3_configs
from torchtitan.models.kimi_k3.context_parallel import (
    AllGatherCPFlexAttention,
    ContextParallelInnerKDA,
    ContextParallelKernel,
    UlyssesCPFlexAttention,
    use_kimi_k3_cp_kernels,
)
from torchtitan.models.kimi_k3.sharding import _set_cp_kernel_local_map


class TestKimiK3CPKernels(unittest.TestCase):
    def test_every_layer_gets_its_cp_kernel(self):
        config = kimi_k3_configs["debugmodel"]("flex", "standard")
        mla_layers = [l for l in config.layers if l.attention is not None]
        kda_layers = [l for l in config.layers if l.delta_attention is not None]
        self.assertTrue(mla_layers and kda_layers)
        # A non-default field must ride through the swap.
        mla_layers[0].attention.inner_attention = replace(
            mla_layers[0].attention.inner_attention, block_size=64
        )

        use_kimi_k3_cp_kernels(config)

        for layer in mla_layers:
            inner = layer.attention.inner_attention
            self.assertIsInstance(inner, UlyssesCPFlexAttention.Config)
            self.assertTrue(issubclass(UlyssesCPFlexAttention, ContextParallelKernel))
        self.assertEqual(mla_layers[0].attention.inner_attention.block_size, 64)
        for layer in kda_layers:
            inner = layer.delta_attention.inner_kda
            self.assertIsInstance(inner, ContextParallelInnerKDA.Config)
            self.assertEqual(inner.head_dim, 128)

    def test_all_gather_kernel_is_a_choice_and_sticks(self):
        config = kimi_k3_configs["debugmodel"]("flex", "standard")
        use_kimi_k3_cp_kernels(config, mla_kernel=AllGatherCPFlexAttention)
        mla = [l.attention for l in config.layers if l.attention is not None]
        for attention in mla:
            self.assertIsInstance(
                attention.inner_attention, AllGatherCPFlexAttention.Config
            )
        # The model's own update_from_config calls this again with the
        # default; a kernel already chosen is kept.
        use_kimi_k3_cp_kernels(config)
        for attention in mla:
            self.assertIsInstance(
                attention.inner_attention, AllGatherCPFlexAttention.Config
            )
        for layer in config.layers:
            if layer.delta_attention is not None:
                self.assertIsInstance(
                    layer.delta_attention.inner_kda, ContextParallelInnerKDA.Config
                )

    def test_ulysses_needs_flex_attention(self):
        config = kimi_k3_configs["debugmodel"]("flex", "standard")
        layer = next(l for l in config.layers if l.attention is not None)
        layer.attention.inner_attention = VarlenAttention.Config()
        with self.assertRaisesRegex(ValueError, "MLA kernels on FlexAttention"):
            use_kimi_k3_cp_kernels(config)

    def test_cp_kernel_boundary_keeps_tokens_sharded(self):
        # The kernel issues the exchange itself, so the local_map boundary is
        # an identity on the cp axis: tokens sharded in, tokens sharded out.
        cfg = FlexAttention.Config()
        _set_cp_kernel_local_map(cfg)
        sc = cfg.sharding_config
        for name in ("q_TNH", "k_TNH", "v_TNH"):
            for side in (sc.in_src_shardings, sc.in_dst_shardings):
                self.assertEqual(
                    _per_axis_types(side[name])[MeshAxisName.CP], spmd.S(0)
                )
                self.assertEqual(
                    _per_axis_types(side[name])[MeshAxisName.TP], spmd.S(1)
                )
        self.assertEqual(
            _per_axis_types(sc.out_src_shardings)[MeshAxisName.CP], spmd.S(0)
        )
        self.assertIsNone(sc.out_dst_shardings)


if __name__ == "__main__":
    unittest.main()
