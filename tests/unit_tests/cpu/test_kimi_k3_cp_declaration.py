# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ulysses is the MLA inner attention's declaration: token shard in, head shard out."""

import unittest

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.kimi_k3.sharding import _set_ulysses_inner_attention


class TestUlyssesDeclaration(unittest.TestCase):
    def test_cp_axis_trades_tokens_for_heads(self):
        cfg = FlexAttention.Config()
        _set_ulysses_inner_attention(cfg)
        sc = cfg.sharding_config
        for name in ("q_TNH", "k_TNH", "v_TNH"):
            self.assertEqual(
                _per_axis_types(sc.in_src_shardings[name])[MeshAxisName.CP], spmd.S(0)
            )
            self.assertEqual(
                _per_axis_types(sc.in_dst_shardings[name])[MeshAxisName.CP], spmd.S(1)
            )
            # TP keeps its head shard on both sides of the boundary.
            self.assertEqual(
                _per_axis_types(sc.in_dst_shardings[name])[MeshAxisName.TP], spmd.S(1)
            )
        self.assertEqual(
            _per_axis_types(sc.out_src_shardings)[MeshAxisName.CP], spmd.S(1)
        )
        self.assertEqual(
            _per_axis_types(sc.out_dst_shardings)[MeshAxisName.CP], spmd.S(0)
        )


if __name__ == "__main__":
    unittest.main()
