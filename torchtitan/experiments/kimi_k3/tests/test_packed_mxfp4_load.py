# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loading K3's packed-MXFP4 experts, without the 1.56 TB download.

The released checkpoint is 1.561 TB and stores routed experts as
``.weight_packed`` + ``.weight_scale``. The load path can still be exercised
completely: build a synthetic checkpoint in the OFFICIAL key naming and byte
layout at k3mini scale, push it through the same key map and dequantizer a real
load would use, and check the experts end up holding the right values.

The byte-layout claim is validated against torchao rather than against our own
packer, which would be circular: torchao packs, we decode, and the result must
match torchao's own dequantize exactly.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.experiments.kimi_k3.hf_key_map import official_to_titan
from torchtitan.experiments.kimi_k3.model import KimiLinearModel
from torchtitan.experiments.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.experiments.kimi_k3.packed_mxfp4 import (
    dequantize_mxfp4,
    load_packed_experts,
    quantize_mxfp4,
)

_KDA = {i for i in range(21) if (i + 1) not in {4, 8, 12, 16, 20, 21}}


class TestMXFP4ByteLayout(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "torchao MX needs CUDA")
    def test_our_decoder_matches_torchao_on_torchao_bytes(self):
        """The decisive check: not "our packer round-trips" (circular) but "we
        read bytes produced by an independent packer". A swapped nibble order
        would pass a round-trip and fail here."""
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        torch.manual_seed(0)
        w = (torch.randn(16, 128) * 0.2).cuda().bfloat16()
        mx = MXTensor.to_mx(w, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)
        ours = dequantize_mxfp4(
            mx.qdata, mx.scale.view(torch.uint8), dtype=torch.float32
        )
        theirs = mx.dequantize().float()
        self.assertEqual(
            ((ours - theirs).norm() / theirs.norm()).item(), 0.0
        )

    def test_shapes_follow_the_released_layout(self):
        w = torch.randn(16, 128)
        packed, scale = quantize_mxfp4(w)
        self.assertEqual(packed.shape, (16, 64))  # two nibbles per byte
        self.assertEqual(scale.shape, (16, 4))  # one byte per 32 values
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scale.dtype, torch.uint8)

    def test_round_trip_error_is_in_the_4_bit_band(self):
        torch.manual_seed(0)
        w = torch.randn(16, 128) * 0.2
        back = dequantize_mxfp4(*quantize_mxfp4(w), dtype=torch.float32)
        rel = ((back - w).norm() / w.norm()).item()
        # 4 bits with 2 mantissa bits on Gaussian data; torchao measures ~0.117
        self.assertGreater(rel, 0.05)
        self.assertLess(rel, 0.20)

    def test_zero_scale_byte_means_zero_not_a_tiny_power_of_two(self):
        packed = torch.zeros(1, 16, dtype=torch.uint8)
        scale = torch.zeros(1, 1, dtype=torch.uint8)
        out = dequantize_mxfp4(packed, scale, dtype=torch.float32)
        self.assertTrue(torch.all(out == 0.0))

    def test_mismatched_scale_group_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "groups"):
            dequantize_mxfp4(
                torch.zeros(4, 64, dtype=torch.uint8),
                torch.zeros(4, 3, dtype=torch.uint8),
            )

    def test_non_uint8_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            dequantize_mxfp4(
                torch.zeros(4, 64, dtype=torch.int8),
                torch.zeros(4, 2, dtype=torch.uint8),
            )


class TestSyntheticOfficialCheckpointLoad(unittest.TestCase):
    """A whole-model load, driven by official key strings."""

    def _model(self):
        cfg = build_kimi_linear_config("k3mini", vocab_size=256)
        with torch.device("meta"):
            m = KimiLinearModel(cfg)
        m.to_empty(device="cpu")
        m.init_weights()
        return m, cfg

    @staticmethod
    def _first_moe_layer(model) -> str:
        # layer 0 is dense (first_k_dense_replace), so its ffn has no _moe
        for name, layer in model.layers.items():
            if hasattr(layer.ffn, "_moe"):
                return name
        raise AssertionError("k3mini must have a MoE layer")

    def test_official_keys_drive_a_complete_expert_load(self):
        model, cfg = self._model()
        layer_idx = int(self._first_moe_layer(model))
        experts = model.layers[str(layer_idx)].ffn._moe.routed_experts.inner_experts

        # Build the synthetic checkpoint slice with OFFICIAL key names.
        torch.manual_seed(0)
        truth, tensors = {}, {}
        for w_official, our_name in (
            ("w1", "w1_EFD"), ("w2", "w2_EDF"), ("w3", "w3_EFD"),
        ):
            shape = experts._parameters[our_name].shape
            for e in range(cfg.num_experts):
                block = torch.randn(*shape[1:]) * 0.2
                packed, scale = quantize_mxfp4(block)
                base = (
                    f"language_model.model.layers.{layer_idx}."
                    f"block_sparse_moe.experts.{e}.{w_official}"
                )
                ours_p, kind_p = official_to_titan(
                    f"{base}.weight_packed", kda_layers=_KDA
                )
                ours_s, kind_s = official_to_titan(
                    f"{base}.weight_scale", kda_layers=_KDA
                )
                self.assertEqual((kind_p, kind_s), ("expert_packed", "expert_scale"))
                self.assertEqual(ours_p, ours_s)  # same destination, two parts
                tensors[ours_p.split(".")[-1]] = packed
                tensors[ours_s.split(".")[-1] + ":scale"] = scale
                truth[(our_name, e)] = dequantize_mxfp4(
                    packed, scale, dtype=torch.float32
                )

        written = load_packed_experts(
            experts, tensors, num_experts=cfg.num_experts, dtype=torch.float32
        )
        self.assertEqual(written, 3 * cfg.num_experts)

        for (name, e), expected in truth.items():
            got = experts._parameters[name][e]
            self.assertTrue(
                torch.equal(got, expected.to(got.dtype)),
                f"{name}[{e}] did not load exactly",
            )

    def test_a_missing_slice_refuses_the_load(self):
        """A partial load is worse than a failure: the unwritten experts keep
        their init values and the model still trains, which is the exact failure
        mode that cost this repo every recorded MoE loss."""
        model, cfg = self._model()
        layer_idx = int(self._first_moe_layer(model))
        experts = model.layers[str(layer_idx)].ffn._moe.routed_experts.inner_experts
        shape = experts._parameters["w1_EFD"].shape
        tensors = {}
        for e in range(cfg.num_experts - 1):  # deliberately one short
            packed, scale = quantize_mxfp4(torch.randn(*shape[1:]))
            tensors[f"w1_EFD[{e}]"] = packed
            tensors[f"w1_EFD[{e}]:scale"] = scale
        with self.assertRaisesRegex(KeyError, "partial load"):
            load_packed_experts(experts, tensors, num_experts=cfg.num_experts)

    def test_wrong_shape_is_rejected(self):
        model, cfg = self._model()
        layer_idx = int(self._first_moe_layer(model))
        experts = model.layers[str(layer_idx)].ffn._moe.routed_experts.inner_experts
        tensors = {}
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            for e in range(cfg.num_experts):
                packed, scale = quantize_mxfp4(torch.randn(8, 64))  # wrong
                tensors[f"{name}[{e}]"] = packed
                tensors[f"{name}[{e}]:scale"] = scale
        with self.assertRaisesRegex(ValueError, "expects"):
            load_packed_experts(experts, tensors, num_experts=cfg.num_experts)


if __name__ == "__main__":
    unittest.main()
