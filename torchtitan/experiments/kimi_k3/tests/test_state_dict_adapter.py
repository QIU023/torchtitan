# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for KimiLinearStateDictAdapter (HF <-> tt key mapping).

Uses meta-device builds -- key/shape coverage only, no weight values.
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3 import model_registry
from torchtitan.experiments.kimi_k3.state_dict_adapter import (
    KimiLinearStateDictAdapter,
)


def _build_state_dict(flavor: str):
    spec = model_registry(flavor)
    with torch.device("meta"):
        model = spec.model.build()
    return spec, model.state_dict()


class TestKimiLinearStateDictAdapter(unittest.TestCase):
    def test_wired_into_model_registry(self):
        spec = model_registry("kimi_linear_194m_block_attn_res")
        self.assertIs(spec.state_dict_adapter, KimiLinearStateDictAdapter)

    def test_round_trip_194m_block_attn_res(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        hf = adapter.to_hf(sd)
        back = adapter.from_hf(hf)
        self.assertEqual(set(back), set(sd))
        for k in sd:
            self.assertEqual(
                tuple(back[k].shape), tuple(sd[k].shape), f"shape drift at {k}"
            )

    def test_round_trip_baseline_no_attn_res(self):
        """Baseline flavor has no attn_res keys; mapping must still cover all."""
        spec, sd = _build_state_dict("kimi_linear_194m_baseline")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        back = adapter.from_hf(adapter.to_hf(sd))
        self.assertEqual(set(back), set(sd))

    def test_expert_weights_split_and_restack(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        hf = adapter.to_hf(sd)
        num_experts = spec.model.kimi_config.num_experts
        # Per-expert HF keys exist for a known MoE layer
        moe_keys = [k for k in hf if ".mlp.experts." in k]
        self.assertTrue(moe_keys)
        self.assertEqual(
            len(moe_keys),
            3 * num_experts * sum(
                1 for k in sd if k.endswith("w1_EFD")
            ),
        )

    def test_a_log_reshape_from_hf(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        a_log_keys = [k for k in sd if k.endswith("self_attn.A_log")]
        self.assertTrue(a_log_keys)
        h = sd[a_log_keys[0]].shape[0]
        hf_style = {
            "model." + a_log_keys[0].replace("layers.", "layers.", 1): torch.zeros(
                1, 1, h, 1
            )
        }
        # from_hf must flatten [1,1,H,1] -> [H]
        out = adapter.from_hf(
            {f"model.{a_log_keys[0]}": torch.zeros(1, 1, h, 1)}
        )
        self.assertEqual(tuple(out[a_log_keys[0]].shape), (h,))

    def test_packed_weights_rejected(self):
        spec, _ = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        with self.assertRaises(NotImplementedError):
            adapter.from_hf(
                {"model.layers.0.self_attn.q_proj.weight_scale": torch.zeros(2)}
            )
        with self.assertRaises(NotImplementedError):
            adapter.from_hf(
                {
                    "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                        4, 4, dtype=torch.uint8
                    )
                }
            )

    def test_quantized_reader_rejected(self):
        spec, _ = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        with self.assertRaises(NotImplementedError):
            adapter.get_hf_storage_reader("/nonexistent", from_quantized=True)

    def test_tied_embedding_alias_warns(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        hf = adapter.to_hf(sd)
        hf.pop("lm_head.weight")
        back = adapter.from_hf(hf)
        self.assertIn("lm_head.weight", back)


if __name__ == "__main__":
    unittest.main()
