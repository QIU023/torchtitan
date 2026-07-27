# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated MLA (K3 delta, provisional) tests.

Gated MLA is graft-viable via near-identity gate init.
This locks that a checkpoint pretrained WITHOUT gated MLA is ~preserved
at step 0 when the gate is enabled (near-identity, not bit-exact -- the
sigmoid(6)=0.9975 leak distinguishes it from the alpha graft gate which
IS bit-exact). Exact gate form is PROVISIONAL, reconciles at 7.27.
"""

import dataclasses
import unittest

import torch

from torchtitan.experiments.kimi_k3.model import KimiLinearConfig, KimiLinearModel


def _cfg():
    return KimiLinearConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=2016, intermediate_size=512,
        moe_intermediate_size=256, num_experts=8, kv_lora_rank=128,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        kda_head_dim=64, kda_num_heads=4,
        kda_layers=[1], full_attn_layers=[2],
    )


@unittest.skipIf(not torch.cuda.is_available(), "KDA needs CUDA (fla triton)")
class TestGatedMLA(unittest.TestCase):
    def test_near_identity_at_init_and_grad(self):
        torch.manual_seed(0)
        cfg = _cfg()
        with torch.device("cuda"):
            plain = KimiLinearModel(cfg)
            plain.init_weights()
            gated = KimiLinearModel(dataclasses.replace(cfg, mla_gated=True))
            gated.init_weights()
        gated.load_state_dict(plain.state_dict(), strict=False)
        tok = torch.randint(0, 2016, (2, 96), device="cuda")
        plain.eval()
        gated.eval()
        with torch.no_grad():
            lp = plain(tok).float()
            lg = gated(tok).float()
        # near-identity: relative (scale-invariant) is robust to
        # random-init amplification; the sigmoid(6) gate leak keeps it
        # small but NON-zero (not bit-exact, unlike the alpha gate).
        rel = ((lp - lg).norm() / lp.norm()).item()
        self.assertLess(rel, 2e-2)
        self.assertGreater(rel, 0.0)
        # gate trains
        gated.train()
        gated(tok).float().sum().backward()
        gp = dict(gated.named_parameters())
        gk = [k for k in gp if k.endswith("attn_gate_proj.weight")]
        self.assertTrue(gk)
        for k in gk:
            self.assertIsNotNone(gp[k].grad, k)


if __name__ == "__main__":
    unittest.main()
