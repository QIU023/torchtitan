# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CI smoke for the kimi_linear_debugmodel flavor.

Config build + a CPU forward/backward through the real module tree
(KDA fla CPU fallback, MLA SDPA, 8-expert MoE, Block AttnRes) in a few
seconds. The GPU train smoke lives in the launcher docs:
``--module kimi_k3 --config kimi_linear_debugmodel`` (10 steps).
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3 import config_registry


class TestKimiDebugModel(unittest.TestCase):
    def test_trainer_config_builds(self):
        cfg = config_registry.kimi_linear_debugmodel()
        self.assertEqual(cfg.model_spec.flavor, "kimi_linear_debugmodel")
        kimi = cfg.model_spec.model.kimi_config
        self.assertEqual(kimi.num_hidden_layers, 4)
        self.assertEqual(kimi.vocab_size, 2016)
        self.assertEqual(kimi.num_experts, 8)

    def test_cpu_forward_backward(self):
        cfg = config_registry.kimi_linear_debugmodel()
        model = cfg.model_spec.model.build()
        model.init_weights()
        tokens = torch.randint(0, 2016, (1, 32))
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (1, 32, 2016))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        # AttnRes projections get gradients (zero-init but on the path).
        for name, p in model.named_parameters():
            if name.endswith("attn_res_proj.weight"):
                self.assertIsNotNone(p.grad, f"no grad at {name}")
                break


if __name__ == "__main__":
    unittest.main()
