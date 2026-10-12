# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline_memory import enable_backward_resharding, RecomputeMatmulsSelectiveAC
from torch import nn
from torchtitan.models.base.module import Module


class FakeFSDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(3.0))
        self.reshard = False
        self.observed = []
        self.weight.register_hook(self.observe)

    def observe(self, grad):
        self.observed.append(self.reshard)
        return grad

    def set_reshard_after_backward(self, enabled):
        self.reshard = enabled

    def forward(self, x):
        return self.weight * x


class PipelineMemoryTests(unittest.TestCase):
    def test_hook_runs_after_pipeline_override_and_preserves_gradient(self):
        model = FakeFSDP()
        with patch("pipeline_memory.FSDPModule", FakeFSDP):
            handles = enable_backward_resharding([model])
        y = model(torch.tensor(2.0, requires_grad=True))
        model.set_reshard_after_backward(False)
        y.backward()
        self.assertEqual(model.observed, [True])
        self.assertEqual(model.weight.grad.item(), 2.0)
        handles[0].remove()
        model.zero_grad(set_to_none=True)
        y = model(torch.tensor(2.0, requires_grad=True))
        model.set_reshard_after_backward(False)
        y.backward()
        self.assertEqual(model.observed, [True, False])
        self.assertEqual(model.weight.grad.item(), 2.0)

    def test_rejects_non_fsdp_before_installing_hooks(self):
        model = FakeFSDP()
        with patch("pipeline_memory.FSDPModule", FakeFSDP):
            with self.assertRaises(TypeError):
                enable_backward_resharding([model, nn.Linear(1, 1)])
        self.assertEqual(len(model._backward_pre_hooks), 0)

    def test_policy_builds_and_retains_communication(self):
        policy = RecomputeMatmulsSelectiveAC.Config().build()
        self.assertIsInstance(policy, RecomputeMatmulsSelectiveAC)

        def recomputed(qualified):
            module = SimpleNamespace(
                remat_region_name=lambda local: qualified,
                _remat_save_patterns=policy.config.save_regions,
                _remat_recompute_patterns=policy.config.recompute_regions,
            )
            return Module.remat_should_recompute(module, "")

        self.assertTrue(recomputed("layers.16::attention.wq_b.linear"))
        self.assertTrue(recomputed("layers.1::moe.routed_experts.w13.grouped_mm"))
        self.assertFalse(recomputed("layers.1::moe.routed_experts.w2.grouped_mm"))
        self.assertFalse(recomputed("layers.0::attention.wq_b.tp_gather"))
        self.assertFalse(recomputed("layers.1::moe.token_dispatcher.dispatch"))


if __name__ == "__main__":
    unittest.main()
