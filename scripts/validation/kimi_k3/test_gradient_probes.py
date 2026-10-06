# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from gradient_probes import changed_elements, select_probes


class GradientProbeTest(unittest.TestCase):
    def test_inactive_first_expert_does_not_hide_active_expert(self):
        parameter = torch.nn.Parameter(torch.ones(3, 8192, dtype=torch.bfloat16))
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[2, 5000] = 2
        self.assertEqual(torch.count_nonzero(parameter.grad.flatten()[:4096]), 0)
        probes = select_probes({"expert": [("experts", parameter)]}, scan_chunk=8192)
        self.assertGreater(probes[0]["norm"], 0)
        with torch.no_grad():
            parameter.sub_(parameter.grad * 0.5)
        self.assertEqual(changed_elements(probes), {"experts": 1})

    def test_searches_later_parameter_and_rejects_no_active_gradient(self):
        inactive = torch.nn.Parameter(torch.ones(16))
        inactive.grad = torch.zeros_like(inactive)
        active = torch.nn.Parameter(torch.ones(16))
        active.grad = torch.ones_like(active)
        probes = select_probes({"expert": [("inactive", inactive), ("active", active)]})
        self.assertEqual(probes[0]["name"], "active")
        with self.assertRaisesRegex(AssertionError, "No active"):
            select_probes({"expert": [("inactive", inactive)]})

    def test_rejects_nonfinite_gradient(self):
        parameter = torch.nn.Parameter(torch.ones(16))
        parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaisesRegex(AssertionError, "Nonfinite"):
            select_probes({"dense": [("dense", parameter)]})


if __name__ == "__main__":
    unittest.main()
