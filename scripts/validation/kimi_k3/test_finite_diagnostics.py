# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path

import torch

from finite_diagnostics import check_updated_state, install


class FiniteDiagnosticsTest(unittest.TestCase):
    def test_finite_corrupted_parameter_is_rejected(self):
        model = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight[0, 0] = 4.36e35
        with tempfile.TemporaryDirectory() as output:
            report = check_updated_state([model], [], output, 0, 1)
        self.assertTrue(report["finite"])
        self.assertFalse(report["bounded"])
        self.assertEqual(report["failures"][0]["name"], "weight")
        self.assertGreater(report["failures"][0]["max_abs"], 1e35)

    def test_parameters_and_momentum_distinguish_nan_and_infinity(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.state[model.weight]["momentum_buffer"] = torch.ones_like(model.weight)
        with tempfile.TemporaryDirectory() as output:
            report = check_updated_state([model], [optimizer], output, 0, 1)
            self.assertTrue(report["finite"])
            self.assertEqual((report["parameters"], report["momentum_buffers"]), (2, 1))
            with torch.no_grad():
                model.bias[0] = float("nan")
                optimizer.state[model.weight]["momentum_buffer"][0, 0] = float("inf")
            report = check_updated_state([model], [optimizer], output, 0, 2)
            self.assertFalse(report["finite"])
            self.assertEqual(
                [(r["name"], r["nan"], r["posinf"]) for r in report["failures"]],
                [("bias", 1, 0), ("weight", 0, 1)],
            )
            rows = [
                json.loads(x)
                for x in Path(output, "finite-diagnostics-rank-0.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([r["step"] for r in rows], [1, 2])

    def test_noncontiguous_empty_and_scalar_parameters(self):
        model = torch.nn.Module()
        model.matrix = torch.nn.Parameter(torch.ones(3, 2).t())
        model.empty = torch.nn.Parameter(torch.empty(0))
        model.scalar = torch.nn.Parameter(torch.tensor(float("-inf")))
        with tempfile.TemporaryDirectory() as output:
            report = check_updated_state([model], [], output, 0, 1)
        self.assertEqual(report["parameters"], 3)
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["failures"][0]["neginf"], 1)

    def test_hooks_remain_active_and_identify_second_forward(self):
        model = torch.nn.Linear(2, 1, bias=False)
        step = 1
        with tempfile.TemporaryDirectory() as output:
            handles = install([model], output, 0, step=lambda: step)
            model(torch.ones(1, 2))
            step = 2
            with self.assertRaisesRegex(AssertionError, "First nonfinite output"):
                model(torch.full((1, 2), float("nan")))
            rows = [
                json.loads(x)
                for x in Path(output, "finite-diagnostics-rank-0.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(rows[-1]["step"], 2)
            self.assertEqual(rows[-1]["phase"], "first_nonfinite_output")
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    unittest.main()
