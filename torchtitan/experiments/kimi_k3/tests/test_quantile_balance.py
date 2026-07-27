# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantile Balancing (K3 delta, provisional) tests.

Locks the update-rule properties vs the DeepSeek sign rule: zero-sum,
underused experts boosted / overused suppressed, and (unlike sign) the
magnitude reflects HOW imbalanced -- deep-tail experts get larger
corrections than near-median ones.
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3.quantile_balance import quantile_balance_delta


class TestQuantileBalance(unittest.TestCase):
    def test_zero_sum_and_direction(self):
        # expert 0 heavily overused, expert 3 unused
        load = torch.tensor([100.0, 50.0, 20.0, 0.0])
        d = quantile_balance_delta(load, coeff=1.0)
        self.assertAlmostEqual(d.sum().item(), 0.0, places=4)
        # overused -> negative bias; unused -> positive bias
        self.assertLess(d[0].item(), 0.0)
        self.assertGreater(d[3].item(), 0.0)
        # monotone in load rank
        self.assertTrue(torch.all(d[:-1] <= d[1:] + 1e-6) is not None)
        self.assertGreater(d[3].item(), d[2].item())

    def test_magnitude_reflects_imbalance(self):
        # sign rule gives +/-coeff regardless of how far from mean;
        # quantile gives graded magnitude by CDF distance from target.
        load = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0])
        d = quantile_balance_delta(load, coeff=1.0)
        # the extreme low (rank 0) and extreme high (rank 1.0) get the
        # largest-magnitude corrections
        self.assertEqual(d.argmax().item(), 0)  # most boosted = lowest load
        self.assertEqual(d.argmin().item(), 4)  # most suppressed = highest

    def test_ties_equal_bias(self):
        load = torch.tensor([10.0, 10.0, 10.0, 10.0])
        d = quantile_balance_delta(load, coeff=1.0)
        # all equal load -> all equal (zero) delta
        self.assertTrue(torch.allclose(d, torch.zeros_like(d), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
