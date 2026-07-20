# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""(Per-Head) Muon optimizer tests.

Locks the base Muon algorithm (published; K3's exact per-head variant
reconciles at 7.27): Newton-Schulz equalizes singular values, Muon
optimizes matrices while non-2-D params take the AdamW fallback, and
the per-head path orthogonalizes head blocks independently.
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3.muon import _newton_schulz, Muon


@unittest.skipIf(not torch.cuda.is_available(), "bf16 NS on CUDA")
class TestMuon(unittest.TestCase):
    def test_newton_schulz_equalizes_singular_values(self):
        torch.manual_seed(0)
        G = torch.randn(128, 64, device="cuda")
        Q = _newton_schulz(G, steps=5)
        # Muon's NS pushes singular values toward 1 -> condition number
        # (max/min sigma) drops sharply vs the raw Gaussian matrix.
        cond_in = torch.linalg.svdvals(G.float())
        cond_out = torch.linalg.svdvals(Q.float())
        r_in = (cond_in.max() / cond_in.min()).item()
        r_out = (cond_out.max() / cond_out.min()).item()
        self.assertLess(r_out, r_in)
        self.assertLess(r_out, 3.0)  # near-orthogonal

    def test_muon_matrix_adamw_fallback(self):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(64, 32, device="cuda"))
        b = torch.nn.Parameter(torch.ones(64, device="cuda"))
        target = torch.randn(64, 32, device="cuda")
        opt = Muon([W, b], lr=0.05, adamw_lr=0.02)
        first = last = None
        for i in range(60):
            loss = (W - target).pow(2).mean() + b.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        self.assertLess(last, first)
        # AdamW fallback drove the bias vector toward 0.
        self.assertLess(b.abs().mean().item(), 1.0)

    def test_per_head_path(self):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(128, 32, device="cuda"))
        W._muon_heads = 4
        opt = Muon([W], lr=0.05, per_head=True)
        first = W.pow(2).mean().item()
        for _ in range(20):
            loss = W.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(W.pow(2).mean().item(), first)


if __name__ == "__main__":
    unittest.main()
