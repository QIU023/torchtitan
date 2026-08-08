# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chaining the tower's block shares must equal running the whole encoder.

Report 5.2.3 asks for vision forward and backward "balanced across PP stages", which
means the tower spans several stages and each carries a contiguous share of blocks.
This pins the arithmetic of that split BEFORE any pipeline wiring exists, because a
mismatch found later at pp4 would be indistinguishable from a PP plumbing bug.

Two properties are separate and both matter:

* **Chaining equals whole.** ``run_blocks`` over [0,k) then [k,n) with the final norm
  applied only at the end reproduces ``forward``.
* **Each share recomputes its own block inputs.** ``freqs_cis`` and ``cu_seqlens`` are
  derived per stage from that stage's ``grid_thws`` rather than sent over the pipe --
  PP's metadata inference pushes dummy values through pipe tensors, and these are used
  as RoPE indices and segment bounds where a dummy asserts out of bounds. So the test
  calls each share the way a stage would: with grid_thws and nothing else.

Run in fp32 with a tight tolerance. The only legitimate difference here is reduction
order, at 1e-6; a loose tolerance would pass while hiding a real defect, which has
already happened once in this model's history (see test_moonvit_dynamic_cp_tower).
"""

from __future__ import annotations

import unittest

import torch


def _encoder(num_layers: int):
    from torchtitan.models.kimi_k3.moonvit import MoonViTConfig, MoonViTEncoder

    cfg = MoonViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        qkv_hidden_size=32,
        num_hidden_layers=num_layers,
        patch_size=2,
        text_hidden_size=32,
        rope_max_grid=64,
    )
    torch.manual_seed(0)
    enc = MoonViTEncoder(cfg).to(torch.float32)
    enc.eval()
    return enc


class TestMoonViTStageSplit(unittest.TestCase):
    def _inputs(self, num_patches: int, dim: int = 32):
        torch.manual_seed(1)
        x = torch.randn(num_patches, dim, dtype=torch.float32)
        # One image, t=1, and h*w == num_patches so the segment bounds cover exactly
        # the tokens present.
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        return x, grid

    def test_two_shares_equal_whole(self):
        enc = _encoder(4)
        x, grid = self._inputs(16)

        with torch.no_grad():
            whole = enc(x, grid)
            first = enc.run_blocks(
                x, grid, block_slice=slice(0, 2), apply_final_norm=False
            )
            second = enc.run_blocks(
                first, grid, block_slice=slice(2, 4), apply_final_norm=True
            )

        torch.testing.assert_close(second, whole, rtol=1e-5, atol=1e-6)

    def test_four_shares_equal_whole(self):
        """One block per share -- the finest split, and the one most likely to expose
        a prologue that was only correct when computed once."""
        enc = _encoder(4)
        x, grid = self._inputs(16)

        with torch.no_grad():
            whole = enc(x, grid)
            h = x
            for i in range(4):
                h = enc.run_blocks(
                    h,
                    grid,
                    block_slice=slice(i, i + 1),
                    apply_final_norm=(i == 3),
                )

        torch.testing.assert_close(h, whole, rtol=1e-5, atol=1e-6)

    def test_final_norm_only_on_last_share(self):
        """Applying the norm on every share must NOT reproduce the whole encoder.

        A guard on the test itself: if it passed either way, it would not be testing
        where the norm goes.
        """
        enc = _encoder(4)
        x, grid = self._inputs(16)

        with torch.no_grad():
            whole = enc(x, grid)
            wrong = enc.run_blocks(
                x, grid, block_slice=slice(0, 2), apply_final_norm=True
            )
            wrong = enc.run_blocks(
                wrong, grid, block_slice=slice(2, 4), apply_final_norm=True
            )

        self.assertFalse(
            torch.allclose(wrong, whole, rtol=1e-5, atol=1e-6),
            "norm-on-every-share matched the whole encoder, so this test cannot "
            "detect a misplaced final norm",
        )

    def test_block_inputs_are_recomputable_per_share(self):
        """Same grid_thws -> same (freqs_cis, cu_seqlens), so a later stage can derive
        them locally instead of receiving them over the pipe."""
        enc = _encoder(2)
        x, grid = self._inputs(16)

        with torch.no_grad():
            f1, c1 = enc.block_inputs(x, grid)
            mid = enc.run_blocks(
                x, grid, block_slice=slice(0, 1), apply_final_norm=False
            )
            f2, c2 = enc.block_inputs(mid, grid)

        torch.testing.assert_close(f1, f2, rtol=0, atol=0)
        torch.testing.assert_close(c1, c2, rtol=0, atol=0)

    def test_gradients_flow_through_chained_shares(self):
        """The split has to be differentiable end to end, since the report balances
        vision BACKWARD passes too."""
        enc = _encoder(4)
        x, grid = self._inputs(16)
        x.requires_grad_(True)

        first = enc.run_blocks(x, grid, block_slice=slice(0, 2), apply_final_norm=False)
        second = enc.run_blocks(
            first, grid, block_slice=slice(2, 4), apply_final_norm=True
        )
        second.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        # Blocks in BOTH shares must have received gradient, or the chain silently
        # trained only one end.
        for share in (0, 3):
            weight = enc.blocks[share].wqkv.weight
            self.assertIsNotNone(weight.grad, f"block {share} got no gradient")
            self.assertTrue(weight.grad.abs().sum() > 0, f"block {share} grad is zero")


if __name__ == "__main__":
    unittest.main()
