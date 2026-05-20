# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only tests for ``Qwen3VLPixelShufflePlusLinearProjector``.

These tests verify:
  * Shape contract — input ``(B, N_vision, in_features)`` + ``grid_thw``
    -> output ``(B, N_vision / shuffle_unit, lm_dim)``.
  * Param-count sanity — at the pre-merger path (in_features=1152,
    shuffle_ratio=2, lm_dim=4096) the projector has ~19M params (matches
    the spec's "~17M" target ballpark).
  * Numerics — no NaN / Inf on a random input.
  * Multi-frame (t > 1) case is handled by flattening temporal into batch.
  * Edge case: PixelShuffle requires even (h, w); odd dims must raise.

CPU-only. Do NOT add CUDA-specific paths here.
"""

from __future__ import annotations

import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common import Linear
from torchtitan.models.qwen3_vl.pixelshuffle_projector import (
    Qwen3VLPixelShufflePlusLinearProjector,
)


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}


def _make_linear(in_f: int, out_f: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_f, out_features=out_f, bias=True, param_init=_LINEAR_INIT
    )


def _build_projector(
    *,
    in_features: int,
    lm_dim: int,
    shuffle_ratio: int = 2,
) -> Qwen3VLPixelShufflePlusLinearProjector:
    proj_in = in_features * (shuffle_ratio ** 2)
    cfg = Qwen3VLPixelShufflePlusLinearProjector.Config(
        in_features=in_features,
        lm_dim=lm_dim,
        shuffle_ratio=shuffle_ratio,
        proj=_make_linear(proj_in, lm_dim),
    )
    model = cfg.build()
    model.init_states()
    return model


class TestPixelShuffleProjector(unittest.TestCase):
    """Shape + numerics + param-count contract at the production hyperparam point."""

    def test_output_shape_single_image_20x20(self):
        """Single-image / single-frame, grid 20x20 at vit_dim=1152.

        Input  : (B=2, N=400, in_features=1152)
        grid_thw : [[1, 20, 20], [1, 20, 20]]
        Expected output : (2, 100, 4096)  (400 / 4 = 100)
        """
        torch.manual_seed(0)
        model = _build_projector(in_features=1152, lm_dim=4096, shuffle_ratio=2)
        x = torch.randn(2, 400, 1152)
        grid_thw = torch.tensor([[1, 20, 20], [1, 20, 20]], dtype=torch.long)
        with torch.no_grad():
            out = model(x, grid_thw=grid_thw)
        self.assertEqual(out.shape, torch.Size([2, 100, 4096]))
        self.assertFalse(torch.isnan(out).any().item(), "Output has NaN")
        self.assertFalse(torch.isinf(out).any().item(), "Output has Inf")

    def test_output_shape_3cam_4f_1680(self):
        """Multi-cam case: 1680 tokens / item -> 420 tokens / item.

        Mocks the 3-cam x 4-frame nuScenes planning input where one
        visual item carries (t=4, h=14, w=30) = 1680 patches at the raw
        ViT grid (before the in-encoder merger). Using shuffle_ratio=2:
            output_per_item = 4 * (14/2) * (30/2) = 4 * 7 * 15 = 420
        """
        torch.manual_seed(0)
        model = _build_projector(in_features=1152, lm_dim=4096, shuffle_ratio=2)
        x = torch.randn(2, 1680, 1152)
        grid_thw = torch.tensor([[4, 14, 30], [4, 14, 30]], dtype=torch.long)
        with torch.no_grad():
            out = model(x, grid_thw=grid_thw)
        self.assertEqual(out.shape, torch.Size([2, 420, 4096]))
        self.assertFalse(torch.isnan(out).any().item())

    def test_output_shape_postmerger_path(self):
        """Post-merger consumption (in_features == lm_dim).

        This is the wiring path used by ``_8b_pixelshuffle()``: the
        in-encoder PatchMerger has already projected to lm_dim, and the
        PixelShuffle projector compresses further by 4x. ``(h, w)`` here
        is the **post-merger** grid (raw grid / spatial_merge_size = 2).

        Picks (t=1, h=14, w=20) so the post-merger grid is even on both
        sides (PixelShuffle 2x requires even h, w).
        """
        torch.manual_seed(0)
        model = _build_projector(in_features=4096, lm_dim=4096, shuffle_ratio=2)
        N = 1 * 14 * 20  # 280
        x = torch.randn(1, N, 4096)
        grid_thw = torch.tensor([[1, 14, 20]], dtype=torch.long)
        with torch.no_grad():
            out = model(x, grid_thw=grid_thw)
        self.assertEqual(out.shape, torch.Size([1, N // 4, 4096]))
        self.assertFalse(torch.isnan(out).any().item())

    def test_param_count_premerger_path(self):
        """At the pre-merger path the projector has ~19M params.

        ``Linear(4 * 1152, 4096)`` with bias has:
            weight: 4608 * 4096 = 18_874_368
            bias  : 4096
            total : 18_878_464  ~= 19M
        """
        model = _build_projector(in_features=1152, lm_dim=4096, shuffle_ratio=2)
        n_params = sum(p.numel() for p in model.parameters())
        # 17M lower bound is the spec's stated target; the exact value is
        # 18.88M for in_features=1152.
        self.assertGreater(n_params, 17_000_000)
        # Generous upper bound — guard against accidental dim doubling.
        self.assertLess(n_params, 25_000_000)
        # Save for reporting.
        TestPixelShuffleProjector._param_count_premerger = n_params

    def test_param_count_postmerger_path(self):
        """At the post-merger path the projector has ~67M params.

        ``Linear(4 * 4096, 4096)`` with bias has:
            weight: 16384 * 4096 = 67_108_864
            bias  : 4096
            total : 67_112_960  ~= 67M
        """
        model = _build_projector(in_features=4096, lm_dim=4096, shuffle_ratio=2)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 60_000_000)
        self.assertLess(n_params, 75_000_000)
        TestPixelShuffleProjector._param_count_postmerger = n_params

    def test_grad_flow_tiny(self):
        """Gradients flow on a tiny instance (CPU-friendly)."""
        torch.manual_seed(0)
        model = _build_projector(in_features=8, lm_dim=16, shuffle_ratio=2)
        x = torch.randn(2, 16, 8, requires_grad=True)  # grid 4x4
        grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
        out = model(x, grid_thw=grid_thw)
        self.assertEqual(out.shape, torch.Size([2, 4, 16]))
        loss = out.square().sum()
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"No grad for {name}")
            self.assertTrue(
                torch.isfinite(p.grad).all().item(), f"Non-finite grad in {name}"
            )
        self.assertIsNotNone(x.grad)

    def test_odd_grid_raises(self):
        """PixelShuffle 2x on an odd grid dim must raise ValueError."""
        model = _build_projector(in_features=4, lm_dim=8, shuffle_ratio=2)
        # h=3 is odd -> must raise. We need N >= t*h*w = 1*3*4 = 12.
        x = torch.randn(1, 12, 4)
        grid_thw = torch.tensor([[1, 3, 4]], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x, grid_thw=grid_thw)

    def test_padding_preserved_in_output_length(self):
        """Padded extra slots in input produce padded extra slots in output.

        The downstream ``_get_vision_embeds`` valid_mask drops these
        slots, so the projector just needs to keep ``N // shuffle_unit``
        as the new padded length.
        """
        torch.manual_seed(0)
        model = _build_projector(in_features=4, lm_dim=8, shuffle_ratio=2)
        # Valid: 1*4*4 = 16 tokens, padded to 24.
        N = 24
        x = torch.randn(1, N, 4)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
        with torch.no_grad():
            out = model(x, grid_thw=grid_thw)
        # Output padded length should be N // shuffle_unit = 24 // 4 = 6,
        # of which the first 4 (= 16 // 4) are valid and the last 2 are
        # zero-padded.
        self.assertEqual(out.shape, torch.Size([1, 6, 8]))
        # Tail padding should be exact zeros.
        self.assertTrue(torch.all(out[:, 4:, :] == 0).item())

    def test_mixed_grid_thw_raises(self):
        """Mixed (t, h, w) per item is not supported in a single call."""
        torch.manual_seed(0)
        model = _build_projector(in_features=4, lm_dim=8, shuffle_ratio=2)
        x = torch.randn(2, 16, 4)
        grid_thw = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x, grid_thw=grid_thw)


if __name__ == "__main__":
    unittest.main()
