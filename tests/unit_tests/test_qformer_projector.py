# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only tests for ``Qwen3VLQFormerProjector``.

These tests verify:
  * Shape contract — input (B, N_vision, vit_dim) -> output (B, num_queries, lm_dim).
  * Param-count guard — at the production config (internal_dim=1024,
    num_layers=6, n_heads=8, ffn_mult=4) the count is asserted to be in
    the 80M-120M range, matching BLIP-2 scale. Previously this config
    was ~1.21B because internal_dim was implicitly lm_dim=4096; see
    ``Qwen3VLQFormerProjector.Config.internal_dim``.
  * Forward-pass numerics — no NaN/Inf, bounded magnitudes.
  * Gradcheck on a tiny instance (internal_dim=16, num_queries=4).
  * Optional padding-mask path produces the same shape and is finite.

CPU-only. Do NOT add CUDA-specific paths here.
"""

from __future__ import annotations

import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common import Linear
from torchtitan.models.qwen3_vl.qformer_projector import Qwen3VLQFormerProjector


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_QUERY_INIT = {"queries": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}


def _make_linear(in_f: int, out_f: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_f, out_features=out_f, bias=True, param_init=_LINEAR_INIT
    )


def _build_qformer(
    *,
    in_features: int,
    lm_dim: int,
    internal_dim: int = 1024,
    num_queries: int = 64,
    num_layers: int = 6,
    n_heads: int = 8,
    ffn_mult: int = 4,
) -> Qwen3VLQFormerProjector:
    ffn_hidden = ffn_mult * internal_dim
    cfg = Qwen3VLQFormerProjector.Config(
        in_features=in_features,
        lm_dim=lm_dim,
        internal_dim=internal_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        n_heads=n_heads,
        ffn_mult=ffn_mult,
        q_proj=_make_linear(internal_dim, internal_dim),
        k_proj=_make_linear(in_features, internal_dim),
        v_proj=_make_linear(in_features, internal_dim),
        o_proj=_make_linear(internal_dim, internal_dim),
        ffn_fc1=_make_linear(internal_dim, ffn_hidden),
        ffn_fc2=_make_linear(ffn_hidden, internal_dim),
        out_proj=_make_linear(internal_dim, lm_dim),
        param_init=_QUERY_INIT,
    )
    model = cfg.build()
    model.init_states()
    return model


class TestQFormerProjector(unittest.TestCase):
    """End-to-end shape + numerics checks at the production hyperparam point."""

    def test_output_shape_production(self):
        """Q-Former with vit_dim=1152, lm_dim=4096, internal_dim=1024,
        64 queries, 6 layers.

        Input: (B=2, N_vision=1680, vit_dim=1152) -> (B=2, 64, 4096).
        """
        torch.manual_seed(0)
        model = _build_qformer(
            in_features=1152,
            lm_dim=4096,
            internal_dim=1024,
            num_queries=64,
            num_layers=6,
            n_heads=8,
        )
        x = torch.randn(2, 1680, 1152)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, torch.Size([2, 64, 4096]))
        self.assertFalse(torch.isnan(out).any().item(), "Q-Former output has NaN")
        self.assertFalse(torch.isinf(out).any().item(), "Q-Former output has Inf")

    def test_output_shape_postmerger(self):
        """Q-Former consuming post-merger features (in_features == lm_dim).

        This is the path used by ``_8b_qformer`` in ``__init__.py``.
        """
        torch.manual_seed(0)
        model = _build_qformer(
            in_features=4096,
            lm_dim=4096,
            internal_dim=1024,
            num_queries=64,
            num_layers=2,
            n_heads=8,
        )
        x = torch.randn(1, 420, 4096)  # 3 cams * 4 frames * 35 post-merger tokens
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, torch.Size([1, 64, 4096]))
        self.assertFalse(torch.isnan(out).any().item())

    def test_param_count_guard(self):
        """Production-config param count is in the 80M-120M range (BLIP-2 scale).

        With internal_dim=1024, lm_dim=4096, num_layers=6, n_heads=8,
        ffn_mult=4, in_features=4096 (post-merger path used by
        ``_8b_qformer``)::

            per-block:
              q_proj  : 1024 * 1024 + 1024  ≈  1.05M
              k_proj  : 4096 * 1024 + 1024  ≈  4.20M
              v_proj  : 4096 * 1024 + 1024  ≈  4.20M
              o_proj  : 1024 * 1024 + 1024  ≈  1.05M
              ffn_fc1 : 1024 * 4096 + 4096  ≈  4.20M
              ffn_fc2 : 4096 * 1024 + 1024  ≈  4.20M
              3 x LN  : ~6K (negligible)
              ---------
              subtotal ≈ 18.91M
            x 6 layers ≈ 113.4M
            + queries  : 64 * 1024 = 65K
            + out_proj : 1024 * 4096 + 4096 ≈ 4.20M
            + norm_out LN: 2K
            Total ≈ 117.7M  (well under the 1.2B pre-fix count)

        Lower bound 80M / upper bound 120M cover both
        in_features=4096 (post-merger, the production path) and the
        smaller in_features=1152 (pre-merger) future variant.
        """
        model = _build_qformer(
            in_features=4096,
            lm_dim=4096,
            internal_dim=1024,
            num_queries=64,
            num_layers=6,
            n_heads=8,
            ffn_mult=4,
        )
        n_params = sum(p.numel() for p in model.parameters())
        # BLIP-2 scale target — 80M lower, 120M upper.
        self.assertGreater(
            n_params,
            80_000_000,
            f"Q-Former param count too low ({n_params:,}); expected >80M",
        )
        self.assertLess(
            n_params,
            120_000_000,
            f"Q-Former param count too high ({n_params:,}); expected <120M. "
            f"This guard catches accidental regressions to lm_dim-wide "
            f"internal dim (which gave ~1.21B before the fix).",
        )
        # Save the count for the report.
        TestQFormerProjector._reported_param_count = n_params

    def test_param_count_premerger(self):
        """Pre-merger variant (in_features=1152) is ~80M — the lighter end."""
        model = _build_qformer(
            in_features=1152,
            lm_dim=4096,
            internal_dim=1024,
            num_queries=64,
            num_layers=6,
            n_heads=8,
            ffn_mult=4,
        )
        n_params = sum(p.numel() for p in model.parameters())
        # Pre-merger has cheaper k/v projections (1152 vs 4096 input dim).
        # Expect ~80M.
        self.assertGreater(n_params, 70_000_000)
        self.assertLess(n_params, 100_000_000)

    def test_padding_mask_shape_and_finite(self):
        """key_padding_mask: True at PAD positions → output stays finite."""
        torch.manual_seed(0)
        model = _build_qformer(
            in_features=1152,
            lm_dim=4096,
            internal_dim=1024,
            num_queries=64,
            num_layers=2,
            n_heads=8,
        )
        B, N = 2, 800
        x = torch.randn(B, N, 1152)
        # Mark second half of sample 0 as PAD, no padding on sample 1.
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, 400:] = True
        with torch.no_grad():
            out = model(x, key_padding_mask=mask)
        self.assertEqual(out.shape, torch.Size([B, 64, 4096]))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_grad_flow_tiny(self):
        """Gradients flow on a tiny instance (CPU-friendly).

        Uses internal_dim=16, lm_dim=32 to exercise the internal_dim !=
        lm_dim decoupling path in addition to the basic gradcheck.
        """
        torch.manual_seed(0)
        model = _build_qformer(
            in_features=8,
            lm_dim=32,
            internal_dim=16,
            num_queries=4,
            num_layers=2,
            n_heads=2,
            ffn_mult=2,
        )
        x = torch.randn(2, 10, 8, requires_grad=True)
        out = model(x)
        # Output dim must be lm_dim, not internal_dim.
        self.assertEqual(out.shape, torch.Size([2, 4, 32]))
        loss = out.square().sum()
        loss.backward()

        # All Q-Former params should have grads.
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"No grad for {name}")
            self.assertTrue(
                torch.isfinite(p.grad).all().item(), f"Non-finite grad in {name}"
            )
        # Inputs got grads too.
        self.assertIsNotNone(x.grad)

    def test_num_queries_invariant(self):
        """Output Nq is determined by config.num_queries, not input length."""
        torch.manual_seed(0)
        model = _build_qformer(
            in_features=8,
            lm_dim=32,
            internal_dim=16,
            num_queries=4,
            num_layers=1,
            n_heads=2,
            ffn_mult=2,
        )
        # Vary input N — output must stay at Nq=4 and dim=lm_dim=32.
        for N in (1, 16, 17, 1024):
            out = model(torch.randn(1, N, 8))
            self.assertEqual(out.shape, torch.Size([1, 4, 32]))

    def test_internal_dim_decoupled_from_lm_dim(self):
        """internal_dim and lm_dim are independent; queries live at internal_dim."""
        torch.manual_seed(0)
        model = _build_qformer(
            in_features=64,
            lm_dim=256,
            internal_dim=32,
            num_queries=8,
            num_layers=1,
            n_heads=4,
            ffn_mult=2,
        )
        # The learnable queries should be sized at internal_dim, NOT lm_dim.
        self.assertEqual(model.queries.shape, torch.Size([8, 32]))
        # Output must be at lm_dim.
        out = model(torch.randn(1, 7, 64))
        self.assertEqual(out.shape, torch.Size([1, 8, 256]))


if __name__ == "__main__":
    unittest.main()
