# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT-V2 against the released vision_config and report sec 2.4."""

from __future__ import annotations

import json
import pathlib
import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.kimi_k3.moonvit import (
    MoonViT,
    MoonViTConfig,
    PatchMergerV2,
)

_ARTIFACT = (
    pathlib.Path(__file__).resolve().parents[5]
    / "phase13_k3like_48b_posttrain"
    / "official_k3"
    / "config.json"
)


def _tiny() -> MoonViTConfig:
    """Same structure, small enough for CPU. Extents only."""
    return MoonViTConfig(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        qkv_hidden_size=48,  # head_dim 24, still wider than hidden_size
        intermediate_size=64,
        patch_size=4,
        init_pos_emb_time=4,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        text_hidden_size=64,
    )


class TestMoonViTConfigMatchesOfficial(unittest.TestCase):
    def test_defaults_match_the_released_vision_config(self):
        if not _ARTIFACT.exists():
            self.skipTest("official artifact not present")
        v = json.loads(_ARTIFACT.read_text())["vision_config"]
        c = MoonViTConfig()
        for ours, theirs in (
            ("num_hidden_layers", "vt_num_hidden_layers"),
            ("hidden_size", "vt_hidden_size"),
            ("num_attention_heads", "vt_num_attention_heads"),
            ("intermediate_size", "vt_intermediate_size"),
            ("qkv_hidden_size", "qkv_hidden_size"),
            ("patch_size", "patch_size"),
            ("text_hidden_size", "text_hidden_size"),
            ("init_pos_emb_time", "init_pos_emb_time"),
            ("init_pos_emb_height", "init_pos_emb_height"),
            ("init_pos_emb_width", "init_pos_emb_width"),
            ("projector_ln_eps", "projector_ln_eps"),
        ):
            self.assertEqual(getattr(c, ours), v[theirs], ours)
        self.assertEqual(list(c.merge_kernel_size), v["merge_kernel_size"])
        self.assertEqual(c.pos_emb_interpolation_mode, v["pos_emb_interpolation_mode"])
        # head_dim follows from the two width fields, and is the value that
        # makes qkv_hidden_size != hidden_size make sense
        self.assertEqual(c.head_dim, 128)

    def test_parameter_count_matches_the_reported_0p4b(self):
        """The report says "roughly 0.4B parameters". That number is the
        evidence for shared spatial/temporal projections: separate per-pass
        projections would put the tower at ~0.57B."""
        model = MoonViT(MoonViTConfig())
        n = model.encoder_num_parameters()
        self.assertGreater(n, 3.6e8, f"{n/1e6:.1f}M is too small for 0.4B")
        self.assertLess(n, 4.4e8, f"{n/1e6:.1f}M is too large for 0.4B")

        # and the counterfactual: doubling the attention projections (one set
        # per pass) leaves the "roughly 0.4B" band
        c = MoonViTConfig()
        per_layer_attn = 3 * c.hidden_size * c.qkv_hidden_size + (
            c.qkv_hidden_size * c.hidden_size
        )
        unshared = n + c.num_hidden_layers * per_layer_attn
        self.assertGreater(unshared, 4.4e8)

    def test_no_biases_anywhere_in_the_tower(self):
        # report sec 2.4: "removes all bias terms from its linear and attention
        # projections". LayerNorm in the projector is the one exception, and it
        # is a norm, not a projection.
        model = MoonViT(_tiny())
        offenders = [
            name
            for name, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None
        ]
        self.assertEqual(offenders, [])

    def test_norms_are_rmsnorm_except_the_projector_layernorm(self):
        model = MoonViT(_tiny())
        for name, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not isinstance(m, nn.RMSNorm):
                self.assertTrue(
                    name.startswith("projector"),
                    f"{name} is a LayerNorm outside the projector",
                )


class TestMoonViTForward(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        m = MoonViT(_tiny())
        m.init_weights()
        return m

    def test_image_forward_shape_and_token_reduction(self):
        m = self._model()
        # 32x32 pixels at patch 4 -> an 8x8 patch grid -> 64 tokens, then the
        # 2x2 merge takes it to 16
        out = m(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, 1, 16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_bare_4d_input_is_treated_as_one_frame(self):
        m = self._model()
        pixels = torch.randn(1, 3, 32, 32)
        self.assertTrue(
            torch.equal(m(pixels), m(pixels.unsqueeze(1)))
        )

    def test_video_forward_pools_time(self):
        m = self._model()
        out = m(torch.randn(1, 4, 3, 32, 32))
        # 4 frames, temporal_pool_size 2 -> 2 output frames
        self.assertEqual(out.shape, (1, 2, 16, 64))

    def test_odd_frame_count_keeps_the_trailing_frame(self):
        m = self._model()
        out = m(torch.randn(1, 5, 3, 32, 32))
        # 5 frames -> 2 pooled pairs + 1 unpooled tail, not a silent drop
        self.assertEqual(out.shape[1], 3)

    def test_native_resolution_via_position_interpolation(self):
        """One weight set must serve many input sizes -- that is the point of
        interpolating a fixed table instead of learning per-resolution ones."""
        m = self._model()
        for hw in ((32, 32), (16, 16), (48, 24), (64, 32)):
            out = m(torch.randn(1, 3, *hw))
            expected_tokens = (hw[0] // 4 // 2) * (hw[1] // 4 // 2)
            self.assertEqual(out.shape[2], expected_tokens, str(hw))
            self.assertTrue(torch.isfinite(out).all(), str(hw))

    def test_temporal_pass_actually_mixes_frames(self):
        """Factorized attention is only factorized if the temporal pass runs;
        without it a video is a batch of independent images."""
        m = self._model()
        frames = torch.randn(1, 2, 3, 32, 32)
        together = m(frames)
        # Encode each frame alone, then apply the same pooling by hand.
        alone = torch.cat([m(frames[:, i : i + 1]) for i in range(2)], dim=1)
        pooled_alone = alone.mean(dim=1, keepdim=True)
        rel = (
            (together - pooled_alone).norm() / pooled_alone.norm()
        ).item()
        self.assertGreater(
            rel, 1e-3, "frames did not interact -- temporal pass is inert"
        )

    def test_spatial_merge_is_space_to_depth_not_pooling(self):
        """Pixel shuffle must preserve the 2x2 neighbourhood's content in
        channels; averaging would make two different neighbourhoods with the
        same mean indistinguishable."""
        cfg = _tiny()
        merger = PatchMergerV2(cfg)
        with torch.no_grad():
            merger.norm.weight.fill_(1.0)
            merger.norm.bias.zero_()
            nn.init.eye_(merger.fc1.weight)
            nn.init.normal_(merger.fc2.weight, std=0.05)
        a = torch.zeros(1, 1, 2, 2, cfg.hidden_size)
        a[0, 0, 0, 0] = 1.0
        b = torch.zeros(1, 1, 2, 2, cfg.hidden_size)
        b[0, 0, 1, 1] = 1.0  # same mean, different position
        self.assertFalse(torch.allclose(merger(a), merger(b)))

    def test_grid_indivisible_by_the_merge_kernel_is_rejected(self):
        m = self._model()
        # 28x32 at patch 4 -> a 7x8 grid; 7 is odd, so the 2x2 merge cannot
        # tile it. Fail loudly rather than dropping a row.
        with self.assertRaisesRegex(ValueError, "merge kernel"):
            m(torch.randn(1, 3, 28, 32))

    def test_gradients_reach_the_position_tables(self):
        m = self._model()
        m(torch.randn(1, 2, 3, 32, 32)).sum().backward()
        for name in ("time", "spatial"):
            g = getattr(m.pos_emb, name).grad
            self.assertIsNotNone(g, name)
            self.assertTrue(torch.isfinite(g).all(), name)
            self.assertGreater(g.abs().sum().item(), 0.0, name)


if __name__ == "__main__":
    unittest.main()
