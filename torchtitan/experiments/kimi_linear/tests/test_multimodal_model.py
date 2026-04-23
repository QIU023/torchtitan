# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU smoke tests for KimiLinearMultimodalModel scaffolding.

Phase 4e scope: verify the module layout + forward path compose
cleanly on CPU without requiring a real vision tower / pretrained
SigLIP or CLIP weights. A ``_DummyViT`` stand-in produces
deterministic features so we can validate projector shapes +
vision-feature injection without the network.

MoE path is avoided (CPU histc limitation); tests run with dense FFN
only (``first_k_dense_replace=n_layers``) so every layer uses
:class:`KimiMLP` not :class:`KimiMoE`.

Actual multimodal pretraining infra (data loader, image preprocessing,
loss masking, real ViT integration) is Phase 5 work.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.kimi_linear.model import KimiLinearConfig
from torchtitan.experiments.kimi_linear.multimodal_model import (
    KimiLinearMultimodalModel,
    KimiMultimodalConfig,
    KimiVisionProjector,
)


class _DummyViT(nn.Module):
    """Stand-in for SigLIP/CLIP: takes [B, C, H, W], returns
    [B, N_patches, D_vision]. Deterministic, parameter-free.
    """

    def __init__(self, *, num_patches: int, vision_hidden_size: int):
        super().__init__()
        self.num_patches = num_patches
        self.vision_hidden_size = vision_hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        # Deterministic placeholder: zero features of the right shape.
        # Tests validate shape / control flow, not feature fidelity.
        return torch.zeros(
            B, self.num_patches, self.vision_hidden_size,
            dtype=pixel_values.dtype, device=pixel_values.device,
        )


def _mm_cfg(num_hidden_layers: int = 2) -> KimiMultimodalConfig:
    kimi = KimiLinearConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        kda_num_heads=4,
        kda_head_dim=16,
        kda_short_conv_kernel_size=4,
        kda_layers=[],
        full_attn_layers=list(range(1, num_hidden_layers + 1)),
        num_experts=None,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=num_hidden_layers,
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
    )
    return KimiMultimodalConfig(
        kimi_config=kimi,
        num_blocks=None,
        vision_hidden_size=32,
        projector_hidden_size=64,
        vision_token_id=-200,
    )


class TestKimiVisionProjector(unittest.TestCase):
    def test_forward_shape(self):
        p = KimiVisionProjector(
            vision_hidden_size=32,
            projector_hidden_size=64,
            llm_hidden_size=128,
        )
        x = torch.randn(2, 7, 32)
        out = p(x)
        self.assertEqual(out.shape, (2, 7, 128))


class TestKimiLinearMultimodalModel(unittest.TestCase):
    def test_text_only_forward_when_vision_tower_none(self):
        cfg = _mm_cfg()
        model = KimiLinearMultimodalModel(cfg, vision_tower=None)
        input_ids = torch.randint(0, cfg.kimi_config.vocab_size, (2, 8))
        logits = model(input_ids=input_ids, pixel_values=None)
        self.assertEqual(
            logits.shape, (2, 8, cfg.kimi_config.vocab_size)
        )

    def test_vision_tower_frozen_by_default(self):
        cfg = _mm_cfg()
        vit = _DummyViT(num_patches=16, vision_hidden_size=32)
        model = KimiLinearMultimodalModel(cfg, vision_tower=vit)
        for p in model.vision_tower.parameters():
            self.assertFalse(p.requires_grad)

    def test_multimodal_forward_injects_vision(self):
        """With a dummy ViT + sentinel tokens, forward expands the
        sequence by num_patches per image and returns valid logits."""
        cfg = _mm_cfg()
        num_patches = 16
        vit = _DummyViT(
            num_patches=num_patches, vision_hidden_size=cfg.vision_hidden_size
        )
        model = KimiLinearMultimodalModel(cfg, vision_tower=vit)

        B, T = 2, 8
        V = cfg.vision_token_id
        # input_ids with ONE vision sentinel per sample at position 2:
        input_ids = torch.randint(1, cfg.kimi_config.vocab_size, (B, T))
        input_ids[:, 2] = V
        # 1 image per sample, 3 channel, 32×32 (placeholder).
        pixel_values = torch.randn(B, 1, 3, 32, 32)

        logits = model(input_ids=input_ids, pixel_values=pixel_values)
        # Expanded length: original T=8 + num_patches - 1 (sentinel replaced
        # by num_patches feature slots → +num_patches-1 per sample).
        expected_T = T + num_patches - 1
        self.assertEqual(
            logits.shape, (B, expected_T, cfg.kimi_config.vocab_size)
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_rejects_more_vision_tokens_than_images(self):
        cfg = _mm_cfg()
        vit = _DummyViT(num_patches=4, vision_hidden_size=cfg.vision_hidden_size)
        model = KimiLinearMultimodalModel(cfg, vision_tower=vit)
        # 2 vision sentinels but only 1 image → error
        input_ids = torch.randint(1, cfg.kimi_config.vocab_size, (1, 6))
        input_ids[0, 1] = cfg.vision_token_id
        input_ids[0, 4] = cfg.vision_token_id
        pixel_values = torch.randn(1, 1, 3, 32, 32)  # only 1 image
        with self.assertRaises(RuntimeError):
            model(input_ids=input_ids, pixel_values=pixel_values)


if __name__ == "__main__":
    unittest.main()
