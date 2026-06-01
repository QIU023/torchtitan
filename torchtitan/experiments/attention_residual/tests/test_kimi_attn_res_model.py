# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU smoke tests for KimiLinearAttnResModel.

Exercises the AttnRes weave end-to-end on a tiny config with dense
FFN only (no MoE, to dodge the ``torch.histc(Long)`` CPU limitation
in torchtitan's router). MoE path lives on GPU — see
``test_layers.py::TestKimiDeltaAttention::test_forward_shape_chunk_mode_cuda``
for the fla-core + CUDA side, and Phase 4b will add a full GPU model
integration test once the adapter training run frees the box.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.experiments.attention_residual.kimi_linear.attn_res_model import (
    KimiAttnResDecoderLayer,
    KimiLinearAttnResModel,
)
from torchtitan.experiments.attention_residual.kimi_linear.model import KimiLinearConfig


def _dense_mla_only_config(num_hidden_layers: int = 4) -> KimiLinearConfig:
    """Small config: all MLA (no KDA), all dense FFN (no MoE). KDA
    requires CUDA/Triton and MoE CPU forward hits a torch.histc Long
    limitation, so both are skipped here. The AttnRes weave itself is
    independent of which attention / FFN variant sits below it.
    """
    return KimiLinearConfig(
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
        # No KDA layers; all layers will fall back to MLA via is_mla property.
        kda_layers=[],
        full_attn_layers=list(range(1, num_hidden_layers + 1)),
        # MoE disabled
        num_experts=None,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=num_hidden_layers,  # all dense
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
    )


class TestKimiAttnResDecoderLayer(unittest.TestCase):
    def test_attn_res_params_present(self):
        cfg = _dense_mla_only_config()
        layer = KimiAttnResDecoderLayer(cfg, layer_idx=0)
        names = {n for n, _ in layer.named_children()}
        for expected in (
            "attn_res_proj",
            "mlp_res_proj",
            "attn_res_norm",
            "mlp_res_norm",
            "self_attn",
            "ffn",
            "input_layernorm",
            "post_attention_layernorm",
        ):
            self.assertIn(expected, names)

    def test_forward_threads_blocks_and_partial(self):
        cfg = _dense_mla_only_config()
        layer = KimiAttnResDecoderLayer(cfg, layer_idx=0)

        B, T, D = 2, 16, cfg.hidden_size
        blocks = [torch.randn(B, T, D) for _ in range(2)]
        partial = torch.randn(B, T, D)

        new_blocks, new_partial = layer(blocks, partial, is_block_start=True)
        # On block start, partial is committed into blocks -> +1 entry.
        self.assertEqual(len(new_blocks), 3)
        self.assertEqual(new_partial.shape, (B, T, D))

        # Non-block-start: blocks unchanged, partial accumulates.
        new_blocks2, new_partial2 = layer(new_blocks, new_partial, is_block_start=False)
        self.assertEqual(len(new_blocks2), 3)
        self.assertEqual(new_partial2.shape, (B, T, D))


class TestKimiLinearAttnResModel(unittest.TestCase):
    def test_instantiate_full_attnres(self):
        """Full AttnRes: num_blocks == num_hidden_layers, one block per layer."""
        cfg = _dense_mla_only_config(num_hidden_layers=4)
        model = KimiLinearAttnResModel(cfg, num_blocks=4)
        self.assertEqual(model.num_blocks, 4)
        self.assertEqual(model.layers_per_block, 1)
        # Pseudo-queries init to zero per paper.
        model.init_weights()
        for layer in model.layers.values():
            self.assertTrue(torch.all(layer.attn_res_proj.weight == 0))
            self.assertTrue(torch.all(layer.mlp_res_proj.weight == 0))
        self.assertTrue(torch.all(model.final_attn_res_proj.weight == 0))

    def test_instantiate_block_attnres(self):
        """Block AttnRes N=2: 4 layers, 2 blocks, 2 layers per block."""
        cfg = _dense_mla_only_config(num_hidden_layers=4)
        model = KimiLinearAttnResModel(cfg, num_blocks=2)
        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.layers_per_block, 2)

    def test_forward_cpu_dense_only(self):
        """End-to-end forward on CPU with MLA + dense FFN only.

        Initial loss should be finite and roughly ``log(vocab_size)``
        because pseudo-queries are zero-init (initial AttnRes softmax
        is uniform -> equivalent to standard residuals) and the model
        is freshly initialized.
        """
        cfg = _dense_mla_only_config(num_hidden_layers=4)
        torch.manual_seed(0)
        model = KimiLinearAttnResModel(cfg, num_blocks=2)
        model.init_weights()

        B, T = 2, 8
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(tokens)
        self.assertEqual(logits.shape, (B, T, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

        # Loss at init: log(vocab_size) for uniform output distribution.
        # With random init it won't be exactly uniform, but finite.
        import math

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size),
            tokens.view(-1),
        )
        self.assertTrue(torch.isfinite(loss).all())
        # Sanity: CE loss should be at most a few times log(vocab_size).
        self.assertLess(loss.item(), 10 * math.log(cfg.vocab_size))

    def test_rejects_non_divisible_num_blocks(self):
        cfg = _dense_mla_only_config(num_hidden_layers=5)  # prime
        with self.assertRaises(AssertionError):
            # 5 layers, 2 blocks -> not divisible -> reject
            KimiLinearAttnResModel(cfg, num_blocks=2)


if __name__ == "__main__":
    unittest.main()
