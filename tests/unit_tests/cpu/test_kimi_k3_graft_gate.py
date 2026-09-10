# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The graft gate: alpha-gated attention-residual reads.

With every alpha at zero the gated model computes the plain backbone (the
standard residual stream), which is what preserves a pretrained backbone's
function at step 0 of a graft; a non-zero alpha moves the output, and the
alphas receive gradients.
"""

import unittest

import torch

from torchtitan.models.kimi_k3 import _apply_graft_gate, _kimi_linear_config
from torchtitan.models.kimi_k3.model import _gate_residual_read, KimiK3Model


def _mla_only_config(gated: bool) -> KimiK3Model.Config:
    """A Kimi-Linear-shaped model of three MLA layers, so it runs on CPU
    (the KDA kernels need CUDA); two-layer residual blocks."""
    config = _kimi_linear_config(
        dim=64,
        vocab_size=64,
        num_layers=3,
        full_attention_layers={0, 1, 2},
        attn_res_block_size=2,
        num_heads=2,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=128,
        expert_hidden_dim=32,
        num_experts=4,
        top_k=2,
        num_shared_experts=1,
        route_scale=2.446,
        attn_backend="flex",
    )
    return _apply_graft_gate(config) if gated else config


def _build(config: KimiK3Model.Config, device: torch.device) -> KimiK3Model:
    torch.manual_seed(0)
    with torch.device(device):
        model = config.build()
    model.init_states(buffer_device=device)
    return model


def _plain_backbone(model: KimiK3Model, tokens, masks, positions) -> torch.Tensor:
    """The standard residual stream through the same modules, in the op order
    the gated read's plain stream accumulates in."""
    h = model.tok_embeddings(tokens)
    for layer in model.layers.values():
        a = layer.attention_norm(h)
        if layer.attention is not None:
            a = layer.attention(a, masks, positions)
        else:
            assert layer.delta_attention is not None
            a = layer.delta_attention(a, None, positions)
        h = h + a
        f = layer.ffn_norm(h)
        f = layer.moe(f) if layer.moe is not None else layer.feed_forward(f)
        h = h + f
    return model.lm_head(model.norm(h))


class TestGraftGate(unittest.TestCase):
    def _inputs(self, model: KimiK3Model, device: torch.device):
        tokens = torch.randint(0, 64, (16,), device=device)
        positions = torch.arange(16, device=device)
        masks = model.get_attention_masks(
            positions=positions,
            padding_mask=None,
            max_num_documents=1,
            max_context_length=16,
        )
        return tokens, positions, masks

    def test_alphas_start_at_zero_and_the_plain_model_has_none(self):
        gated = _build(_mla_only_config(gated=True), torch.device("cpu"))
        alphas = {n: p for n, p in gated.named_parameters() if n.endswith("alpha")}
        self.assertEqual(len(alphas), 2 * 3 + 1)
        for value in alphas.values():
            self.assertEqual(value.item(), 0.0)
        plain = _build(_mla_only_config(gated=False), torch.device("cpu"))
        self.assertEqual([n for n, _ in plain.named_parameters() if "alpha" in n], [])

    def test_alpha_zero_is_the_plain_backbone(self):
        """Flex attention has no CPU backward, so the forward runs without grad."""
        device = torch.device("cpu")
        gated = _build(_mla_only_config(gated=True), device)
        tokens, positions, masks = self._inputs(gated, device)
        with torch.no_grad():
            out = gated(tokens, positions=positions, attention_masks=masks)
            reference = _plain_backbone(gated, tokens, masks, positions)
            self.assertTrue(torch.equal(out, reference))

            # The ungated reads are the attention-residual mixes, a different
            # function from the plain backbone.
            ungated = _build(_mla_only_config(gated=False), device)
            ungated.load_state_dict(
                {k: v for k, v in gated.state_dict().items() if "alpha" not in k}
            )
            out_ungated = ungated(tokens, positions=positions, attention_masks=masks)
            self.assertFalse(torch.equal(out_ungated, reference))

            # A non-zero alpha moves the output.
            gated.layers["1"].ffn_res_alpha.fill_(0.5)
            moved = gated(tokens, positions=positions, attention_masks=masks)
            self.assertFalse(torch.equal(moved, reference))

    def test_alpha_receives_the_read_minus_plain_gradient(self):
        alpha = torch.zeros(1, requires_grad=True)
        read = torch.randn(4, 8)
        plain = torch.randn(4, 8)
        _gate_residual_read(read, plain, alpha).sum().backward()
        assert alpha.grad is not None
        torch.testing.assert_close(alpha.grad, (read - plain).sum().reshape(1))

    @unittest.skipUnless(torch.cuda.is_available(), "backward needs CUDA")
    def test_model_gradients_reach_every_alpha(self):
        device = torch.device("cuda")
        gated = _build(_mla_only_config(gated=True), device)
        tokens, positions, masks = self._inputs(gated, device)
        with torch.no_grad():
            out = gated(tokens, positions=positions, attention_masks=masks)
            reference = _plain_backbone(gated, tokens, masks, positions)
        self.assertTrue(torch.equal(out, reference))
        out = gated(tokens, positions=positions, attention_masks=masks)
        out.float().square().mean().backward()
        for name, param in gated.named_parameters():
            if name.endswith("alpha"):
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
